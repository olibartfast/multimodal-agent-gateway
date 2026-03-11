"""API request/response handling for LLM and VLM providers."""

import json
import time

import requests

from multimodal_agent_gateway.image import encode_image, is_url
from multimodal_agent_gateway.models import Agent, AgentResult


def create_payload(
    prompt: str,
    image_paths: list[str] | None,
    model: str,
    detail: str,
    max_tokens: int,
    resize: bool = False,
    target_size: tuple[int, int] = (512, 512),
) -> dict:
    """
    Build an OpenAI-compatible chat-completions payload.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        "max_tokens": max_tokens,
    }
    for image_path in image_paths or []:
        if is_url(image_path) and not resize:
            img_block = {
                "type": "image_url",
                "image_url": {"url": image_path, "detail": detail},
            }
        else:
            b64 = encode_image(image_path, resize, target_size)
            img_block = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": detail},
            }
        payload["messages"][0]["content"].append(img_block)
    return payload


def build_video_payload(
    model: str,
    system_prompt: str,
    user_prompt: str,
    frame_b64_list: list[str],
    max_tokens: int = 1024,
    detail: str = "low",
) -> dict:
    """
    Build an OpenAI chat-completions payload with multiple base64 frames.

    The frames are sent as individual image_url content blocks.
    This is the de-facto standard for video-as-frames via the OpenAI API,
    supported natively by vLLM, SGLang, Together AI, and others.
    """
    content: list[dict] = [{"type": "text", "text": user_prompt}]

    for b64 in frame_b64_list:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": detail,
                },
            }
        )

    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        "max_tokens": max_tokens,
    }


def create_anthropic_payload(
    prompt: str,
    image_paths: list[str] | None,
    model: str,
    max_tokens: int,
    resize: bool = False,
    target_size: tuple[int, int] = (512, 512),
) -> dict:
    """
    Build an Anthropic Messages API payload with images.

    Uses Anthropic's native image format (base64 or URL source blocks)
    instead of OpenAI's image_url wrapper.
    """
    content: list[dict] = [{"type": "text", "text": prompt}]
    for image_path in image_paths or []:
        if is_url(image_path) and not resize:
            img_block = {
                "type": "image",
                "source": {"type": "url", "url": image_path},
            }
        else:
            b64 = encode_image(image_path, resize, target_size)
            img_block = {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": b64,
                },
            }
        content.append(img_block)
    return {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": content}],
    }


def build_anthropic_video_payload(
    model: str,
    system_prompt: str,
    user_prompt: str,
    frame_b64_list: list[str],
    max_tokens: int = 1024,
) -> dict:
    """
    Build an Anthropic Messages API payload for multi-frame video analysis.

    System prompt is sent as a top-level field (Anthropic convention).
    Frames are encoded as base64 image source blocks.
    """
    content: list[dict] = [{"type": "text", "text": user_prompt}]
    for b64 in frame_b64_list:
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": b64,
                },
            }
        )
    return {
        "model": model,
        "system": system_prompt,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": content}],
    }


def send_request(
    api_key: str,
    url: str,
    payload: dict,
    timeout: int = 120,
) -> dict:
    """POST to an OpenAI-compatible endpoint and return the JSON response."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(f"API request failed: {response.status_code} {response.text}")
    return response.json()


def send_anthropic_request(
    api_key: str,
    url: str,
    payload: dict,
    timeout: int = 120,
) -> dict:
    """POST to the Anthropic Messages API using native auth headers."""
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(f"Anthropic API request failed: {response.status_code} {response.text}")
    return response.json()


def normalize_response(response: dict, provider: str = "openai") -> str:
    """
    Extract text content from any supported provider response shape.

    Supports:
    - OpenAI-compatible: choices[0].message.content
    - Anthropic: content[0].text
    - Google: candidates[0].content.parts[0].text
    """
    try:
        return response["choices"][0]["message"]["content"]  # OpenAI-compatible
    except (KeyError, IndexError):
        pass
    try:
        return response["content"][0]["text"]  # Anthropic
    except (KeyError, IndexError):
        pass
    try:
        return response["candidates"][0]["content"]["parts"][0]["text"]  # Google
    except (KeyError, IndexError):
        pass
    return json.dumps(response)


def run_agent(
    agent: Agent,
    prompt: str,
    image_paths: list[str] | None,
    detail: str,
    max_tokens: int,
    resize: bool,
    target_size: tuple[int, int],
) -> AgentResult:
    """Execute a single agent call and return an AgentResult."""
    if agent.provider == "anthropic":
        payload = create_anthropic_payload(prompt, image_paths, agent.model, max_tokens, resize, target_size)
    else:
        payload = create_payload(prompt, image_paths, agent.model, detail, max_tokens, resize, target_size)
    t0 = time.time()
    try:
        if agent.provider == "anthropic":
            response = send_anthropic_request(agent.api_key, agent.endpoint, payload)
        else:
            response = send_request(agent.api_key, agent.endpoint, payload)
        latency_ms = (time.time() - t0) * 1000
        return AgentResult(
            agent_id=agent.agent_id,
            model=agent.model,
            provider=agent.provider,
            content=normalize_response(response, agent.provider),
            raw_response=response,
            latency_ms=latency_ms,
        )
    except Exception as exc:
        return AgentResult(
            agent_id=agent.agent_id,
            model=agent.model,
            provider=agent.provider,
            content="",
            raw_response={},
            latency_ms=(time.time() - t0) * 1000,
            error=str(exc),
        )
