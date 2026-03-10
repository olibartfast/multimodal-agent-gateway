# VLMs for Video Understanding & Agentic Monitoring

A guide to video-capable Vision Language Models and how to build monitoring
applications driven by agentic AI — using cloud APIs and self-hosted inference
(vLLM / SGLang).

---

## Video-Capable VLM Landscape (2025)

### Tier 1 — Frontier Video VLMs

| Model | Params | Video Support | Key Strengths | Access |
|-------|--------|--------------|---------------|--------|
| **Qwen3-VL** | 8B / 235B-A22B (MoE) | Native video + dynamic FPS | 3D grounding, visual agent, code gen from video | HF, vLLM, DashScope API |
| **Qwen2.5-VL** | 3B / 7B / 32B / 72B | Native video, 1hr+ length | Event pinpointing, absolute time encoding, structured output | HF, vLLM, Together AI API |
| **GLM-4.1V-Thinking** | 9B | Native video, time-index tokens | Reasoning-first (RLCS), competitive with 72B models | HF, vLLM, Zhipu API |
| **GLM-4.5V** | 106B-A12B (MoE) | Native video | SOTA open-source on 42 benchmarks, 128K context | HF, vLLM |
| **GPT-4o / 4o-mini** | Closed | Frames as images | Broad capability, widely available | OpenAI API |
| **Gemini 2.x Flash/Pro** | Closed | Native video upload | Long video (up to hours), audio track analysis | Google API |

### Tier 2 — Established Video VLMs

| Model | Params | Notes |
|-------|--------|-------|
| **LLaVA-Video** (formerly LLaVA-NeXT-Video) | 7B / 72B | Trained on LLaVA-Video-178K synthetic dataset, strong on VideoMME |
| **LLaVA-OneVision** | 0.5B / 7B / 72B | Unified image + multi-image + video, good zero-shot transfer |
| **InternVL 2.5** | 1B–78B | MPO-optimized, strong multimodal reasoning |
| **Phi-3.5-Vision** | 4.2B | Compact, multi-frame support, good for edge |

### How Video Input Works

All these models process video through one of two mechanisms:

**Native video path** — The model's preprocessor handles frame extraction internally.
Qwen2.5-VL, Qwen3-VL, and GLM-4.1V-Thinking support this natively through
transformers or vLLM, with configurable FPS and resolution constraints.

**Frames-as-images** — You extract frames yourself and send them as multiple
`image_url` content blocks in a single chat-completions request. This is the
universal approach that works with any OpenAI-compatible API (GPT-4o, vLLM,
Together AI, Groq, etc.) and is what our example script uses.

```
Video → [Frame Extractor] → JPEG frames → [base64 encode] → API request
         (OpenCV/ffmpeg)                    (multiple image_url blocks)
```

---

## Self-Hosted Deployment with vLLM

vLLM is the recommended serving engine. It exposes an OpenAI-compatible API
and supports all the models listed above.

### Quick Start: Qwen2.5-VL-7B

```bash
pip install vllm>=0.7.2 qwen-vl-utils[decord]==0.0.8

vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
    --limit-mm-per-prompt image=16 video=1 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9
```

### Quick Start: GLM-4.1V-9B-Thinking

```bash
vllm serve zai-org/GLM-4.1V-9B-Thinking \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code
```

### Quick Start: Qwen3-VL-8B

```bash
pip install qwen-vl-utils==0.0.14

vllm serve Qwen/Qwen3-VL-8B-Instruct \
    --limit-mm-per-prompt image=16 video=1 \
    --max-model-len 128000 \
    --async-scheduling
```

### Hardware Guidelines

| Model | Min GPU | Recommended | Notes |
|-------|---------|-------------|-------|
| Qwen2.5-VL-3B | RTX 3060 12GB | RTX 4070 | Edge-suitable |
| Qwen2.5-VL-7B | RTX 3090 24GB | A100 40GB | Good quality/cost balance |
| GLM-4.1V-9B-Thinking | RTX 3090 24GB | A100 40GB | Best 9B-class reasoning |
| Qwen2.5-VL-72B | 4× A100 80GB | 4× H100 | Production quality |
| Qwen3-VL-235B (MoE) | 8× A100 80GB | 8× H100/H200 | Flagship |

> **Tip for your RTX 3060:** Qwen2.5-VL-3B with `--max-model-len 4096` and
> AWQ quantization fits comfortably. Use `--limit-mm-per-prompt image=4 video=0`
> to keep memory under control.

---

## Building a Monitoring Application

The `examples/video_monitoring_agent.py` script implements a complete
monitoring pipeline:

```
┌─────────────┐     ┌───────────────┐     ┌──────────────┐     ┌─────────────┐
│ Video Source │────▶│ Frame Extract │────▶│  VLM Agent   │────▶│   Alerting  │
│ file/webcam/ │     │ OpenCV @ FPS  │     │ ReAct-style  │     │ console/log │
│ RTSP stream  │     │ → base64 JPEG │     │ analysis     │     │ webhook/etc │
└─────────────┘     └───────────────┘     └──────────────┘     └─────────────┘
```

### Architecture

1. **Frame Extraction** — OpenCV captures frames at a configurable FPS rate.
   For a 30fps source with `--fps 1`, we sample 1 frame per second. Frames
   are JPEG-compressed and base64-encoded.

2. **VLM Analysis** — Frames are sent as a batch to the VLM with a structured
   system prompt. The agent must output a specific format:
   `Thought → Alert (YES/NO) → Summary → Confidence → Recommended Action`.

3. **Alert Routing** — The parsed response is dispatched to alert handlers
   (console, JSONL file). Extend with webhooks, MQTT, email, etc.

4. **Continuous Mode** — With `--continuous`, the script loops: capture a
   window of N frames → analyze → wait → repeat. Designed for live sources.

### Example: Fall Detection

```bash
# Single video file analysis
python examples/video_monitoring_agent.py \
    --video ./elderly_room_recording.mp4 \
    --provider together \
    --model Qwen/Qwen2.5-VL-72B-Instruct \
    --alert-prompt "Is anyone falling, lying on the floor, or in distress?" \
    --fps 1 --max-frames 30

# Continuous webcam monitoring with local vLLM
python examples/video_monitoring_agent.py \
    --video 0 \
    --provider openai \
    --endpoint http://localhost:8000/v1/chat/completions \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --alert-prompt "Is anyone falling or showing signs of distress?" \
    --fps 0.5 --continuous --interval 15 --window-frames 8 \
    --output-jsonl ./fall_alerts.jsonl
```

### Example: Security Monitoring

```bash
python examples/video_monitoring_agent.py \
    --video rtsp://camera.local:554/stream \
    --provider openai \
    --model gpt-4o \
    --alert-prompt "Has anyone entered the restricted zone marked by yellow tape?" \
    --fps 0.5 --continuous --interval 10
```

### Example: Industrial Safety

```bash
python examples/video_monitoring_agent.py \
    --video ./factory_floor.mp4 \
    --provider openai \
    --endpoint http://localhost:8000/v1/chat/completions \
    --model zai-org/GLM-4.1V-9B-Thinking \
    --alert-prompt "Is any worker not wearing a hard hat or safety vest?" \
    --fps 1 --max-frames 20
```

---

## Integration with vlm-agent-gateway Workflows

The monitoring agent is a standalone script, but it's designed to compose
with the gateway's workflow patterns:

### MoA (Mixture-of-Agents) for Higher Confidence

Run the same frames through multiple VLMs and aggregate:

```bash
# Step 1: Extract frames once, run through multiple models
# Step 2: Use the gateway's MoA mode with an aggregator

python vlm-agent-gateway/main.py \
    --workflow moa \
    --prompt "Is anyone in this scene falling or in distress?" \
    --images frame_001.jpg frame_002.jpg frame_003.jpg \
    --models Qwen/Qwen2.5-VL-72B-Instruct gpt-4o \
    --providers together openai \
    --endpoints https://api.together.xyz/v1/chat/completions https://api.openai.com/v1/chat/completions \
    --aggregator-model gpt-4o \
    --aggregator-provider openai
```

### Sequential for Multi-Stage Analysis

First detect → then classify → then assess severity:

```bash
python vlm-agent-gateway/main.py \
    --workflow sequential \
    --prompt "Analyze these surveillance frames for safety incidents" \
    --images frame_*.jpg \
    --models gpt-4o-mini gpt-4o \
    --providers openai openai
```

---

## Model Selection Guide

| Use Case | Recommended Model | Why |
|----------|-------------------|-----|
| **Quick prototyping** | GPT-4o via OpenAI API | No setup, reliable, good video understanding |
| **Cost-effective cloud** | Qwen2.5-VL-72B via Together AI | Strong performance, ~$0.60/M input tokens |
| **Best reasoning (self-hosted)** | GLM-4.1V-9B-Thinking on vLLM | 9B model that competes with 72B, chain-of-thought |
| **Edge / low-resource** | Qwen2.5-VL-3B + AWQ on vLLM | Fits on 12GB GPU, still capable |
| **Long video (1hr+)** | Qwen2.5-VL-72B or Gemini 2.x | Dynamic FPS + absolute time encoding |
| **Maximum accuracy** | Qwen3-VL-235B or GLM-4.5V | Frontier performance, needs multi-GPU |

---

## Key Considerations for Production

### Frame Sampling Strategy

- **Higher FPS** = better temporal resolution but more tokens and cost.
- **Typical monitoring**: 0.5–1 FPS is sufficient for fall detection, intrusion.
- **Fast-moving events** (traffic, sports): 2–5 FPS may be needed.
- Use `--detail low` to minimize token usage per frame (~85 tokens vs ~765 for high).

### Latency Budget

| Backend | Typical Latency (8 frames) | Notes |
|---------|---------------------------|-------|
| GPT-4o API | 3–8s | Consistent, no setup |
| Together AI (72B) | 5–15s | Variable under load |
| vLLM local (7B, A100) | 2–5s | Best for continuous monitoring |
| vLLM local (3B, RTX 3060) | 3–8s | Viable for edge |

### Cost Estimation

For continuous monitoring at 1 cycle/10s with 8 frames/cycle:
- **GPT-4o**: ~$0.30–0.60/hour (low detail)
- **Together AI (72B)**: ~$0.10–0.20/hour
- **Self-hosted**: GPU cost only (amortized)

### Extending the Alert Pipeline

The `AlertEvent` dataclass and handler pattern are designed for extension:

```python
# Webhook handler example
def alert_handler_webhook(event: AlertEvent, url: str) -> None:
    if event.alert:
        requests.post(url, json=event.__dict__, timeout=5)

# MQTT handler for IoT integration
def alert_handler_mqtt(event: AlertEvent, topic: str, client) -> None:
    if event.alert:
        client.publish(topic, json.dumps(event.__dict__))

# Home Assistant integration
def alert_handler_ha(event: AlertEvent, ha_url: str, token: str) -> None:
    if event.alert:
        requests.post(
            f"{ha_url}/api/services/notify/notify",
            headers={"Authorization": f"Bearer {token}"},
            json={"message": f"🚨 {event.summary}", "title": "Video Alert"},
        )
```

---

## Further Reading

- [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923)
- [Qwen3-VL GitHub](https://github.com/QwenLM/Qwen3-VL)
- [GLM-4.1V-Thinking Paper](https://arxiv.org/abs/2507.01006)
- [GLM-V GitHub](https://github.com/zai-org/GLM-V)
- [LLaVA-Video Blog](https://llava-vl.github.io/blog/2024-09-30-llava-video/)
- [LLaVA-NeXT GitHub](https://github.com/LLaVA-VL/LLaVA-NeXT)
- [vLLM VLM Serving Guide](https://docs.vllm.ai/en/stable/models/vlm.html)
- [vLLM Qwen3-VL Recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html)
- [NVIDIA Jetson VLM Agent (edge deployment)](https://developer.nvidia.com/blog/develop-generative-ai-powered-visual-ai-agents-for-the-edge/)
