#!/usr/bin/env python3
"""
Fall Detection Example

Demonstrates using the VLM Agent Gateway for fall detection monitoring.
Works with local vLLM, Together AI, or OpenAI.

Usage:
    # With Together AI (Qwen2.5-VL-72B)
    TOGETHER_API_KEY=... python examples/fall_detection.py ./elderly_room.mp4

    # With local vLLM serving Qwen2.5-VL-7B
    python examples/fall_detection.py ./elderly_room.mp4 \
        --endpoint http://localhost:8000/v1/chat/completions \
        --model Qwen/Qwen2.5-VL-7B-Instruct

    # Continuous webcam monitoring
    python examples/fall_detection.py 0 --continuous
"""

import argparse
import sys

from vlm_agent_gateway import run_monitoring
from vlm_agent_gateway.config import get_api_key, get_default_endpoint


def main():
    parser = argparse.ArgumentParser(description="Fall detection with VLM monitoring")
    parser.add_argument("video", help="Video file, RTSP URL, or device index (0 for webcam)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-72B-Instruct")
    parser.add_argument("--provider", default="together")
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--interval", type=float, default=15.0)
    args = parser.parse_args()

    endpoint = args.endpoint or get_default_endpoint(args.provider)

    try:
        api_key = get_api_key(args.provider)
    except RuntimeError:
        if "localhost" in endpoint or "127.0.0.1" in endpoint:
            api_key = "EMPTY"
        else:
            print(f"Error: Set {args.provider.upper()}_API_KEY environment variable")
            sys.exit(1)

    alert_prompt = (
        "Is anyone falling, lying on the floor, collapsed, or showing signs of distress? "
        "Pay attention to sudden changes in posture, people on the ground, or unusual positions."
    )

    result = run_monitoring(
        video_source=args.video,
        endpoint=endpoint,
        api_key=api_key,
        model=args.model,
        alert_prompt=alert_prompt,
        fps=args.fps,
        max_frames=args.max_frames,
        continuous=args.continuous,
        interval_seconds=args.interval,
    )

    if not args.continuous:
        print(f"\nResult: {'ALERT' if result.get('alert') else 'OK'}")
        print(f"Summary: {result.get('summary')}")


if __name__ == "__main__":
    main()
