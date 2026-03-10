#!/usr/bin/env python3
"""
Security Monitoring Example

Demonstrates using the VLM Agent Gateway for security camera monitoring.
Monitors for intrusion, unauthorized access, or suspicious activity.

Usage:
    # Monitor RTSP camera stream
    python examples/security_monitoring.py rtsp://camera.local:554/stream

    # Monitor video file
    python examples/security_monitoring.py ./parking_lot.mp4

    # Continuous webcam monitoring with GPT-4o
    OPENAI_API_KEY=... python examples/security_monitoring.py 0 \
        --provider openai --model gpt-4o --continuous
"""

import argparse
import sys

from vlm_agent_gateway import run_monitoring
from vlm_agent_gateway.config import get_api_key, get_default_endpoint


def main():
    parser = argparse.ArgumentParser(description="Security monitoring with VLM")
    parser.add_argument("video", help="Video file, RTSP URL, or device index")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--zone", default="the monitored area", help="Description of restricted zone")
    parser.add_argument("--fps", type=float, default=0.5)
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--output", default=None, help="Output JSONL file for alerts")
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
        f"Has anyone entered or is attempting to enter {args.zone}? "
        "Look for: people approaching, climbing, reaching through barriers, "
        "or any unauthorized presence. Also watch for suspicious objects or packages."
    )

    result = run_monitoring(
        video_source=args.video,
        endpoint=endpoint,
        api_key=api_key,
        model=args.model,
        alert_prompt=alert_prompt,
        fps=args.fps,
        continuous=args.continuous,
        interval_seconds=args.interval,
        output_jsonl=args.output,
    )

    if not args.continuous:
        print(f"\nResult: {'ALERT' if result.get('alert') else 'OK'}")
        print(f"Summary: {result.get('summary')}")


if __name__ == "__main__":
    main()
