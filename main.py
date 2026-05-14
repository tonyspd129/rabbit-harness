#!/usr/bin/env python3
"""
Rabbit Harness entry point.

Usage (inside the sandbox container):
    python3 /opt/rabbit-harness/main.py \\
        --prompt "Read SKILL.md and complete the task." \\
        --model  "qwen/qwen3.5-35b-a3b" \\
        [--provider     AlibabaCloud] \\
        [--temperature  0.0] \\
        [--seed         42] \\
        [--max-tokens   8192] \\
        [--max-turns    40] \\
        [--turns-out    /workspace/turns.jsonl]

Environment:
    OPENROUTER_API_KEY  (or OPENAI_API_KEY as fallback)
"""

import argparse
import os
import sys
from pathlib import Path

# Allow `import loop` etc. when invoked as `python3 /opt/rabbit-harness/main.py`
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from loop import run  # noqa: E402 — must come after sys.path patch


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Custom ReAct agent harness")
    p.add_argument("--prompt",      required=True,  help="Initial user prompt")
    p.add_argument("--model",       required=True,  help="Model ID (e.g. qwen/qwen3.5-35b-a3b)")
    p.add_argument("--provider",    default="AlibabaCloud", help="OpenRouter provider to pin")
    p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    p.add_argument("--seed",        type=int,   default=None, help="RNG seed (optional)")
    p.add_argument("--max-tokens",  type=int,   default=8192, dest="max_tokens")
    p.add_argument("--max-turns",   type=int,   default=40,   dest="max_turns")
    p.add_argument(
        "--turns-out",
        default="/workspace/turns.jsonl",
        dest="turns_out",
        help="Path to write turns.jsonl",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    api_key = (
        os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    )
    if not api_key:
        print(
            "[harness] ERROR: no API key found in OPENROUTER_API_KEY / OPENAI_API_KEY",
            file=sys.stderr,
        )
        return 1

    print(
        f"[harness] model={args.model} provider={args.provider} "
        f"temperature={args.temperature} max_turns={args.max_turns}",
        file=sys.stderr,
    )

    try:
        run(
            prompt=args.prompt,
            model=args.model,
            provider=args.provider,
            api_key=api_key,
            temperature=args.temperature,
            seed=args.seed,
            max_tokens=args.max_tokens,
            max_turns=args.max_turns,
            turns_out=args.turns_out,
        )
    except Exception as exc:
        print(f"[harness] FATAL: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1

    print(f"[harness] done, turns written to {args.turns_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
