#!/usr/bin/env python3
"""
ReAct loop — synchronous, no asyncio required.

Recovery behaviour:
  - prefill_retries counter resets to 0 on EVERY successful tool-call turn
  - MAX_PREFILL=5 thinking-only retries per streak
  - Empty-content nudge: up to 3 retries with an explicit continuation prompt
"""

import sys

from client import OpenRouterClient
from session import Session
from tools import TOOLS_SCHEMA, execute_tool

MAX_PREFILL = 5
MAX_EMPTY_RETRIES = 3

_NUDGE_MSG = (
    "Please continue. If you have finished, provide your final answer. "
    "If you need to use a tool, call it now."
)


def run(
    prompt: str,
    model: str,
    provider: str,
    api_key: str,
    temperature: float = 0.0,
    seed: int | None = None,
    max_tokens: int = 8192,
    max_turns: int = 40,
    turns_out: str = "/workspace/turns.jsonl",
) -> Session:

    client = OpenRouterClient(
        api_key=api_key,
        model=model,
        provider=provider,
        temperature=temperature,
        seed=seed,
        max_tokens=max_tokens,
    )
    session = Session(model=model)
    session.add_user(prompt)

    prefill_retries = 0
    empty_retries = 0

    for turn_n in range(max_turns):
        print(f"[harness] turn {turn_n + 1}", file=sys.stderr)

        result = client.complete(session.messages, TOOLS_SCHEMA)

        # --- Tool calls (structured or leak-recovered) ---
        if result.tool_calls:
            prefill_retries = 0
            empty_retries = 0
            session.add_assistant(result)
            for tc in result.tool_calls:
                print(
                    f"[harness]   tool={tc['name']} args={list(tc['args'].keys())}",
                    file=sys.stderr,
                )
                out = execute_tool(tc["name"], tc["args"])
                session.add_tool_result(tc["id"], tc["name"], out)
            continue

        # --- Thinking-only turn (reasoning but no content, no tools) ---
        if result.is_thinking_only and prefill_retries < MAX_PREFILL:
            prefill_retries += 1
            print(
                f"[harness]   thinking-only turn, prefill retry {prefill_retries}/{MAX_PREFILL}",
                file=sys.stderr,
            )
            session.add_prefill(result)
            continue

        # --- Completely empty turn (no content, no reasoning, no tools) ---
        if not result.content and not result.tool_calls and not result.reasoning:
            if empty_retries < MAX_EMPTY_RETRIES:
                empty_retries += 1
                print(
                    f"[harness]   empty response, nudge retry {empty_retries}/{MAX_EMPTY_RETRIES}",
                    file=sys.stderr,
                )
                session.add_assistant(result)
                session.add_user(_NUDGE_MSG)
                continue

        # --- Final response ---
        print(
            f"[harness]   final response (finish_reason={result.finish_reason})",
            file=sys.stderr,
        )
        session.add_assistant(result)
        break

    session.export_jsonl(turns_out)
    return session
