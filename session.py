#!/usr/bin/env python3
"""
Conversation state and turns.jsonl export.

The final line of turns.jsonl contains {"actual_cost_usd": N} so that
sandbox_harness._parse_session_cost() can read it unchanged.
"""

import json
from pathlib import Path

# Approximate pricing for provider-pinned calls via OpenRouter.
# Format: model_id -> (prompt_price_per_M, completion_price_per_M)
_PRICE_PER_M: dict = {
    "qwen/qwen3.5-35b-a3b": (0.14, 0.14),
    "qwen/qwen3.5-35b-a3b-20260224": (0.14, 0.14),
}


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    prices = _PRICE_PER_M.get(model)
    if not prices:
        # Strip provider prefix variants (e.g. "openrouter/qwen/...")
        for key, val in _PRICE_PER_M.items():
            if model.endswith(key):
                prices = val
                break
    if not prices:
        return 0.0
    p_price, c_price = prices
    return (prompt_tokens * p_price + completion_tokens * c_price) / 1_000_000


class Session:
    def __init__(self, model: str = ""):
        self.model = model
        self.messages: list[dict] = []
        self._turns: list[dict] = []          # one entry per LLM call, for jsonl
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_assistant(self, result) -> None:
        """Append a TurnResult as an assistant message."""
        msg: dict = {"role": "assistant"}
        if result.content:
            msg["content"] = result.content
        else:
            msg["content"] = None
        if result.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["args"]),
                    },
                }
                for tc in result.tool_calls
            ]
        self.messages.append(msg)
        self._record_turn(result)

    def add_prefill(self, result) -> None:
        """Append a thinking-only result as an interim assistant message."""
        msg: dict = {
            "role": "assistant",
            "content": None,
        }
        if result.reasoning:
            msg["_thinking_prefill"] = True
        self.messages.append(msg)
        self._record_turn(result)

    def add_tool_result(self, call_id: str, tool_name: str, output: str) -> None:
        self.messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "name": tool_name,
            "content": output,
        })

    def _record_turn(self, result) -> None:
        usage = result.usage or {}
        pt = int(usage.get("prompt_tokens") or 0)
        ct = int(usage.get("completion_tokens") or 0)
        self._total_prompt_tokens += pt
        self._total_completion_tokens += ct
        self._turns.append({
            "finish_reason": result.finish_reason,
            "tool_calls": len(result.tool_calls),
            "prompt_tokens": pt,
            "completion_tokens": ct,
        })

    def export_jsonl(self, path: str) -> None:
        cost = _estimate_cost(
            self.model, self._total_prompt_tokens, self._total_completion_tokens
        )
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            for turn in self._turns:
                f.write(json.dumps(turn) + "\n")
            # Final summary line — read by sandbox_harness._parse_session_cost()
            f.write(
                json.dumps({
                    "actual_cost_usd": round(cost, 8),
                    "total_prompt_tokens": self._total_prompt_tokens,
                    "total_completion_tokens": self._total_completion_tokens,
                })
                + "\n"
            )
