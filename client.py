#!/usr/bin/env python3
"""
OpenRouter SSE client — stdlib only (urllib, json, re).

Handles:
  - Provider locking via provider.order + allow_fallbacks=false
  - SSE streaming accumulation
  - Multi-source reasoning-leak recovery (reasoning field AND content field)
  - Structured tool_call delta merging
"""

import hashlib
import json
import re
import urllib.error
import urllib.request
from typing import Any


# ---------------------------------------------------------------------------
# TurnResult
# ---------------------------------------------------------------------------

class TurnResult:
    __slots__ = (
        "content",
        "tool_calls",   # list of {"id", "name", "args": dict}
        "usage",        # {"prompt_tokens": int, "completion_tokens": int}
        "is_thinking_only",
        "finish_reason",
        "reasoning",
    )

    def __init__(self, content, tool_calls, usage, is_thinking_only, finish_reason, reasoning=""):
        self.content = content
        self.tool_calls = tool_calls
        self.usage = usage
        self.is_thinking_only = is_thinking_only
        self.finish_reason = finish_reason
        self.reasoning = reasoning


# ---------------------------------------------------------------------------
# Tool-call XML / markdown leak parser
# ---------------------------------------------------------------------------

def _synthetic_id(name: str, args_str: str) -> str:
    digest = hashlib.md5(f"{name}:{args_str}".encode()).hexdigest()[:16]
    return f"call_{digest}"


def _parse_leaked_tool_calls(text: str) -> list:
    """
    Extract tool calls from leaked text (reasoning or content).

    Handles three formats Qwen3.5 emits:

    Format A — JSON body directly inside tag:
        <tool_call>{"name": "terminal", "arguments": {"command": "ls"}}</tool_call>

    Format B — separate <name> / <arguments> children:
        <tool_call>
          <name>terminal</name>
          <arguments>{"command": "ls"}</arguments>
        </tool_call>

    Format C — fenced code block labelled with the tool name:
        ```terminal
        {"command": "ls"}
        ```
    """
    calls = []

    # Format A
    for m in re.finditer(
        r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL
    ):
        body = m.group(1).strip()
        try:
            obj = json.loads(body)
            name = obj.get("name") or obj.get("function") or ""
            raw_args = obj.get("arguments") or obj.get("parameters") or obj.get("args") or {}
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    raw_args = {"_raw": raw_args}
            if name:
                calls.append({
                    "id": _synthetic_id(name, json.dumps(raw_args, sort_keys=True)),
                    "name": name,
                    "args": raw_args,
                })
        except json.JSONDecodeError:
            pass

    if calls:
        return calls

    # Format B
    for m in re.finditer(
        r"<tool_call>\s*<name>(.*?)</name>\s*<arguments>(.*?)</arguments>\s*</tool_call>",
        text,
        re.DOTALL,
    ):
        name = m.group(1).strip()
        args_str = m.group(2).strip()
        try:
            raw_args = json.loads(args_str)
        except json.JSONDecodeError:
            raw_args = {"_raw": args_str}
        if name:
            calls.append({
                "id": _synthetic_id(name, args_str),
                "name": name,
                "args": raw_args,
            })

    if calls:
        return calls

    # Format C — fenced block whose language label is a known tool name
    _KNOWN_TOOLS = {"terminal", "read_file", "write_file", "execute_code", "search_files"}
    for m in re.finditer(
        r"```(\w+)\n(.*?)```", text, re.DOTALL
    ):
        label = m.group(1).strip()
        body = m.group(2).strip()
        if label not in _KNOWN_TOOLS:
            continue
        try:
            raw_args = json.loads(body)
        except json.JSONDecodeError:
            raw_args = {"_raw": body}
        calls.append({
            "id": _synthetic_id(label, body),
            "name": label,
            "args": raw_args,
        })

    return calls


def _content_is_only_tool_calls(content: str, calls: list) -> bool:
    """Return True when content consists entirely of tool-call markup."""
    stripped = re.sub(
        r"(<tool_call>.*?</tool_call>|```\w+\n.*?```)",
        "",
        content,
        flags=re.DOTALL,
    ).strip()
    return not stripped


# ---------------------------------------------------------------------------
# SSE stream helpers
# ---------------------------------------------------------------------------

def _merge_tool_call_delta(acc: dict, delta_tc: dict) -> None:
    """Merge a streaming tool_calls delta chunk into accumulator dict."""
    idx = delta_tc.get("index", 0)
    if idx not in acc:
        acc[idx] = {"id": "", "name": "", "arguments": ""}
    entry = acc[idx]
    if delta_tc.get("id"):
        entry["id"] = delta_tc["id"]
    fn = delta_tc.get("function", {})
    if fn.get("name"):
        entry["name"] = fn["name"]
    if fn.get("arguments"):
        entry["arguments"] += fn["arguments"]


def _finalise_tool_calls(acc: dict) -> list:
    calls = []
    for idx in sorted(acc):
        entry = acc[idx]
        name = entry["name"]
        if not name:
            continue
        try:
            raw_args = json.loads(entry["arguments"]) if entry["arguments"] else {}
        except json.JSONDecodeError:
            raw_args = {"_raw": entry["arguments"]}
        call_id = entry["id"] or _synthetic_id(name, entry["arguments"])
        calls.append({"id": call_id, "name": name, "args": raw_args})
    return calls


# ---------------------------------------------------------------------------
# OpenRouterClient
# ---------------------------------------------------------------------------

_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        provider: str = "AlibabaCloud",
        temperature: float = 0.0,
        seed: int | None = None,
        max_tokens: int = 8192,
    ):
        self.api_key = api_key
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens

    def complete(self, messages: list, tools: list) -> TurnResult:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "provider": {
                "order": [self.provider],
                "allow_fallbacks": False,
            },
        }
        if self.seed is not None:
            body["seed"] = self.seed

        data = json.dumps(body).encode()
        req = urllib.request.Request(
            _BASE_URL,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/trajectoryRL/trajrl-bench",
                "X-Title": "custom-harness",
            },
            method="POST",
        )

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tc_delta_acc: dict = {}
        finish_reason: str = "stop"
        usage: dict = {}

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").rstrip("\n\r")
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue

                    choices = chunk.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        if delta.get("content"):
                            content_parts.append(delta["content"])
                        if delta.get("reasoning"):
                            reasoning_parts.append(delta["reasoning"])
                        for tc in delta.get("tool_calls") or []:
                            _merge_tool_call_delta(tc_delta_acc, tc)
                        fr = choices[0].get("finish_reason")
                        if fr:
                            finish_reason = fr

                    # usage comes in a separate chunk (stream_options.include_usage)
                    if chunk.get("usage"):
                        usage = chunk["usage"]

        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"OpenRouter HTTP {exc.code}: {body_text}"
            ) from exc

        accumulated_content = "".join(content_parts)
        accumulated_reasoning = "".join(reasoning_parts)
        tool_calls = _finalise_tool_calls(tc_delta_acc)

        # Multi-source leak recovery
        if not tool_calls:
            # 1. reasoning field first
            leaked = _parse_leaked_tool_calls(accumulated_reasoning)
            # 2. content field if nothing found in reasoning
            if not leaked and accumulated_content:
                leaked = _parse_leaked_tool_calls(accumulated_content)
                if leaked and _content_is_only_tool_calls(accumulated_content, leaked):
                    # content was entirely tool-call markup; clear it
                    accumulated_content = ""
            tool_calls = leaked

        is_thinking_only = (
            bool(accumulated_reasoning)
            and not accumulated_content
            and not tool_calls
        )

        usage_norm = {
            "prompt_tokens": int((usage.get("prompt_tokens") or 0)),
            "completion_tokens": int((usage.get("completion_tokens") or 0)),
        }

        return TurnResult(
            content=accumulated_content,
            tool_calls=tool_calls,
            usage=usage_norm,
            is_thinking_only=is_thinking_only,
            finish_reason=finish_reason,
            reasoning=accumulated_reasoning,
        )
