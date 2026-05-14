#!/usr/bin/env python3
"""
Tool schemas and executor for the custom ReAct harness.

Tool set mirrors the PI coding-agent (packages/coding-agent/src/core/tools/)
while keeping hermes-compatible names for the core file/shell tools.

Tools:
  terminal      - run shell commands (bash)
  read_file     - read file with offset/limit pagination
  write_file    - write/overwrite a file
  edit_file     - partial in-place edits via [{old_text, new_text}] pairs
  execute_code  - run Python code
  grep          - search file contents (ripgrep / grep with glob + context)
  find_files    - find files by glob pattern (fd / find)
  ls            - list directory entries
"""

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


# ── Output limits (match PI defaults) ────────────────────────────────────────
_MAX_LINES = 2000
_MAX_BYTES = 50 * 1024   # 50 KB
_MAX_GREP_LINE = 500     # chars per grep match line
_MAX_GREP_MATCHES = 100
_MAX_FIND_RESULTS = 1000
_MAX_LS_ENTRIES = 500


def _truncate_output(text: str, max_lines: int = _MAX_LINES, max_bytes: int = _MAX_BYTES) -> str:
    """Truncate output to max_lines or max_bytes, whichever comes first."""
    lines = text.splitlines(keepends=True)
    kept = []
    total_bytes = 0
    for i, line in enumerate(lines):
        if i >= max_lines:
            remaining = len(lines) - i
            kept.append(f"\n... ({remaining} more lines truncated)")
            break
        total_bytes += len(line.encode())
        if total_bytes > max_bytes:
            remaining = len(lines) - i
            kept.append(f"\n... ({remaining} more lines truncated, {total_bytes // 1024}KB limit reached)")
            break
        kept.append(line)
    return "".join(kept)


# ── TOOLS_SCHEMA ──────────────────────────────────────────────────────────────

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "terminal",
            "description": (
                "Run a shell command. Returns stdout, stderr, and the return code. "
                "Prefer this for any operation not covered by a dedicated tool."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default 60).",
                        "default": 60,
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read file contents. For large files use `offset` and `limit` "
                "to read in chunks (offset is 1-indexed line number). "
                "A continuation hint is appended when the file is truncated."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute file path.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Start reading from this line number (1-indexed). Default: 1.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to return. Default: 2000.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write content to a file, creating or overwriting it. "
                "Parent directories are created automatically. "
                "For small edits to an existing file, prefer `edit_file`."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute file path.",
                    },
                    "content": {
                        "type": "string",
                        "description": "File content to write.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Make precise in-place edits to a file without rewriting it entirely. "
                "Provide a list of {old_text, new_text} pairs. "
                "Each old_text must appear exactly once in the file. "
                "All edits are applied against the original content simultaneously."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute file path.",
                    },
                    "edits": {
                        "type": "array",
                        "description": "List of text replacements to apply.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "old_text": {
                                    "type": "string",
                                    "description": "Exact text to find (must be unique in the file).",
                                },
                                "new_text": {
                                    "type": "string",
                                    "description": "Replacement text.",
                                },
                            },
                            "required": ["old_text", "new_text"],
                        },
                        "minItems": 1,
                    },
                },
                "required": ["path", "edits"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "Execute Python code. Standard library is available.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python source code to run.",
                    },
                    "language": {
                        "type": "string",
                        "description": "Language (only 'python' supported).",
                        "default": "python",
                    },
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": (
                "Search file contents for a pattern. "
                "Returns matching lines with file names and line numbers. "
                "Respects .gitignore. Faster than reading each file manually."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory or file to search (default: /workspace).",
                        "default": "/workspace",
                    },
                    "glob": {
                        "type": "string",
                        "description": "File glob filter, e.g. '*.py', '*.sh' (optional).",
                    },
                    "ignore_case": {
                        "type": "boolean",
                        "description": "Case-insensitive search (default false).",
                        "default": False,
                    },
                    "literal": {
                        "type": "boolean",
                        "description": "Treat pattern as a literal string, not a regex (default false).",
                        "default": False,
                    },
                    "context": {
                        "type": "integer",
                        "description": "Number of lines of context before/after each match (default 0).",
                        "default": 0,
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_files",
            "description": (
                "Find files matching a glob pattern. "
                "Respects .gitignore. Use for directory exploration."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern, e.g. '*.py', '**/*.sh', 'Makefile'.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search (default: /workspace).",
                        "default": "/workspace",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default 1000).",
                        "default": 1000,
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ls",
            "description": (
                "List directory contents, sorted alphabetically. "
                "Directories are indicated with a trailing '/'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory to list (default: /workspace).",
                        "default": "/workspace",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max entries to return (default 500).",
                        "default": 500,
                    },
                },
                "required": [],
            },
        },
    },
]


# ── Implementations ───────────────────────────────────────────────────────────

def _tool_terminal(command: str, timeout: int = 60) -> str:
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        parts = []
        if result.stdout:
            parts.append(_truncate_output(result.stdout))
        if result.stderr:
            parts.append(f"[stderr]\n{_truncate_output(result.stderr)}")
        parts.append(f"[rc={result.returncode}]")
        return "\n".join(parts)
    except subprocess.TimeoutExpired:
        return f"[error] command timed out after {timeout}s"
    except Exception as exc:
        return f"[error] {exc}"


def _tool_read_file(path: str, offset: int | None = None, limit: int | None = None) -> str:
    try:
        text = Path(path).read_text(errors="replace")
    except Exception as exc:
        return f"[error] {exc}"

    lines = text.splitlines(keepends=True)
    total_lines = len(lines)

    start = max(0, (offset - 1) if offset is not None else 0)
    end = start + (limit if limit is not None else _MAX_LINES)
    chunk = lines[start:end]

    result = "".join(chunk)
    # Byte truncation within chunk
    if len(result.encode()) > _MAX_BYTES:
        result = result.encode()[:_MAX_BYTES].decode(errors="replace")
        result += f"\n... (truncated at {_MAX_BYTES // 1024}KB)"
    elif end < total_lines:
        result += (
            f"\n... ({total_lines - end} more lines, "
            f"use offset={end + 1} to continue)"
        )

    return result


def _tool_write_file(path: str, content: str) -> str:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"[ok] wrote {len(content)} bytes to {path}"
    except Exception as exc:
        return f"[error] {exc}"


def _tool_edit_file(path: str, edits: list) -> str:
    try:
        content = Path(path).read_text(errors="replace")
    except Exception as exc:
        return f"[error] could not read {path}: {exc}"

    original = content
    applied = 0
    for i, edit in enumerate(edits):
        old = edit.get("old_text", "")
        new = edit.get("new_text", "")
        if not old:
            return f"[error] edit #{i + 1}: old_text is empty"
        count = content.count(old)
        if count == 0:
            return (
                f"[error] edit #{i + 1}: old_text not found in file. "
                f"Snippet: {repr(old[:80])}"
            )
        if count > 1:
            return (
                f"[error] edit #{i + 1}: old_text matches {count} locations — "
                f"provide more context to make it unique. "
                f"Snippet: {repr(old[:80])}"
            )
        content = content.replace(old, new, 1)
        applied += 1

    try:
        Path(path).write_text(content)
    except Exception as exc:
        return f"[error] could not write {path}: {exc}"

    return f"[ok] applied {applied} edit(s) to {path}"


def _tool_execute_code(code: str, language: str = "python") -> str:
    if language not in ("python", "python3"):
        return f"[error] unsupported language: {language}"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as tmp:
            tmp.write(code)
            tmp_path = tmp.name
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=120,
        )
        parts = []
        if result.stdout:
            parts.append(_truncate_output(result.stdout))
        if result.stderr:
            parts.append(f"[stderr]\n{_truncate_output(result.stderr)}")
        parts.append(f"[rc={result.returncode}]")
        return "\n".join(parts)
    except subprocess.TimeoutExpired:
        return "[error] code execution timed out after 120s"
    except Exception as exc:
        return f"[error] {exc}"
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def _tool_grep(
    pattern: str,
    path: str = "/workspace",
    glob: str | None = None,
    ignore_case: bool = False,
    literal: bool = False,
    context: int = 0,
) -> str:
    # Prefer ripgrep (rg) for speed; fall back to grep.
    _rg = _which("rg")
    if _rg:
        cmd = [_rg, "--line-number", "--no-heading", "--color=never"]
        if ignore_case:
            cmd.append("--ignore-case")
        if literal:
            cmd.append("--fixed-strings")
        if glob:
            cmd += ["--glob", glob]
        if context:
            cmd += ["--context", str(context)]
        cmd += [f"--max-count={_MAX_GREP_MATCHES}", pattern, path]
    else:
        cmd = ["grep", "-rEn", "--color=never"]
        if ignore_case:
            cmd.append("-i")
        if literal:
            cmd.append("-F")
        if glob:
            cmd += ["--include", glob]
        if context:
            cmd += [f"-C{context}"]
        cmd += [pattern, path]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        )
        output = result.stdout
        if not output:
            if result.returncode not in (0, 1):
                return f"[error] grep exited {result.returncode}: {result.stderr[:200]}"
            return "[no matches]"
        # Truncate individual lines and total output
        lines = output.splitlines()
        truncated = []
        for line in lines[:_MAX_GREP_MATCHES]:
            if len(line) > _MAX_GREP_LINE:
                line = line[:_MAX_GREP_LINE] + "..."
            truncated.append(line)
        if len(lines) > _MAX_GREP_MATCHES:
            truncated.append(f"... ({len(lines) - _MAX_GREP_MATCHES} more matches truncated)")
        return "\n".join(truncated)
    except subprocess.TimeoutExpired:
        return "[error] search timed out after 30s"
    except Exception as exc:
        return f"[error] {exc}"


def _tool_find_files(
    pattern: str,
    path: str = "/workspace",
    limit: int = _MAX_FIND_RESULTS,
) -> str:
    # Prefer fd for speed; fall back to find.
    _fd = _which("fd") or _which("fdfind")
    if _fd:
        cmd = [_fd, "--hidden", "--no-ignore", "--glob", pattern, path]
    else:
        # Construct a find command with -name glob
        name_pat = pattern.split("/")[-1] if "/" not in pattern else pattern
        cmd = ["find", path, "-name", name_pat]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        )
        lines = [l for l in result.stdout.splitlines() if l]
        # Sort, add trailing / for dirs
        entries = []
        for line in sorted(lines):
            p = Path(line)
            try:
                if p.is_dir():
                    entries.append(line.rstrip("/") + "/")
                else:
                    entries.append(line)
            except Exception:
                entries.append(line)

        total = len(entries)
        entries = entries[:limit]
        output = "\n".join(entries)
        if total > limit:
            output += f"\n... ({total - limit} more results truncated)"
        return output or "[no results]"
    except subprocess.TimeoutExpired:
        return "[error] find timed out after 30s"
    except Exception as exc:
        return f"[error] {exc}"


def _tool_ls(path: str = "/workspace", limit: int = _MAX_LS_ENTRIES) -> str:
    try:
        p = Path(path)
        if not p.exists():
            return f"[error] path does not exist: {path}"
        if not p.is_dir():
            return f"[error] not a directory: {path}"
        entries = []
        for entry in p.iterdir():
            try:
                name = entry.name + ("/" if entry.is_dir() else "")
                entries.append(name)
            except Exception:
                pass
        entries.sort(key=lambda s: s.lower())
        total = len(entries)
        entries = entries[:limit]
        output = "\n".join(entries)
        if total > limit:
            output += f"\n... ({total - limit} more entries truncated)"
        return output or "[empty directory]"
    except Exception as exc:
        return f"[error] {exc}"


def _which(name: str) -> str | None:
    try:
        result = subprocess.run(
            ["which", name], capture_output=True, text=True, timeout=5
        )
        path = result.stdout.strip()
        return path if path else None
    except Exception:
        return None


# ── Dispatcher ────────────────────────────────────────────────────────────────

def execute_tool(name: str, args: dict) -> str:
    if name == "terminal":
        return _tool_terminal(
            command=args.get("command", ""),
            timeout=int(args.get("timeout", 60)),
        )
    if name == "read_file":
        return _tool_read_file(
            path=args.get("path", ""),
            offset=args.get("offset"),
            limit=args.get("limit"),
        )
    if name == "write_file":
        return _tool_write_file(
            path=args.get("path", ""),
            content=args.get("content", ""),
        )
    if name == "edit_file":
        return _tool_edit_file(
            path=args.get("path", ""),
            edits=args.get("edits", []),
        )
    if name == "execute_code":
        return _tool_execute_code(
            code=args.get("code", ""),
            language=args.get("language", "python"),
        )
    if name in ("grep", "search_files"):   # accept legacy name too
        return _tool_grep(
            pattern=args.get("pattern", ""),
            path=args.get("path", "/workspace"),
            glob=args.get("glob"),
            ignore_case=bool(args.get("ignore_case", False)),
            literal=bool(args.get("literal", False)),
            context=int(args.get("context", 0)),
        )
    if name == "find_files":
        return _tool_find_files(
            pattern=args.get("pattern", ""),
            path=args.get("path", "/workspace"),
            limit=int(args.get("limit", _MAX_FIND_RESULTS)),
        )
    if name == "ls":
        return _tool_ls(
            path=args.get("path", "/workspace"),
            limit=int(args.get("limit", _MAX_LS_ENTRIES)),
        )
    return f"[error] unknown tool: {name}"
