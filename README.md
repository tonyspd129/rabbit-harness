# Rabbit Harness

Standalone Python ReAct agent for sandbox evaluation.  
Zero external dependencies — stdlib only.

---

## Tools

### `terminal`
Runs a shell command. Returns stdout, stderr, and exit code.  
`command` (required), `timeout` in seconds (default 60).

### `read_file`
Reads a file. Supports pagination for large files via `offset` (1-indexed line) and `limit` (max lines).  
A continuation hint is appended when the file is truncated: `use offset=N to continue`.

### `write_file`
Writes or overwrites a file. Creates parent directories automatically.

### `edit_file`
Makes precise in-place edits without rewriting the whole file.  
Takes `edits: [{old_text, new_text}]`. Each `old_text` must appear exactly once — returns an error if not found or ambiguous.

### `execute_code`
Runs Python code in a subprocess. Returns stdout, stderr, and exit code.

### `grep`
Searches file contents by regex pattern. Uses `rg` (ripgrep) if available, falls back to `grep`.  
Optional: `glob` (file filter), `ignore_case`, `literal`, `context` (surrounding lines).  
Max 100 matches, 500 chars per line.

### `find_files`
Finds files matching a glob pattern. Uses `fd` if available, falls back to `find`.  
`pattern` (required), `path` (default `/workspace`), `limit` (default 1000).

### `ls`
Lists directory contents alphabetically. Directories shown with trailing `/`.  
`path` (default `/workspace`), `limit` (default 500).

---

## Mechanism

```
main.py          CLI entry — reads OPENROUTER_API_KEY from env
  └─ loop.py     ReAct loop (sync, max 40 turns)
       ├─ client.py   OpenRouter SSE client (urllib, no deps)
       │    └─ provider.order locking + reasoning-leak recovery
       ├─ tools.py    Tool schemas + executor
       └─ session.py  Message history + turns.jsonl export
```

**Provider locking** — every request pins a single OpenRouter provider via  
`provider.order: ["AlibabaCloud"], allow_fallbacks: false`.  
Eliminates cross-provider variance between runs.

**Reasoning-leak recovery** — when `delta.tool_calls` is empty but the model  
leaked tool-call XML into `delta.reasoning` or `delta.content`, the client  
extracts and parses them (3 formats: JSON-in-tag, name+arguments children, fenced block).

**Thinking-only recovery** — turns with reasoning but no content and no tool calls  
trigger a prefill retry (up to 5). The counter resets on every successful tool turn.
