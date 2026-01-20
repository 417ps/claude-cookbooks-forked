---
title: Tool Patterns Analysis - Claude Cookbooks
date: 2026-01-19
source: claude-cookbooks-forked/tool_use/
category: Technical Reference
---

# Tool Patterns Analysis - Claude Cookbooks

This document extracts and documents reusable tool patterns from the official Claude cookbooks repository. The patterns cover memory management, context optimization, tool discovery, and dynamic tool loading - all essential for building production-grade Claude applications.

## Overview

The cookbooks demonstrate several key architectural patterns for tool-based Claude applications:

- **Memory persistence** across conversations using file-based storage
- **Context compaction** to manage long-running agent sessions
- **Tool discovery** via semantic search and meta-tools
- **Cache-preserving tool loading** with deferred definitions

---

## Pattern 1: Memory Tool Handler with Path Validation

The memory tool enables Claude to persist information across conversations through a file-based system. The handler implementation includes critical security measures to prevent directory traversal attacks.

**Source:** `/Users/personal/Documents/1._Claude_Code/AI-Agent-Tools/claude-cookbooks-forked/tool_use/memory_tool.py` (lines 37-74)

```python
def _validate_path(self, path: str) -> Path:
    """
    Validate and resolve memory paths to prevent directory traversal attacks.
    """
    if not path.startswith("/memories"):
        raise ValueError(
            f"Path must start with /memories, got: {path}. "
            "All memory operations must be confined to the /memories directory."
        )

    # Remove /memories prefix and any leading slashes
    relative_path = path[len("/memories") :].lstrip("/")

    # Resolve to absolute path within memory_root
    if relative_path:
        full_path = (self.memory_root / relative_path).resolve()
    else:
        full_path = self.memory_root.resolve()

    # Verify the resolved path is still within memory_root
    try:
        full_path.relative_to(self.memory_root.resolve())
    except ValueError as e:
        raise ValueError(
            f"Path '{path}' would escape /memories directory. "
            "Directory traversal attempts are not allowed."
        ) from e

    return full_path
```

**When to use:**

- Building any tool that accepts file paths from Claude
- Implementing sandboxed file operations
- Creating secure memory or storage systems for agents

**Key security measures:**

- Require paths to start with a known prefix (`/memories`)
- Resolve paths to absolute form before validation
- Use `relative_to()` to verify containment within allowed directory
- Double-check paths before destructive operations (delete, rename)

---

## Pattern 2: Cross-Conversation Learning

The memory tool enables Claude to learn patterns in one session and apply them in future sessions. This creates agents that improve over time without re-learning.

**Source:** `/Users/personal/Documents/1._Claude_Code/AI-Agent-Tools/claude-cookbooks-forked/tool_use/memory_cookbook.ipynb`

**The workflow:**

1. **Session 1**: Claude encounters a problem, solves it, stores the pattern
2. **Session 2**: Claude checks memory first, finds the stored pattern, applies it immediately

```python
# Enabling memory in API calls
response = client.beta.messages.create(
    model="claude-sonnet-4-5",
    messages=messages,
    tools=[{"type": "memory_20250818", "name": "memory"}],
    betas=["context-management-2025-06-27"],
    max_tokens=2048
)
```

**Memory tool commands:**

| Command | Description | Example Input |
|---------|-------------|---------------|
| `view` | Show directory or file contents | `{"command": "view", "path": "/memories"}` |
| `create` | Create or overwrite a file | `{"command": "create", "path": "/memories/notes.md", "file_text": "..."}` |
| `str_replace` | Replace text in a file | `{"command": "str_replace", "path": "...", "old_str": "...", "new_str": "..."}` |
| `insert` | Insert text at line number | `{"command": "insert", "path": "...", "insert_line": 2, "insert_text": "..."}` |
| `delete` | Delete a file or directory | `{"command": "delete", "path": "/memories/old.txt"}` |
| `rename` | Rename or move a file | `{"command": "rename", "old_path": "...", "new_path": "..."}` |

**When to use:**

- Code review assistants that learn debugging patterns
- Research assistants accumulating domain knowledge
- Customer support bots remembering user preferences
- Any agent that benefits from persistent knowledge

---

## Pattern 3: Context Compaction (clear_thinking, clear_tool_uses)

Long-running agents accumulate context that eventually exceeds limits. Context editing strategies automatically compress conversation history while preserving essential information.

**Source:** `/Users/personal/Documents/1._Claude_Code/AI-Agent-Tools/claude-cookbooks-forked/tool_use/automatic-context-compaction.ipynb`

**Two clearing strategies:**

1. **`clear_thinking_20251015`** - Removes extended thinking blocks from previous turns
2. **`clear_tool_uses_20250919`** - Clears old tool results when token threshold is exceeded

```python
CONTEXT_MANAGEMENT = {
    "edits": [
        # Thinking management MUST come first when combining strategies
        {
            "type": "clear_thinking_20251015",
            "keep": {"type": "thinking_turns", "value": 1}  # Keep only last turn's thinking
        },
        {
            "type": "clear_tool_uses_20250919",
            "trigger": {"type": "input_tokens", "value": 35000},  # Trigger at 35k tokens
            "keep": {"type": "tool_uses", "value": 5},  # Keep last 5 tool uses
            "clear_at_least": {"type": "input_tokens", "value": 3000}  # Clear at least 3k
        }
    ]
}

response = client.beta.messages.create(
    betas=["context-management-2025-06-27"],
    model="claude-sonnet-4-5",
    messages=messages,
    tools=[{"type": "memory_20250818", "name": "memory"}],
    thinking={"type": "enabled", "budget_tokens": 10000},  # Required for clear_thinking
    context_management=CONTEXT_MANAGEMENT,
    max_tokens=2048
)
```

**Configuration options:**

| Parameter | Description | Example Values |
|-----------|-------------|----------------|
| `trigger.type` | What triggers clearing | `"input_tokens"` |
| `trigger.value` | Token count threshold | `35000` (production), `5000` (demo) |
| `keep.type` | What to preserve | `"tool_uses"`, `"thinking_turns"` |
| `keep.value` | How many to keep | `5` tool uses, `1` thinking turn |
| `clear_at_least.value` | Minimum tokens to clear | `3000` |

**When to use:**

- Multi-step workflows processing many items (ticket queues, batch operations)
- Long conversations with extensive tool use
- Sessions using extended thinking that accumulate thinking blocks
- Any agent that might exceed the 200k token context limit

**Key insight:** Short-term context (tool results, thinking) gets cleared, but long-term memory (stored patterns in `/memories`) persists across sessions.

---

## Pattern 4: Tool Choice Modes (auto, tool, any)

The `tool_choice` parameter controls how Claude decides which tools to call. Three modes provide different levels of control.

**Source:** `/Users/personal/Documents/1._Claude_Code/AI-Agent-Tools/claude-cookbooks-forked/tool_use/tool_choice.ipynb`

### Mode: `auto` (Default)

Claude decides whether to call any provided tools or not.

```python
response = client.messages.create(
    model="claude-sonnet-4-5",
    messages=messages,
    tool_choice={"type": "auto"},
    tools=[web_search_tool],
)
```

**Best for:** General assistants where tool use is optional.

### Mode: `tool` (Force Specific Tool)

Force Claude to always use a particular tool.

```python
response = client.messages.create(
    model="claude-sonnet-4-5",
    messages=messages,
    tool_choice={"type": "tool", "name": "print_sentiment_scores"},
    tools=tools,
)
```

**Best for:** Structured output extraction, forcing specific analysis formats.

### Mode: `any` (Force Some Tool)

Claude must call one of the provided tools, but chooses which one.

```python
response = client.messages.create(
    model="claude-sonnet-4-5",
    messages=messages,
    tool_choice={"type": "any"},
    tools=[send_text_tool, get_customer_info_tool],
)
```

**Best for:** Chatbots that must always respond through tools (SMS bots, API-only interfaces).

**Prompt engineering note:** With `auto` mode, write detailed prompts explaining when tools should and should not be used. Claude can be over-eager to call tools without clear guidance.

---

## Pattern 5: Embedding-Based Tool Selection

When applications have dozens or hundreds of tools, providing all definitions upfront consumes context and increases costs. Semantic search enables dynamic tool discovery.

**Source:** `/Users/personal/Documents/1._Claude_Code/AI-Agent-Tools/claude-cookbooks-forked/tool_use/tool_search_with_embeddings.ipynb`

**Architecture:**

1. Convert tool definitions to text representations
2. Generate embeddings using SentenceTransformer
3. Claude calls `tool_search` with natural language query
4. Return matching tools via `tool_reference` content blocks

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Load embedding model (384 dimensions, runs locally)
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def tool_to_text(tool: dict) -> str:
    """Convert tool definition to searchable text."""
    text_parts = [
        f"Tool: {tool['name']}",
        f"Description: {tool['description']}",
    ]
    if "input_schema" in tool and "properties" in tool["input_schema"]:
        params = tool["input_schema"]["properties"]
        param_descriptions = []
        for param_name, param_info in params.items():
            param_desc = param_info.get("description", "")
            param_type = param_info.get("type", "")
            param_descriptions.append(f"{param_name} ({param_type}): {param_desc}")
        if param_descriptions:
            text_parts.append("Parameters: " + ", ".join(param_descriptions))
    return "\n".join(text_parts)

# Create embeddings for all tools
tool_texts = [tool_to_text(tool) for tool in TOOL_LIBRARY]
tool_embeddings = embedding_model.encode(tool_texts, convert_to_numpy=True)

def search_tools(query: str, top_k: int = 5) -> list:
    """Search for tools using semantic similarity."""
    query_embedding = embedding_model.encode(query, convert_to_numpy=True)
    similarities = np.dot(tool_embeddings, query_embedding)
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [{"tool": TOOL_LIBRARY[idx], "score": float(similarities[idx])} for idx in top_indices]
```

**Returning discovered tools:**

```python
# Return tool_reference objects for Claude to use
tool_references = [
    {"type": "tool_reference", "tool_name": result["tool"]["name"]}
    for result in search_results
]

tool_results.append({
    "type": "tool_result",
    "tool_use_id": tool_use_id,
    "content": tool_references,
})
```

**When to use:**

- Applications with >20 specialized tools
- Domain-specific APIs with hundreds of endpoints
- Tool libraries that grow over time
- Cost and latency optimization is a priority

---

## Pattern 6: defer_loading for Prompt Cache Preservation

When dynamically adding tools discovered via search, using `defer_loading=True` preserves prompt caching by keeping tool definitions out of the cached prefix.

**Source:** `/Users/personal/Documents/1._Claude_Code/AI-Agent-Tools/claude-cookbooks-forked/tool_use/tool_search_alternate_approaches.ipynb`

```python
# Start with only the discovery tool
active_tools = [DESCRIBE_TOOL]
loaded_tools = set()

# When Claude discovers a tool via describe_tool
if requested_tool in TOOL_LIBRARY:
    if requested_tool not in loaded_tools:
        # Add with defer_loading=True - CRITICAL for caching
        tool_def = {**TOOL_LIBRARY[requested_tool], "defer_loading": True}
        active_tools.append(tool_def)
        loaded_tools.add(requested_tool)

    # Return tool_reference so Claude can use it immediately
    tool_results.append({
        "type": "tool_result",
        "tool_use_id": block.id,
        "content": [{"type": "tool_reference", "tool_name": requested_tool}],
    })
```

**Why this matters:**

- Without `defer_loading`, adding tools changes the context prefix and invalidates cache
- With `defer_loading=True`, tool definitions load at the point of `tool_reference` in conversation
- System prompt and initial tools stay cached even as new tools are discovered

**When to use:**

- Any application using dynamic tool discovery
- Long-running sessions where cache efficiency matters
- Applications with many potential tools but few used per session

---

## Pattern 7: describe_tool Meta-Tool Pattern

An alternative to embedding-based search: list all tool names in the system prompt and provide a `describe_tool` meta-tool that loads full definitions on demand.

**Source:** `/Users/personal/Documents/1._Claude_Code/AI-Agent-Tools/claude-cookbooks-forked/tool_use/tool_search_alternate_approaches.ipynb`

```python
DESCRIBE_TOOL = {
    "name": "describe_tool",
    "description": "Load a tool's full definition into context. Call this before using any tool for the first time.",
    "input_schema": {
        "type": "object",
        "properties": {
            "tool_name": {
                "type": "string",
                "description": "Name of the tool to load",
            },
        },
        "required": ["tool_name"],
    },
}

# System prompt lists available tools
tool_names = list(TOOL_LIBRARY.keys())
SYSTEM_PROMPT = f"""You are a helpful assistant with access to various tools.

Available tools: {', '.join(tool_names)}

Before using any tool, you must first call describe_tool with the tool name to load it."""
```

**When to use:**

- Tool library is small enough to list names in system prompt
- You want simpler implementation than embedding search
- Tool names are self-descriptive
- You need deterministic tool discovery (no similarity scoring)

**Variants of this pattern:**

- `list_tools` - Returns tool names matching a category or keyword
- **Hierarchical discovery** - Browse categories, then load specific tools
- **Hybrid** - Combine name listing with semantic search for large catalogs

---

## Reusable Utilities

| Utility | Source File | Purpose |
|---------|-------------|---------|
| `MemoryToolHandler` | `memory_tool.py` | Secure file-based memory with path validation |
| `tool_to_text()` | `tool_search_with_embeddings.ipynb` | Convert tool definitions to searchable text |
| `search_tools()` | `tool_search_with_embeddings.ipynb` | Semantic similarity search over tool embeddings |
| `run_conversation_loop()` | `memory_demo/demo_helpers.py` | Handle multi-turn conversations with tool execution |
| `mock_tool_execution()` | `tool_search_with_embeddings.ipynb` | Generate realistic mock responses for demos |

---

## Recommendations

These patterns address different concerns in Claude tool applications. Here is guidance on combining them effectively:

1. **Start with memory for any agent that runs multiple sessions.** The memory tool is low-overhead and provides significant value for learning-based workflows. Implement path validation from day one.

2. **Add context compaction when sessions exceed 30-40k tokens.** Monitor token usage and add `clear_tool_uses` when you see linear growth. Add `clear_thinking` if using extended thinking.

3. **Use tool_choice strategically:**
   - `auto` for general assistants
   - `tool` for structured extraction pipelines
   - `any` for tool-only interfaces

4. **Switch to dynamic tool discovery at >20 tools.** The embedding-based approach scales well and reduces initial context. Use `defer_loading=True` to preserve caching.

5. **Consider the describe_tool pattern for simpler cases.** If tool names are self-descriptive and the catalog is <50 tools, explicit listing may be easier than embedding infrastructure.

6. **Combine memory with context compaction.** Memory files persist across sessions while context clearing manages within-session growth. This separation is intentional and powerful.

7. **Test compaction thresholds empirically.** Start with aggressive thresholds (5-10k) for iterative workflows, higher thresholds (50-100k) for complex multi-phase tasks.
