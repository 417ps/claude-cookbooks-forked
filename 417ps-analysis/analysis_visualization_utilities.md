---
title: "Analysis: Visualization Utilities from Claude Cookbooks"
date: 2026-01-19
source: claude-cookbooks-forked
category: code-analysis
---

# Visualization Utilities Analysis

This document analyzes two visualization modules from the claude-cookbooks-forked repository that provide different approaches to displaying Claude API responses. The agent_visualizer.py focuses on real-time activity tracking and conversation timelines, while visualize.py provides terminal-based response parsing with Rich library formatting.

These utilities demonstrate practical patterns for building developer tools that adapt to different runtime environments and handle the complexity of multi-turn, tool-using agent conversations.

## Key Patterns

### 1. Jupyter vs Terminal Auto-Detection

The agent_visualizer module implements environment detection to switch between HTML rendering (for Jupyter notebooks) and box-drawing character output (for terminals). This allows a single API call to produce appropriate output regardless of execution context.

**Source:** `/Users/personal/Documents/1._Claude_Code/AI-Agent-Tools/claude-cookbooks-forked/claude_agent_sdk/utils/agent_visualizer.py` (lines 44-60)

```python
def _is_jupyter() -> bool:
    """
    Detect if running in a Jupyter notebook environment.

    Returns True for Jupyter notebook/lab, False for terminal/scripts.
    """
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is None:
            return False
        return bool(shell.__class__.__name__ == "ZMQInteractiveShell")
    except ImportError:
        return False
    except Exception:
        return False
```

**When to use:**
- Building tools that run in both notebook and CLI environments
- Creating visualization libraries that need graceful degradation
- Implementing auto-formatting based on output capabilities

### 2. Subagent Nesting and Depth Tracking

The activity tracking system maintains global state to track nested subagent delegations. This enables proper indentation and context display when agents spawn child agents to handle subtasks.

**Source:** `/Users/personal/Documents/1._Claude_Code/AI-Agent-Tools/claude-cookbooks-forked/claude_agent_sdk/utils/agent_visualizer.py` (lines 106-116, 145-156)

```python
# Track subagent state for activity display
# WARNING: This global state is NOT thread-safe.
_subagent_context: dict[str, Any] = {
    "active": False,
    "name": None,
    "depth": 0,
}

# In print_activity(), when Task tool detected:
if tool_name == "Task":
    if hasattr(first_block, "input") and first_block.input:
        subagent_type = first_block.input.get("subagent_type", "unknown")
        description = first_block.input.get("description", "")
        _subagent_context["active"] = True
        _subagent_context["name"] = subagent_type
        _subagent_context["depth"] += 1
```

**When to use:**
- Displaying hierarchical agent workflows
- Tracking parent-child relationships in multi-agent systems
- Debugging complex delegation chains

### 3. Activity Tracking with Visual Indicators

Real-time activity display uses distinct visual indicators to differentiate between agent thinking, tool usage, subagent delegation, and task completion. The depth-based indentation creates a clear visual hierarchy.

**Source:** `/Users/personal/Documents/1._Claude_Code/AI-Agent-Tools/claude-cookbooks-forked/claude_agent_sdk/utils/agent_visualizer.py` (lines 159-177)

```python
elif tool_name:
    # Check if we're inside a subagent context
    if _subagent_context["active"]:
        indent = "   " * _subagent_context["depth"]
        print(f"{indent}[{_subagent_context['name']}] Using: {tool_name}()")
    else:
        print(f"Using: {tool_name}()")
else:
    if _subagent_context["active"]:
        indent = "   " * _subagent_context["depth"]
        print(f"{indent}[{_subagent_context['name']}] Thinking...")
    else:
        print("Thinking...")
```

**When to use:**
- Providing user feedback during long-running operations
- Creating progress indicators for multi-step processes
- Building debugging tools for agent execution

### 4. Cost and Token Aggregation

The print_final_result function extracts and displays cost information from the SDK response, including per-turn averages and model identification. It relies on the SDK's authoritative cost calculation rather than computing costs independently.

**Source:** `/Users/personal/Documents/1._Claude_Code/AI-Agent-Tools/claude-cookbooks-forked/claude_agent_sdk/utils/agent_visualizer.py` (lines 258-276)

```python
# Print cost (use reported cost from SDK - it's authoritative)
# Note: total_cost_usd is model-aware and calculated by the API
reported_cost = getattr(result_msg, "total_cost_usd", None)
num_turns = getattr(result_msg, "num_turns", 1)

if reported_cost is not None:
    print(f"\nCost: ${reported_cost:.2f}")
    if num_turns and num_turns > 1:
        avg_cost = reported_cost / num_turns
        print(f"   ({num_turns} turns, avg ${avg_cost:.4f}/turn)")

# Show model info
if model:
    print(f"   Model: {model}")

# Print duration if available
if hasattr(result_msg, "duration_ms"):
    print(f"Duration: {result_msg.duration_ms / 1000:.2f}s")
```

**When to use:**
- Budget tracking for API usage
- Performance monitoring and optimization
- Usage reporting and analytics dashboards

### 5. Dual-Mode Rendering (HTML vs Terminal)

The visualize_conversation function implements a strategy pattern, checking the environment and dispatching to either HTML or terminal rendering. This keeps the public API simple while supporting multiple output formats.

**Source:** `/Users/personal/Documents/1._Claude_Code/AI-Agent-Tools/claude-cookbooks-forked/claude_agent_sdk/utils/agent_visualizer.py` (lines 317-348)

```python
def visualize_conversation(messages: list[Any]) -> None:
    """
    Create a clean, professional visualization of the agent conversation.

    Auto-detects environment:
    - Jupyter notebooks: Renders styled HTML timeline with color-coded message blocks
    - Terminal/scripts: Falls back to box-drawing character visualization
    """
    # Auto-detect: use HTML in Jupyter, terminal fallback elsewhere
    if _is_jupyter():
        visualize_conversation_html(messages)
        return

    # Terminal fallback: box-drawing visualization
    # Extract model info for cost calculations
    model = extract_model_from_messages(messages)

    # Header
    print()
    print(BOX_TOP)
    print(f"{BOX_SIDE}  AGENT CONVERSATION TIMELINE" + " " * 25 + BOX_SIDE)
    print(BOX_BOTTOM)
```

**When to use:**
- Building libraries with notebook-first but CLI-compatible interfaces
- Creating documentation tools that work in multiple contexts
- Developing visualization components with graceful fallbacks

### 6. Box-Drawing Character Visualization

Terminal output uses Unicode box-drawing characters to create structured visual blocks. Constants define widths and characters for main conversation boxes versus nested subagent blocks.

**Source:** `/Users/personal/Documents/1._Claude_Code/AI-Agent-Tools/claude-cookbooks-forked/claude_agent_sdk/utils/agent_visualizer.py` (lines 63-74)

```python
# Box-drawing configuration constants
BOX_WIDTH = 58  # Width for main conversation boxes
SUBAGENT_WIDTH = 54  # Width for subagent delegation blocks

# Box-drawing characters for clean visual formatting
BOX_TOP = "+" + "-" * BOX_WIDTH + "+"
BOX_BOTTOM = "+" + "-" * BOX_WIDTH + "+"
BOX_DIVIDER = "|" + "-" * BOX_WIDTH + "|"
BOX_SIDE = "|"
SUBAGENT_TOP = "+" + "-" * SUBAGENT_WIDTH + "+"
SUBAGENT_BOTTOM = "+" + "-" * SUBAGENT_WIDTH + "+"
SUBAGENT_SIDE = "|"
```

**When to use:**
- Creating structured terminal output without external dependencies
- Building CLI tools with professional formatting
- Displaying hierarchical data in text-only environments

### 7. Response Parsing Utilities

The visualize.py module provides flexible response parsing that handles both dictionary and Anthropic SDK object formats. The parse_response function normalizes different input types into a consistent ParsedMessage structure.

**Source:** `/Users/personal/Documents/1._Claude_Code/AI-Agent-Tools/claude-cookbooks-forked/tool_use/utils/visualize.py` (lines 67-113)

```python
def parse_response(response: dict[str, Any] | Any) -> ParsedMessage:
    """Parse a Claude API response into a structured format."""
    # Handle dict format (from JSON)
    if isinstance(response, dict):
        role = response.get("role", "unknown")
        content_blocks = response.get("content", [])
        model = response.get("model")
        stop_reason = response.get("stop_reason")
        usage = response.get("usage", {})

        parsed_content = [parse_content_block(block) for block in content_blocks]

        return ParsedMessage(
            role=role,
            content=parsed_content,
            model=model,
            stop_reason=stop_reason,
            usage=usage,
        )

    # Handle Anthropic SDK Message object
    if hasattr(response, "content"):
        role = getattr(response, "role", "unknown")
        content_blocks = response.content
        # ... extraction continues
```

**When to use:**
- Processing API responses from different sources (direct API, SDK, cached JSON)
- Building middleware that transforms response formats
- Creating adapters between different Claude client implementations

### 8. ParsedMessage and ParsedContent Classes

These data classes provide a normalized structure for working with Claude responses. ParsedContent wraps individual content blocks with type information, while ParsedMessage aggregates the full response with metadata.

**Source:** `/Users/personal/Documents/1._Claude_Code/AI-Agent-Tools/claude-cookbooks-forked/tool_use/utils/visualize.py` (lines 15-38)

```python
class ParsedContent:
    """Represents a parsed content block from a Claude message."""

    def __init__(self, content_type: str, data: dict[str, Any]):
        self.type = content_type
        self.data = data


class ParsedMessage:
    """Represents a parsed Claude message with metadata."""

    def __init__(
        self,
        role: str,
        content: list[ParsedContent],
        model: str | None = None,
        stop_reason: str | None = None,
        usage: dict[str, int] | None = None,
    ):
        self.role = role
        self.content = content
        self.model = model
        self.stop_reason = stop_reason
        self.usage = usage or {}
```

**When to use:**
- Building type-safe processing pipelines for Claude responses
- Creating reusable visualization or analysis components
- Implementing response transformers and filters

## Reusable Utilities

| Utility | File | Purpose |
|---------|------|---------|
| `_is_jupyter()` | agent_visualizer.py:44 | Detect Jupyter vs terminal environment |
| `reset_activity_context()` | agent_visualizer.py:207 | Clear subagent tracking state before new query |
| `print_activity()` | agent_visualizer.py:119 | Real-time activity display with depth tracking |
| `print_final_result()` | agent_visualizer.py:229 | Display final result with cost breakdown |
| `visualize_conversation()` | agent_visualizer.py:317 | Auto-detecting conversation timeline display |
| `extract_model_from_messages()` | agent_visualizer.py:77 | Extract model ID from message list |
| `_format_tool_info()` | agent_visualizer.py:278 | Format tool name with key parameters |
| `parse_response()` | visualize.py:67 | Parse API response to normalized structure |
| `parse_content_block()` | visualize.py:41 | Parse individual content blocks |
| `visualize_message()` | visualize.py:284 | Render message with Rich library formatting |
| `show_response()` | visualize.py:365 | Simple helper for single response display |
| `visualize` (class) | visualize.py:324 | Context manager for auto-visualization |

## Recommendations

### For Immediate Reuse

The environment detection pattern (`_is_jupyter()`) is immediately portable to any project that needs notebook-aware behavior. Copy the function directly - it has no external dependencies beyond IPython (which is present in notebook environments).

### For Agent Debugging

The activity tracking system provides a good foundation for agent debugging tools. However, the global state approach has thread-safety limitations noted in the code comments. For production use, consider refactoring to use `contextvars` for proper async context management.

### For Response Processing

The ParsedMessage/ParsedContent pattern from visualize.py creates a clean abstraction layer. This approach is valuable when building tools that need to work with responses from multiple sources (direct API calls, SDK objects, cached JSON responses).

### Architecture Considerations

Both modules handle the same core challenge - making Claude API responses human-readable - but take different approaches:

- agent_visualizer.py: Optimized for real-time streaming with state tracking
- visualize.py: Optimized for post-hoc analysis with Rich library formatting

When building new visualization tools, consider which use case is primary. Real-time display benefits from the streaming-aware patterns in agent_visualizer, while batch analysis benefits from the parsing abstractions in visualize.py.

### Missing Functionality

Neither module currently handles:

- Persistent logging to files or databases
- Filtering or searching within conversation history
- Export to common formats (JSON, CSV, HTML files)
- Integration with observability platforms

These would be natural extensions for production deployment.
