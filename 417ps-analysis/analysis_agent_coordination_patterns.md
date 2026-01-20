---
title: Agent Coordination Patterns Analysis
date: 2026-01-19
source: claude-cookbooks-forked repository
category: Architecture Analysis
---

# Agent Coordination Patterns Analysis

This document extracts and analyzes agent coordination patterns from Anthropic's claude-cookbooks-forked repository. The patterns demonstrate multi-agent orchestration, tool restriction strategies, and configuration approaches used in production-grade Claude agent implementations.

The source code reveals several distinct architectural approaches, from simple orchestrator-worker patterns to sophisticated multi-agent delegation systems with hooks, commands, and output styles.

---

## 1. Orchestrator-Workers Pattern (Research Lead + Subagents)

The research system implements a hierarchical orchestration pattern where a lead agent coordinates multiple specialized subagents. The lead handles planning and synthesis while workers execute focused research tasks.

**Source:** `/Users/personal/Documents/1._Claude_Code/AI-Agent-Tools/claude-cookbooks-forked/patterns/agents/prompts/research_lead_agent.md` (Lines 4-69)

The research process breaks down into five phases: assessment, query type determination, plan development, execution, and answer formatting.

```markdown
<research_process>
Follow this process to break down the user's question and develop an excellent research plan.
1. **Assessment and breakdown**: Analyze and break down the user's prompt
2. **Query type determination**: Explicitly state your reasoning on what type of query this is
3. **Detailed research plan development**: Based on the query type, develop a specific research plan
4. **Methodical plan execution**: Execute the plan fully, using parallel subagents where possible
</research_process>
```

**Query Type Classification** (Lines 12-29):

- **Depth-first query**: Multiple perspectives on a single issue, requires "going deep"
- **Breadth-first query**: Distinct, independent sub-questions, requires "going wide"
- **Straightforward query**: Focused, well-defined, single investigation

**When to use:**

- Research tasks requiring multiple information sources
- Complex questions benefiting from parallel investigation
- Tasks where quality synthesis matters more than speed
- Scenarios needing independent verification of claims

---

## 2. Subagent Count Guidelines

The lead agent prompt includes explicit guidance on scaling subagent deployment based on query complexity.

**Source:** `/Users/personal/Documents/1._Claude_Code/AI-Agent-Tools/claude-cookbooks-forked/patterns/agents/prompts/research_lead_agent.md` (Lines 71-87)

```markdown
<subagent_count_guidelines>
1. **Simple/Straightforward queries**: create 1 subagent
   - Example: "What is the tax deadline this year?" → 1 subagent
2. **Standard complexity queries**: 2-3 subagents
   - Example: "Compare the top 3 cloud providers" → 3 subagents
3. **Medium complexity queries**: 3-5 subagents
   - Example: "Analyze the impact of AI on healthcare" → 4 subagents
4. **High complexity queries**: 5-10 subagents (maximum 20)
   - Example: "Fortune 500 CEOs birthplaces and ages" → 10 subagents
   **IMPORTANT**: Never create more than 20 subagents unless strictly necessary.
</subagent_count_guidelines>
```

**Key constraint:** The 20-subagent maximum prevents runaway resource consumption and forces efficient task consolidation.

---

## 3. Research Subagent Execution Pattern

Subagents follow a structured OODA loop (observe, orient, decide, act) with explicit tool budgeting.

**Source:** `/Users/personal/Documents/1._Claude_Code/AI-Agent-Tools/claude-cookbooks-forked/patterns/agents/prompts/research_subagent.md` (Lines 3-14)

```markdown
<research_process>
1. **Planning**: First, think through the task thoroughly. Make a research plan...
   - Determine a 'research budget' - roughly how many tool calls to conduct
   - Simpler tasks: under 5 tool calls
   - Medium tasks: 5 tool calls
   - Hard tasks: about 10 tool calls
   - Very difficult: up to 15 tool calls

2. **Tool selection**: Reason about what tools would be most helpful

3. **Research loop**: Execute an excellent OODA loop by:
   (a) observing what information has been gathered
   (b) orienting toward what tools and queries would be best
   (c) making an informed, well-reasoned decision
   (d) acting to use this tool
</research_process>
```

**Tool Call Limits** (Lines 44-46):

```markdown
<maximum_tool_call_limit>
To prevent overloading the system, stay under a limit of 20 tool calls and under about 100 sources.
If you exceed this limit, the subagent will be terminated.
</maximum_tool_call_limit>
```

---

## 4. Tool Restriction Pattern (allowed_tools)

The Claude Agent SDK uses explicit tool whitelisting to constrain agent behavior. This pattern appears across all agent implementations with different tool sets based on agent purpose.

**Source:** `/Users/personal/Documents/1._Claude_Code/AI-Agent-Tools/claude-cookbooks-forked/claude_agent_sdk/research_agent/agent.py` (Lines 101-107)

```python
options = ClaudeAgentOptions(
    model=model,
    allowed_tools=["WebSearch", "Read"],
    continue_conversation=continue_conversation,
    system_prompt=RESEARCH_SYSTEM_PROMPT,
    max_buffer_size=10 * 1024 * 1024,  # 10MB buffer
)
```

**Chief of Staff Agent** - broader tool access for orchestration:

**Source:** `/Users/personal/Documents/1._Claude_Code/AI-Agent-Tools/claude-cookbooks-forked/claude_agent_sdk/chief_of_staff_agent/agent.py` (Lines 85-99)

```python
options = ClaudeAgentOptions(
    model="claude-opus-4-5",
    allowed_tools=[
        "Task",  # enables subagent delegation
        "Read",
        "Write",
        "Edit",
        "Bash",
        "WebSearch",
    ],
    continue_conversation=continue_conversation,
    system_prompt=system_prompt,
    permission_mode=permission_mode,
    cwd=os.path.dirname(os.path.abspath(__file__)),
    settings=settings,
    setting_sources=["project", "local"],
)
```

**When to use allowed_tools:**

- Constraining research agents to read-only operations
- Preventing code execution in analysis-focused agents
- Limiting network access for security-sensitive tasks
- Enforcing separation of concerns between agent types

---

## 5. Disallowed Tools Pattern (MCP Enforcement)

The observability agent demonstrates using `disallowed_tools` to force MCP usage over CLI fallbacks.

**Source:** `/Users/personal/Documents/1._Claude_Code/AI-Agent-Tools/claude-cookbooks-forked/claude_agent_sdk/observability_agent/agent.py` (Lines 109-121)

```python
# Configure disallowed tools to ensure MCP usage
# Without this, the agent could bypass MCP by using Bash with gh CLI
disallowed_tools = ["Bash", "Task", "WebSearch", "WebFetch"] if restrict_to_mcp else []

options = ClaudeAgentOptions(
    model=model,
    allowed_tools=allowed_tools,
    disallowed_tools=disallowed_tools,
    continue_conversation=continue_conversation,
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    mcp_servers=servers,
    permission_mode="acceptEdits",
)
```

**When to use disallowed_tools:**

- Forcing specific integration paths (MCP over CLI)
- Preventing fallback behaviors that bypass monitoring
- Ensuring audit trails through controlled interfaces
- Testing MCP server implementations in isolation

---

## 6. MCP Integration Pattern

The observability agent shows how to configure MCP (Model Context Protocol) servers for external service integration.

**Source:** `/Users/personal/Documents/1._Claude_Code/AI-Agent-Tools/claude-cookbooks-forked/claude_agent_sdk/observability_agent/agent.py` (Lines 40-64)

```python
def get_github_mcp_server() -> dict[str, McpServerConfig]:
    """Get the GitHub MCP server configuration."""
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        return {}

    return {
        "github": {
            "command": "docker",
            "args": [
                "run",
                "-i",
                "--rm",
                "-e",
                "GITHUB_PERSONAL_ACCESS_TOKEN",
                "ghcr.io/github/github-mcp-server",
            ],
            "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": token},
        }
    }
```

**Dynamic tool registration from MCP servers** (Lines 106-107):

```python
# Build allowed tools list based on configured MCP servers
allowed_tools = [f"mcp__{name}" for name in servers]
```

**When to use MCP:**

- Integrating with external APIs through standardized protocol
- GitHub, Slack, database, or other service access
- When audit logging of external calls is required
- Containerized tool execution for security isolation

---

## 7. Command Delegation Pattern

Slash commands provide pre-defined workflows that coordinate multiple subagents.

**Source:** `/Users/personal/Documents/1._Claude_Code/AI-Agent-Tools/claude-cookbooks-forked/claude_agent_sdk/chief_of_staff_agent/.claude/commands/budget-impact.md`

```markdown
---
name: budget-impact
description: Analyze the financial impact of a decision on budget, burn rate, and runway
---

Use the financial-analyst subagent to analyze the budget impact of: $ARGUMENTS

Provide a comprehensive analysis including:
1. Total cost (one-time and recurring)
2. Impact on monthly burn rate
3. Change in runway (months)
4. ROI analysis if applicable
5. Alternative options to consider
6. Risk factors
```

**Multi-agent command example:**

**Source:** `/Users/personal/Documents/1._Claude_Code/AI-Agent-Tools/claude-cookbooks-forked/claude_agent_sdk/chief_of_staff_agent/.claude/commands/strategic-brief.md`

```markdown
---
name: strategic-brief
description: Generate a comprehensive strategic brief by coordinating analysis from both financial and talent perspectives
---

Create a strategic brief on: $ARGUMENTS

Coordinate with both the financial-analyst and recruiter subagents to provide:

## Executive Summary
- Key recommendation (1-2 sentences)
- Critical metrics impact

## Financial Analysis (via financial-analyst)
## Talent Perspective (via recruiter)
## Strategic Recommendation
## Alternative Options
```

**When to use commands:**

- Standardizing common multi-step workflows
- Ensuring consistent output structure across similar tasks
- Providing shortcuts for frequently used agent coordination patterns
- Encoding organizational knowledge about which agents handle what

---

## 8. Subagent Definition Pattern

Subagents are defined in markdown files with YAML frontmatter specifying their capabilities and tool access.

**Source:** `/Users/personal/Documents/1._Claude_Code/AI-Agent-Tools/claude-cookbooks-forked/claude_agent_sdk/chief_of_staff_agent/.claude/agents/financial-analyst.md` (Lines 1-5)

```markdown
---
name: financial-analyst
description: Financial analysis expert specializing in startup metrics, burn rate, runway calculations, and investment decisions. Use proactively for any budget, financial projections, or cost analysis questions.
tools: Read, Bash, WebSearch
---
```

**Recruiter subagent example:**

**Source:** `/Users/personal/Documents/1._Claude_Code/AI-Agent-Tools/claude-cookbooks-forked/claude_agent_sdk/chief_of_staff_agent/.claude/agents/recruiter.md` (Lines 1-5)

```markdown
---
name: recruiter
description: Technical recruiting specialist focused on startup hiring, talent pipeline management, and candidate evaluation. Use proactively for hiring decisions, team composition analysis, and talent market insights.
tools: Read, WebSearch, Bash
---
```

**Key elements in subagent definitions:**

- `name`: Identifier used for delegation
- `description`: When to invoke this subagent (loaded into orchestrator context)
- `tools`: Whitelist of allowed tools for this subagent
- Body: Full system prompt with responsibilities, available data, and output guidelines

---

## 9. Output Styles System

Output styles define formatting templates that can be applied to any agent response.

**Source:** `/Users/personal/Documents/1._Claude_Code/AI-Agent-Tools/claude-cookbooks-forked/claude_agent_sdk/chief_of_staff_agent/.claude/output-styles/executive.md`

```markdown
---
name: executive
description: Concise, KPI-focused communication for C-suite executives
---

## Communication Principles

- **Lead with the decision or recommendation** - First sentence should be the key takeaway
- **Use bullet points** - Maximum 3-5 points per section
- **Numbers over narrative** - Specific metrics, percentages, and timeframes
- **Action-oriented** - Clear next steps and decisions required

## Format Template

**RECOMMENDATION:** [One sentence decision/action]

**KEY METRICS:**
- Metric 1: **X%** change
- Metric 2: **$Y** impact
- Metric 3: **Z months** timeline

**RATIONALE:** [2-3 sentences max]

**NEXT STEPS:**
1. Immediate action
2. Near-term action
3. Follow-up required
```

**Applying output styles programmatically:**

**Source:** `/Users/personal/Documents/1._Claude_Code/AI-Agent-Tools/claude-cookbooks-forked/claude_agent_sdk/chief_of_staff_agent/agent.py` (Lines 80-83)

```python
# build options with optional output style
settings = None
if output_style:
    settings = json.dumps({"outputStyle": output_style})
```

**When to use output styles:**

- Adapting agent responses to different audiences (executive, technical, board)
- Ensuring consistent formatting across agent responses
- Reducing prompt length by externalizing format instructions
- Allowing users to switch styles without modifying agent logic

---

## 10. Compliance Hooks Pattern

Hooks enable event-driven side effects, like audit logging, that run after tool execution.

**Source:** `/Users/personal/Documents/1._Claude_Code/AI-Agent-Tools/claude-cookbooks-forked/claude_agent_sdk/chief_of_staff_agent/.claude/hooks/report-tracker.py` (Lines 13-74)

```python
def track_report(tool_name, tool_input, tool_response):
    """Log ALL file creation/modification for audit trail"""

    file_path = tool_input.get("file_path", "")
    if not file_path:
        return

    # Prepare history file path
    history_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../audit/report_history.json"
    )

    # Load existing history or create new
    if os.path.exists(history_file):
        with open(history_file) as f:
            history = json.load(f)
    else:
        history = {"reports": []}

    # Determine action type
    action = "created" if tool_name == "Write" else "modified"

    # Create history entry
    entry = {
        "timestamp": datetime.now().isoformat(),
        "file": os.path.basename(file_path),
        "path": file_path,
        "action": action,
        "word_count": len(content.split()) if content else 0,
        "tool": tool_name,
    }

    # Add to history (keep only last 50 entries)
    history["reports"].append(entry)
    history["reports"] = history["reports"][-50:]
```

**When to use hooks:**

- Audit logging for compliance requirements
- Triggering external notifications (Slack, email)
- Collecting metrics on agent behavior
- Enforcing policies (e.g., blocking certain file writes)

---

## 11. ClaudeAgentOptions Configuration Reference

The SDK exposes many configuration options. Here is a consolidated reference based on usage across the codebase.

| Option | Type | Purpose | Example Value |
|--------|------|---------|---------------|
| `model` | str | Claude model to use | `"claude-opus-4-5"` |
| `allowed_tools` | list | Whitelist of permitted tools | `["Read", "WebSearch"]` |
| `disallowed_tools` | list | Blacklist of forbidden tools | `["Bash", "Task"]` |
| `continue_conversation` | bool | Maintain conversation state | `True` |
| `system_prompt` | str | Agent instructions | `"You are a research agent..."` |
| `permission_mode` | str | Execution mode | `"default"`, `"plan"`, `"acceptEdits"` |
| `cwd` | str | Working directory | `os.path.dirname(__file__)` |
| `settings` | str | JSON settings (e.g., output style) | `'{"outputStyle": "executive"}'` |
| `setting_sources` | list | Where to load settings from | `["project", "local"]` |
| `mcp_servers` | dict | MCP server configurations | See MCP section above |
| `max_buffer_size` | int | Response buffer size | `10 * 1024 * 1024` (10MB) |

---

## 12. Utility Functions

The repository includes reusable utilities for common agent operations.

**Source:** `/Users/personal/Documents/1._Claude_Code/AI-Agent-Tools/claude-cookbooks-forked/patterns/agents/util.py`

| Function | Lines | Purpose |
|----------|-------|---------|
| `llm_call(prompt, system_prompt, model)` | 9-30 | Simple synchronous Claude API call with configurable model |
| `extract_xml(text, tag)` | 33-45 | Parse XML-tagged content from LLM responses |

```python
def llm_call(prompt: str, system_prompt: str = "", model="claude-sonnet-4-5") -> str:
    """Calls the model with the given prompt and returns the response."""
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    messages = [{"role": "user", "content": prompt}]
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system_prompt,
        messages=messages,
        temperature=0.1,
    )
    return response.content[0].text


def extract_xml(text: str, tag: str) -> str:
    """Extracts the content of the specified XML tag from the given text."""
    match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1) if match else ""
```

---

## Recommendations

Based on analysis of these patterns, the following recommendations apply to building multi-agent systems.

**Architecture:**

- Use hierarchical orchestration (lead + workers) for complex research tasks
- Define explicit subagent count limits to prevent runaway resource usage
- Separate concerns between orchestrator (planning/synthesis) and workers (execution)

**Tool Management:**

- Default to minimal tool sets; add tools only when needed
- Use `disallowed_tools` when enforcing specific integration paths
- Consider MCP for external service integrations requiring audit trails

**Configuration:**

- Externalize agent definitions, commands, and output styles into `.claude/` directories
- Use `setting_sources=["project", "local"]` to load filesystem configurations
- Store reusable workflows as slash commands

**Observability:**

- Implement hooks for audit logging on file operations
- Track tool call counts to detect runaway behavior
- Maintain history files with bounded entry counts (e.g., last 50)

**Quality:**

- Require explicit source quality reasoning in research subagents
- Use tool budgets to prevent excessive API calls
- Implement OODA loops for adaptive research behavior
