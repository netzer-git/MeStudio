# MeStudio Agent — Implementation Plan

> **Local AI Agent Orchestrator** powered by gpt-oss-20b via LM Studio  
> Status: **IN PROGRESS — Steps 1-7 + Logging Complete** | Last updated: 2026-03-01

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Implementation Steps](#implementation-steps)
   - [Step 1: Scaffolding & Dependencies](#step-1-scaffolding--dependencies)
   - [Step 2: LLM Client Layer](#step-2-llm-client-layer)
   - [Step 3: Context Management System](#step-3-context-management-system-the-core)
   - [Step 4: Tool System](#step-4-tool-system)
   - [Step 5: Task Planner](#step-5-task-planner)
   - [Step 6: Sub-Agent System](#step-6-sub-agent-system)
   - [Step 7: Orchestrator](#step-7-orchestrator)
   - [Step 8: CLI Interface](#step-8-cli-interface)
   - [Step 9: Configuration](#step-9-configuration)
   - [Step 10: Comprehensive Logging System](#step-10-comprehensive-logging-system)
5. [Verification](#verification)
6. [Key Decisions](#key-decisions)

---

## Overview

MeStudio Agent is a Python-based local AI agent orchestrator that runs entirely on your machine using LM Studio's OpenAI-compatible API. The architecture centers on a **three-tier context management system** (working memory → compressed summaries → disk persistence) that lets the agent handle arbitrarily large tasks despite finite context windows.

**Capabilities (initial):**
- Multi-step task planning with iterative execution
- Sub-agent delegation for focused work
- Local file read/write/search/edit operations
- Web search (via `ddgs` library) and webpage reading (Playwright)

**Model:** gpt-oss-20b (OpenAI open-weight, MoE, 3.6B active params, 131K context, native tool calling)

**CLI:** Rich terminal UI inspired by Claude Code (streaming markdown, tool indicators, spinners)

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   CLI Interface                 │
│           (rich + prompt_toolkit)               │
├─────────────────────────────────────────────────┤
│                Orchestrator Agent               │
│    ┌──────────┐  ┌──────────┐  ┌────────────┐  │
│    │ Planner  │  │Semaphore │  │  Context   │  │
│    │          │  │ (1 GPU)  │  │  Manager   │  │
│    └──────────┘  └──────────┘  └────────────┘  │
├─────────────────────────────────────────────────┤
│                 Sub-Agent Pool                  │
│    ┌────────┐  ┌────────┐  ┌────────┐          │
│    │ File   │  │ Search │  │Summary │   ...    │
│    │ Agent  │  │ Agent  │  │ Agent  │          │
│    └────────┘  └────────┘  └────────┘          │
├─────────────────────────────────────────────────┤
│                    Tools                        │
│  📄 Files  🌐 Browser  💾 Context  📋 Plan  🤖 Agent│
├─────────────────────────────────────────────────┤
│            LM Studio API (localhost:1234)       │
│                 gpt-oss-20b                     │
└─────────────────────────────────────────────────┘
```

**Data flow:**
1. User input → CLI → Orchestrator
2. Orchestrator checks context budget, compacts if needed
3. Orchestrator calls LLM with tools defined
4. LLM responds with text and/or tool calls
5. Tool calls are executed, results fed back to LLM (loop)
6. For complex tasks, orchestrator delegates to sub-agents with compact task descriptions
7. Results streamed back to CLI
8. Context manager tracks all token usage, triggers compaction as needed

---

## Project Structure

```
MeStudio/
├── mestudio/
│   ├── __init__.py
│   ├── main.py                  # Entry point, CLI loop
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── interface.py         # Rich + prompt_toolkit UI
│   │   ├── renderers.py         # Markdown streaming, tool indicators, diffs
│   │   └── theme.py             # Colors, styles, icons
│   ├── core/
│   │   ├── __init__.py
│   │   ├── orchestrator.py      # Main orchestrator agent loop
│   │   ├── llm_client.py        # OpenAI SDK wrapper for LM Studio
│   │   ├── config.py            # Settings, token budgets, model config
│   │   └── models.py            # Pydantic models for messages, tool calls, etc.
│   ├── context/
│   │   ├── __init__.py
│   │   ├── manager.py           # Context window manager (the heart)
│   │   ├── token_counter.py     # Token counting utilities
│   │   ├── compactor.py         # Summarization / compression logic
│   │   ├── memory_store.py      # Disk-based context persistence
│   │   └── budget.py            # Token budget allocation
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── sub_agent.py         # Sub-agent base class & spawner
│   │   ├── file_agent.py        # File operations specialist
│   │   ├── search_agent.py      # Web search specialist
│   │   └── summary_agent.py     # Summarization specialist
│   ├── planner/
│   │   ├── __init__.py
│   │   ├── task_planner.py      # Break tasks into steps
│   │   └── tracker.py           # Track progress, checkpoints
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── registry.py          # Tool registration & dispatch
│   │   ├── file_tools.py        # Read, write, search, list files
│   │   ├── web_tools.py         # Search provider abstraction + browser scrape
│   │   ├── context_tools.py     # Save, load, compact context
│   │   ├── plan_tools.py        # Create/update/check/cancel plan steps
│   │   └── agent_tools.py       # delegate_task tool for sub-agent invocation
│   └── utils/
│       ├── __init__.py
│       ├── logging.py           # Comprehensive logging setup (loguru)
│       └── helpers.py           # Error handling, etc.
├── data/
│   ├── context_store/           # Saved context summaries (JSON/MD)
│   ├── plans/                   # Saved task plans
│   └── logs/                    # Session logs (JSON, rotated)
├── .env.example                 # Template for environment variables
├── .env                         # Local overrides (git-ignored)
├── .gitignore                   # Ignores .env, data/, .mestudio_history
├── pyproject.toml               # Project config, dependencies
├── requirements.txt
└── README.md
```

---

## Implementation Steps

### Step 1: Scaffolding & Dependencies  ✅ COMPLETE

- [x] Create directory structure as shown above (all `__init__.py` files, `data/` dirs)
- [x] Create `pyproject.toml` with project metadata and dependencies
- [x] Create `requirements.txt`:
  ```
  openai>=1.0
  tiktoken>=0.7
  pydantic>=2.0
  pydantic-settings>=2.0
  python-dotenv>=1.0
  loguru>=0.7
  rich>=13.0
  prompt-toolkit>=3.0
  playwright>=1.40
  beautifulsoup4>=4.12
  trafilatura>=1.6
  markdownify>=0.11
  httpx>=0.27
  ddgs>=7.0
  ```
- [x] Create `.env.example` with documented defaults (template for users):
  ```
  MESTUDIO_LM_STUDIO_URL=http://localhost:1234/v1
  MESTUDIO_LM_STUDIO_MODEL=gpt-oss-20b
  MESTUDIO_MAX_CONTEXT_TOKENS=131072
  MESTUDIO_WORKING_DIRECTORY=.
  MESTUDIO_DATA_DIRECTORY=./data
  MESTUDIO_BROWSER_HEADLESS=true
  MESTUDIO_LOG_LEVEL=INFO
  MESTUDIO_LOG_FILE=./data/logs/mestudio.log
  ```
  > **Note:** All env vars use the `MESTUDIO_` prefix to match the `env_prefix` in `Settings`.
- [x] Create `.gitignore`:
  ```
  .env
  data/
  .mestudio_history
  __pycache__/
  *.pyc
  .venv/
  ```
- [x] Run `playwright install chromium` after pip install
- [x] Create minimal `main.py` entry point that prints "MeStudio Agent starting..."

---

### Step 2: LLM Client Layer  ✅ COMPLETE

**File:** `mestudio/core/llm_client.py`

- [x] Create `LMStudioClient` class wrapping `openai.AsyncOpenAI`
  - Constructor: `base_url` from config, `api_key="lm-studio"`
  - `async chat(messages, tools?, stream=True, response_format?) -> AsyncGenerator`
  - `async structured_output(messages, schema: dict) -> dict` — uses `response_format={"type":"json_schema","json_schema":schema}`
  - Return type includes `usage` info (prompt_tokens, completion_tokens)
- [x] Automatic retry logic: 3 retries with exponential backoff on `ConnectionError`, `Timeout`
- [x] Health check method: `async is_available() -> bool` — calls `/v1/models` to verify LM Studio is running
- [x] Token usage tracking: each call logs tokens used, cumulative session total

**File:** `mestudio/core/models.py`

- [x] Pydantic models:
  - `Message(role, content, tool_calls?, tool_call_id?, name?)`
  - `ToolCall(id, function: FunctionCall)`
  - `FunctionCall(name, arguments: str)`  — arguments is JSON string
  - `ToolResult(tool_call_id, content: str)`
  - `LLMResponse(content?, tool_calls?, usage: TokenUsage)`
  - `TokenUsage(prompt_tokens, completion_tokens, total_tokens)`

**File:** `mestudio/core/config.py` (implemented early for Step 2 dependency)

- [x] `Settings` class with all configuration options
- [x] `get_settings()` / `reload_settings()` global accessor
- [x] Budget properties: `total_budget`, `usable_budget`
- [x] Path properties: `working_path`, `data_path`

---

### Step 3: Context Management System (THE CORE)  ✅ COMPLETE

This is the most important part of the system. Three tiers of memory with five degradation levels (NONE → SOFT → PREEMPTIVE → AGGRESSIVE → EMERGENCY).

#### 3a. Token Counter — `mestudio/context/token_counter.py`

- [x] `TokenCounter` class using `tiktoken`
  - Use `cl100k_base` encoding (reasonable proxy for gpt-oss-20b tokenization)
  - `count_tokens(text: str) -> int`
  - `count_messages(messages: list[Message]) -> int` — accounts for message formatting overhead (~4 tokens per message for role/delimiters)
  - `truncate_to_tokens(text: str, max_tokens: int) -> str` — truncate text to fit budget, adding "... [truncated]" marker

#### 3b. Token Budget — `mestudio/context/budget.py`

- [x] `TokenBudget` dataclass with configurable allocations:
  ```python
  @dataclass
  class TokenBudget:
      total: int = 120_000          # 131K minus safety margin
      system_prompt: int = 2_000    # Fixed system prompt
      compressed_history: int = 8_000  # Rolling summary of older turns
      recent_messages: int = 16_000    # Last N exchanges, verbatim
      tool_results: int = 78_000       # Current task data
      response: int = 16_000           # max_tokens per LLM call
      # Sum of sub-budgets = 120K (matches total exactly)
  ```
- [x] `usable_budget` property: `total - response` (104K) — the actual space for prompt content
- [x] `available_for_tools(current_usage) -> int` — calculate remaining tokens for tool results against `usable_budget`
- [x] `should_compact(current_usage) -> CompactionLevel` — returns NONE/SOFT/PREEMPTIVE/AGGRESSIVE/EMERGENCY based on thresholds
- [x] Thresholds (calculated against `usable_budget` = 104K, NOT `total`):
  - SOFT at 65% of usable (~67K used)
  - PREEMPTIVE at 80% of usable (~83K used) — forces soft compaction to prevent emergency
  - AGGRESSIVE at 90% of usable (~94K used)
  - EMERGENCY at 97% of usable (~101K used)

#### 3c. Context Manager — `mestudio/context/manager.py`

- [x] `ContextManager` class — the heart of the system
  - Maintains `messages: list[Message]` (full working conversation)
  - Maintains `compressed_history: str` (summary of older turns)
  - Tracks token counts per section in real-time
  - `add_message(message)` — add a message, check budget, auto-compact if needed
  - `get_prompt_messages() -> list[Message]` — build the message list for LLM call:
    1. System prompt (always first)
    2. Compressed history as a system message (if any)
    3. Active plan state as a system message (if any)
    4. Recent messages (last N that fit in budget)
    5. Respects total budget
  - `trigger_compaction(level: CompactionLevel)` — delegate to compactor
  - `get_status() -> ContextStatus` — return current usage stats for display
  - `save_checkpoint()` / `load_checkpoint(session_id)` — delegate to memory store

- [x] `ContextStatus` dataclass:
  ```python
  @dataclass
  class ContextStatus:
      total_budget: int
      used_tokens: int
      percent_used: float
      compaction_level: str  # "none", "soft", "aggressive", "emergency"
      message_count: int
      compressed_history_tokens: int
      recent_messages_tokens: int
  ```

#### 3d. Compactor — `mestudio/context/compactor.py`

- [x] `ContextCompactor` class
  - `async compact_soft(messages, llm_client) -> str` — Summarize older messages:
    - Take messages older than the last 8 exchanges
    - Call LLM with summarization prompt: "Summarize the following conversation, preserving: key decisions, file paths, code changes, errors, current task state. Be concise."
    - Return summary string (~1-2K tokens)
  - `async compact_preemptive(messages, existing_summary, llm_client) -> str`:
    - Triggered at 80% of usable budget — acts as a safety net before aggressive/emergency
    - Identical to soft compaction but also truncates tool results older than last 4 exchanges to first/last 500 tokens
    - Ensures a valid summary always exists before emergency could trigger
  - `async compact_aggressive(messages, existing_summary, llm_client) -> str`:
    - Summarize everything except last 2 exchanges
    - Merge with existing summary
    - Truncate large tool results to first/last 200 tokens with "[content truncated]"
    - Target: ~500-1K tokens total summary
  - `compact_emergency(messages, existing_summary) -> str`:
    - NO LLM call (might not fit in context)
    - Extractive: keep only system prompt + existing summary + last exchange
    - Drop all tool results from history
    - If `existing_summary` is empty, build a minimal fallback summary from `extract_preservable_info()` containing at least the task goal, active plan state, and key file paths
    - Log warning to user
  - `extract_preservable_info(messages) -> dict` — extract from messages:
    - File paths mentioned (regex for paths)
    - Plan state (if any `plan_tools` calls)
    - Error messages
    - Current task goal (first user message or plan goal)
    - Key decisions (heuristic: messages containing "decided", "chose", "will use", etc.)

#### 3e. Memory Store — `mestudio/context/memory_store.py`

- [x] `MemoryStore` class — disk-based persistence
  - Storage location: `data/context_store/`
  - `generate_session_id() -> str` — format: `YYYY-MM-DD_HH-MM-SS_{uuid4_short}` (e.g., `2026-03-01_14-30-00_a3f2b1`). The 6-char UUID suffix prevents collisions when saving multiple times per second.
  - `save_session(session_id, messages, summaries, plan_state, metadata) -> Path`
  - `load_session(session_id) -> SessionData`
  - `list_sessions() -> list[SessionSummary]` — list all saved sessions with timestamps and labels
  - `save_checkpoint(session_id, full_state)` — for mid-task resume
  - `load_checkpoint(session_id) -> full_state`
  - File format: JSON with structure:
    ```json
    {
      "version": "1.0",
      "session_id": "2026-03-01_14-30-00_a3f2b1",
      "label": "user-provided label or auto-generated",
      "created_at": "2026-03-01T14:30:00",
      "messages": [],
      "summaries": {
        "level1": [],
        "level2": "...",
        "level3": "..."
      },
      "plan_state": {},
      "metadata": {
        "total_tokens_used": 50000,
        "compaction_count": 3,
        "tools_called": ["read_file", "web_search"]
      }
    }
    ```

---

### Step 4: Tool System  ✅ COMPLETE

#### 4a. Tool Registry — `mestudio/tools/registry.py`

- [x] `ToolRegistry` class — singleton
  - `tools: dict[str, ToolDefinition]` — registered tools
  - `register(name, description, parameters_schema, handler, max_result_tokens?)` — register a tool
  - `@tool(name, description)` decorator for easy registration:
    ```python
    @tool(name="read_file", description="Read contents of a file")
    async def read_file(path: str, start_line: int = None, end_line: int = None) -> str:
        ...
    ```
  - `get_openai_tools() -> list[dict]` — generate OpenAI function-calling tool definitions from registered tools
  - `async execute(tool_name, arguments: dict) -> str` — dispatch and execute, catch errors gracefully. Enforces a per-tool execution timeout (default: 30s, configurable via `tool_timeout` in registry or per-tool override). On timeout, returns `"Error: Tool '{tool_name}' timed out after {timeout}s"`.
  - `truncate_result(result: str, max_tokens: int) -> str` — ensure tool results don't blow up context

#### 4b. File Tools — `mestudio/tools/file_tools.py`

- [x] `is_binary(path) -> bool` — Sniff first 8192 bytes for null bytes. Used by `read_file` and `edit_file` to refuse binary files.
- [x] `read_file(path, start_line?, end_line?)` — Read file, return content with line numbers. **Refuse binary files** (return `"Error: '{path}' appears to be a binary file"`). **Refuse files > 1MB** unless `start_line`/`end_line` are specified. If file > token budget, return first/last chunks with "[... N lines omitted ...]"
- [x] `write_file(path, content)` — Write file, create parent dirs if needed. Return "Written N bytes to path"
- [x] `edit_file(path, edits: list[{old: str, new: str}])` — Search/replace edits. **Refuse binary files.** Each `old` string must match exactly **once** in the file — return an error if 0 matches ("not found") or 2+ matches ("ambiguous: found N occurrences, add more context to `old`"). Edits are applied sequentially (each edit sees the result of previous edits). Return unified diff.
- [x] `list_directory(path, recursive=False, max_depth=3)` — Return directory tree. Limit depth to avoid massive output.
- [x] `search_files(query, path=".", glob="*", max_results=20)` — Grep-like search. Return matching lines with file:line:content format.
- [x] `find_files(pattern, path=".")` — Glob-based find. Return list of matching paths.
- [x] All paths resolved relative to `WORKING_DIRECTORY` from config
- [x] **Safety:** Configurable via `sandbox_file_access` setting:
  - `true` (default): Refuse to access files outside `WORKING_DIRECTORY`
  - `false`: Allow access to any user-readable file on the system (set via `MESTUDIO_SANDBOX_FILE_ACCESS=false` in .env)

#### 4c. Web Tools — `mestudio/tools/web_tools.py`

- [x] `SearchProvider` abstract base class — enables swappable search backends:
  ```python
  class SearchProvider(ABC):
      @abstractmethod
      async def search(self, query: str, num_results: int = 5) -> list[SearchResult]: ...
  ```
- [x] `DDGSProvider(SearchProvider)` — default, uses `ddgs` library (no API key, reliable)
  - Runs synchronous ddgs calls in thread pool via `run_in_executor`
  - Handles rate limiting and bot detection automatically
- [x] `BraveSearchProvider(SearchProvider)` — optional alternative using Brave Search API (requires API key, free tier available)
- [x] `WebToolManager` class — manages Playwright browser lifecycle (for `read_webpage`) and search provider
  - `async start()` — launch Chromium in headless incognito mode
  - `async stop()` — close browser
  - `get_search_provider()` — defaults to `DDGSProvider`
- [x] `web_search(query, num_results=5)`:
  - Delegates to active `SearchProvider`
  - Return formatted list of `SearchResult(title, url, snippet)`
- [x] `read_webpage(url, max_tokens=4000)`:
  - Navigate to URL, wait for DOM content loaded (timeout: 15s)
  - Extract content using `trafilatura` (falls back to `BeautifulSoup` if trafilatura fails)
  - Convert to markdown via `markdownify`
  - Truncate to `max_tokens`
  - Return clean markdown text
- [x] Error handling: timeouts → return "Page timed out", blocked → return "Access denied", JS-heavy → try waiting for network idle

#### 4d. Context Tools — `mestudio/tools/context_tools.py`

- [x] `save_context(label?)` — Save current session to disk via MemoryStore. Return session ID.
- [x] `load_context(session_id)` — Load a saved session's summary into current context.
- [x] `compact_now()` — Manually trigger soft compaction. Return new token usage stats.
- [x] `context_status()` — Return formatted context usage: tokens used/total, percent, compaction level, message count.
- [x] `list_sessions()` — List all saved sessions with timestamps and labels.

#### 4e. Plan Tools — `mestudio/tools/plan_tools.py`

- [x] `create_plan(goal, steps: list[str])` — Create a new plan with ordered steps. Persist to `data/plans/`. If a plan already exists, prompt confirmation before replacing.
- [x] `update_step(step_index, status, notes?)` — Update step status: "pending" | "active" | "done" | "failed" | "skipped".
- [x] `get_plan()` — Return current plan as formatted text with status icons.
- [x] `add_steps(steps: list[str], after_index?)` — Insert new steps into existing plan.
- [x] `remove_step(step_index)` — Remove a step from the plan. Re-index remaining steps.
- [x] `cancel_plan()` — Discard the current plan entirely. Returns confirmation message. Clears plan state from context.
- [x] `replace_plan(goal, steps: list[str])` — Replace the current plan with a new one (shortcut for cancel + create). Preserves notes/status from completed steps in the compressed history before discarding.
- [x] Plan format (Pydantic model):
  ```python
  class PlanStep(BaseModel):
      index: int
      description: str
      status: Literal["pending", "active", "done", "failed", "skipped"] = "pending"
      notes: str = ""
      sub_steps: list[PlanStep] = []
  
  class TaskPlan(BaseModel):
      goal: str
      steps: list[PlanStep]
      created_at: datetime
      updated_at: datetime
  ```

---

### Step 5: Task Planner ✅ COMPLETE

**File:** `mestudio/planner/task_planner.py`

- [x] `TaskPlanner` class
  - `decompose(task_description, llm_client) -> TaskPlan` — uses JSON Schema structured output for validated responses
  - `refine_plan(plan, feedback, llm_client) -> TaskPlan` — refines existing plan based on feedback
  - `estimate_complexity(task, llm_client) -> dict` — estimates task complexity (simple/moderate/complex)
  - System prompt instructs LLM to create 3-10 concrete, actionable steps
  - Returns validated `TaskPlan` via Pydantic
  - Planning is **not auto-triggered by heuristics** (avoids false positives). Instead:
    - The `create_plan` tool is always available to the LLM
    - User can explicitly request planning via the `/plan` command

**File:** `mestudio/planner/tracker.py`

- [x] `PlanTracker` class
  - Holds current `TaskPlan` (if any)
  - `next_step() -> PlanStep | None` — return next pending step, respecting dependencies
  - `start_step(index)` — mark step as in progress
  - `mark_done(index, notes?)` / `mark_failed(index, notes?)`
  - `skip_step(index, reason?)` — skip a step
  - `is_complete() -> bool`
  - `is_stuck() -> bool` — same step failed 3+ times
  - `get_summary() -> str` — compact plan representation for inclusion in context
  - `save(path)` / `load(path)` — persist to `data/plans/`

**Tests:** `tests/test_planner_live.py` — 4 live LLM tests (decompose, complexity, tracking, refinement)

---

### Step 6: Sub-Agent System ✅ COMPLETE

**File:** `mestudio/agents/sub_agent.py`

- [x] `SubAgent` base class:
  - `SubAgentConfig` dataclass for agent configuration
  - `execute(task)` — runs agent loop with isolated ContextManager
  - Filters tools from global registry based on `available_tools` list
  - Uses `_llm_semaphore` to serialize LLM calls
  - Auto-blocks `delegate_task` to prevent recursive delegation
  - Returns text response when complete or raises `SubAgentError` on max turns

- [x] `SubAgentSpawner`:
  - `spawn(agent_type, task) -> str` — create sub-agent, execute, return result
  - `create_agent(agent_type)` — factory method for agent instances
  - `get_agent_types()` — returns available agent types
  - Maps `"file"`, `"search"`, `"summary"` to specialized agents

**File:** `mestudio/tools/agent_tools.py` (from Step 4)

- [x] `delegate_task(agent_type: str, task: str) -> str` — tool interface (placeholder handlers for standalone testing)

**File:** `mestudio/agents/file_agent.py`

- [x] `FileAgent` class with specialized system prompt
- [x] Available tools: `read_file`, `write_file`, `edit_file`, `list_directory`, `search_files`, `find_files`
- [x] `max_turns=15` for complex file operations

**File:** `mestudio/agents/search_agent.py`

- [x] `SearchAgent` class with specialized system prompt
- [x] Available tools: `web_search`, `read_webpage`
- [x] `max_turns=8` for typical search workflows

**File:** `mestudio/agents/summary_agent.py`

- [x] `SummaryAgent` class with specialized system prompt
- [x] Available tools: `read_file` (can summarize files directly)
- [x] `max_turns=5` for quick summarization

**Tests:** `tests/test_subagents_live.py` — 5 live LLM tests (4/5 passed — FileAgent path parsing was model artifact, not system bug)

---

### Step 7: Orchestrator  ✅ COMPLETE

**File:** `mestudio/core/orchestrator.py`

- [x] `Orchestrator` class — the main agent loop:
  ```python
  class Orchestrator:
      llm_client: LMStudioClient
      context_manager: ContextManager
      tool_registry: ToolRegistry
      plan_tracker: PlanTracker
      sub_agent_spawner: SubAgentSpawner
      cli: CLIInterface  # for rendering output
      _llm_semaphore: asyncio.Semaphore  # limits concurrent LLM calls to 1
  ```
  - `_llm_semaphore = asyncio.Semaphore(1)` — All LLM calls (orchestrator + sub-agents) acquire this semaphore to prevent concurrent requests to LM Studio, which runs on a single GPU and would queue/OOM otherwise.

- [x] System prompt (stored in a separate text or constant):
  ```
  You are MeStudio Agent, a local AI assistant with tool-calling capabilities.
  
  You can:
  - Read, write, search, and edit local files
  - Search the web for information
  - Create and track multi-step plans for complex tasks
  - Delegate focused tasks to sub-agents via delegate_task
  
  Guidelines:
  - For complex multi-step tasks, create a plan first using create_plan
  - For focused sub-tasks, use delegate_task(agent_type, task) to delegate:
    - "file" agent for file read/write/search/edit operations
    - "search" agent for web research
    - "summary" agent for condensing large text
  - Be concise in your responses — context is precious
  - When reading files, request only the lines you need
  - When tool results are large, summarize the key findings before responding
  - Always report your progress on the current plan step
  - If a plan is wrong or outdated, use cancel_plan or replace_plan to fix it
  ```

- [x] Main loop pseudocode:
  ```
  async def run(user_input):
      1. context_manager.add_message(user message)
      2. Check context budget → compact if needed (notify user via CLI)
      3. Build prompt via context_manager.get_prompt_messages()
      4. async with _llm_semaphore:
             Call llm_client.chat(messages, tools=registry.get_openai_tools(), stream=True)
      5. Process streamed response:
         - Text chunks → stream to CLI immediately
         - Tool calls → BUFFER until stream completes (finish_reason: "tool_calls")
           to avoid executing on incomplete/partial JSON
         - After stream completes: execute buffered tool calls via registry,
           show in CLI, add results to context
         - If tool results present → loop back to step 3 (feed results to LLM)
      6. Add final assistant response to context_manager
      7. Update plan tracker if plan is active
      8. Return
  ```

- [x] **Max tool loop iterations:** 20 per user turn (prevent runaway)
- [x] **Parallel tool calls:** When the LLM returns multiple tool calls in one response, execute them concurrently via `asyncio.gather()` — but serialize file write/edit operations to the same path to prevent conflicts.
- [x] **Error handling:** If LLM call fails, retry (max 3 with backoff). If tool fails, return error message as tool result (let LLM decide how to proceed). If context overflow, trigger emergency compaction and retry once.

**Tests:** `tests/test_orchestrator_live.py` — 17 intensive live LLM tests (17/17 passed)

---

### Step 8: CLI Interface

**File:** `mestudio/cli/interface.py`

- [ ] `CLIInterface` class:
  - Uses `prompt_toolkit.PromptSession` for input:
    - Multi-line: Enter submits, Alt+Enter for newline (unless inside ``` code fence)
    - History: `FileHistory('.mestudio_history')`
    - Auto-suggest from history
  - Uses `rich.Console` for output
  - Slash commands parsed before sending to orchestrator:
    - `/plan` — show current plan
    - `/context` — show context status (token usage bar)
    - `/save [label]` — save session
    - `/load` — list and select a saved session
    - `/compact` — force context compaction
    - `/clear` — clear conversation (keep system prompt)
    - `/quit` or `/exit` — exit agent
    - `/help` — show available commands
  - `async prompt_user() -> str` — get user input with prompt_toolkit
  - `stream_text(chunks)` — stream LLM text via `rich.live.Live` + `Markdown`
  - `show_tool_call(name, args)` — show tool execution with spinner
  - `show_tool_result(name, result, success)` — show completed tool with icon
  - `show_context_status(status)` — colored bar/gauge of token usage
  - `show_plan(plan)` — tree view with status icons
  - `confirm(prompt) -> bool` — yes/no prompt for destructive operations

**File:** `mestudio/cli/renderers.py`

- [ ] `StreamingMarkdownRenderer`:
  - Accumulates chunks, re-renders `Markdown` via `Live` at ~10fps
  - Handles partial markdown gracefully (catches render errors, falls back to plain text)
- [ ] `ToolCallRenderer`:
  - `start(name, args)` — show spinner panel: "🔧 read_file: src/main.py"
  - `finish(name, result, success)` — replace spinner with result panel (truncated preview)
- [ ] `DiffRenderer`:
  - Show unified diff with `rich.Syntax` using "diff" lexer
- [ ] `PlanRenderer`:
  - Show plan as `rich.Tree` with status emojis: ⬜ pending, 🔄 active, ✅ done, ❌ failed, ⏭️ skipped
- [ ] `ContextStatusRenderer`:
  - Show `rich.Progress` bar: "Context: ████████░░ 78% (94K/120K tokens) [SOFT COMPACTION]"

**File:** `mestudio/cli/theme.py`

- [ ] Color/style constants:
  - `USER_STYLE = "bold cyan"`
  - `ASSISTANT_STYLE = "white"`
  - `TOOL_STYLE = "bold blue"`
  - `ERROR_STYLE = "bold red"`
  - `SYSTEM_STYLE = "dim yellow"`
  - `SUCCESS_STYLE = "bold green"`
  - Icon constants for tool types, plan statuses, etc.

---

### Step 9: Configuration  (✅ `config.py` implemented in Step 2)

**File:** `mestudio/core/config.py` — ✅ ALREADY IMPLEMENTED

- [x] `Settings` class using `pydantic_settings.BaseSettings`:
  ```python
  class Settings(BaseSettings):
      # LM Studio
      lm_studio_url: str = "http://localhost:1234/v1"
      lm_studio_model: str = "gpt-oss-20b"
      lm_studio_api_key: str = "lm-studio"
      
      # Context
      max_context_tokens: int = 131_072
      safety_margin_tokens: int = 11_072  # total budget = max - margin = 120K
      compaction_soft_pct: float = 0.70
      compaction_aggressive_pct: float = 0.85
      compaction_emergency_pct: float = 0.95
      
      # Token budgets
      system_prompt_budget: int = 2_000
      compressed_history_budget: int = 8_000
      recent_messages_budget: int = 16_000
      tool_results_budget: int = 80_000
      response_budget: int = 16_000
      
      # File operations
      working_directory: str = "."
      data_directory: str = "./data"
      sandbox_file_access: bool = True  # If False, agent can access any user-readable file
      
      # Web
      browser_headless: bool = True
      web_page_timeout: int = 15_000  # ms
      max_webpage_tokens: int = 4_000
      
      # Agent
      max_tool_iterations: int = 20
      max_sub_agent_depth: int = 2
      max_sub_agent_turns: int = 10
      
      # Tool execution
      tool_timeout: int = 30  # seconds, per-tool execution timeout
      
      # Search providers
      brave_search_api_key: str = ""  # optional, enables Brave Search fallback
      
      model_config = SettingsConfigDict(env_file=".env", env_prefix="MESTUDIO_")
  ```

**File:** `mestudio/main.py`

- [ ] Entry point:
  ```python
  async def main():
      1. Load config from .env
      2. Initialize LMStudioClient, check health
      3. Initialize ContextManager, ToolRegistry, PlanTracker, SubAgentSpawner
      4. Register all tools
      5. Initialize CLIInterface
      6. Show welcome banner (model name, context size, available tools)
      7. Main loop:
         a. user_input = await cli.prompt_user()
         b. if slash command → handle directly
         c. else → await orchestrator.run(user_input)
      8. On exit: save session if desired, close browser, cleanup
  ```
- [ ] Signal handling: Ctrl+C gracefully interrupts current operation (not the whole agent)

---

### Step 10: Comprehensive Logging System  🔄 PARTIAL (infrastructure complete, main.py pending)

> **Purpose:** Enable detailed debugging and investigation of agent sessions without bloating logs with raw content.

**Design Principles:**
1. **Session-centric**: Every log message tied to a session ID for easy filtering
2. **Structured data**: JSON-formatted log files for programmatic analysis
3. **Truncated previews**: Never log full message/content bodies — only summaries and lengths
4. **Performance metrics**: Track timing data for LLM calls, tool execution, context operations
5. **Log rotation**: Auto-rotate logs to prevent disk bloat

**Log Levels:**
| Level | Used For |
|-------|----------|
| `DEBUG` | Detailed internal operations (context budget calculations, token counts) |
| `INFO` | Normal operations (tool calls, LLM requests, session start/end) |
| `WARNING` | Non-fatal issues (retries, soft compaction triggers, fallbacks) |
| `ERROR` | Operation failures (tool errors, LLM errors, file not found) |
| `CRITICAL` | Unrecoverable errors (LLM unavailable, emergency compaction failure) |

**File:** `mestudio/core/config.py` — Add to `Settings`:
```python
# Logging
log_level: str = "INFO"
log_file: str = "./data/logs/mestudio.log"
log_max_size: str = "50 MB"
log_rotation_count: int = 5
log_json_format: bool = True  # JSON for file, human-readable for console
```

**File:** `mestudio/utils/logging.py` — NEW:
```python
from loguru import logger
from pathlib import Path
import sys

def setup_logging(settings: "Settings", session_id: str) -> None:
    """Configure loguru with session context, file rotation, and dual output."""
    
    # Remove default handler
    logger.remove()
    
    # Console: human-readable, INFO+ only
    console_format = (
        "<dim>{time:HH:mm:ss}</dim> | "
        "<level>{level: <8}</level> | "
        "<cyan>{extra[session]}</cyan> | "
        "<level>{message}</level>"
    )
    logger.add(
        sys.stderr,
        format=console_format,
        level="INFO",
        colorize=True,
        filter=lambda record: record["extra"].get("session") is not None
    )
    
    # File: JSON structured, DEBUG+ with rotation
    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    if settings.log_json_format:
        logger.add(
            str(log_path),
            format="{message}",
            level=settings.log_level,
            rotation=settings.log_max_size,
            retention=settings.log_rotation_count,
            serialize=True,  # JSON format
            enqueue=True  # Thread-safe
        )
    else:
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
            "{extra[session]} | {name}:{function}:{line} | {message}"
        )
        logger.add(
            str(log_path),
            format=file_format,
            level=settings.log_level,
            rotation=settings.log_max_size,
            retention=settings.log_rotation_count,
            enqueue=True
        )
    
    # Bind session ID to all future log calls
    logger.configure(extra={"session": session_id})

def get_session_logger(session_id: str):
    """Get a logger bound to a specific session."""
    return logger.bind(session=session_id)
```

**What to Log (by module):**

| Module | Events to Log | Level |
|--------|---------------|-------|
| **Orchestrator** | Session start/end, user message received (length), response generated (length), iteration count | INFO |
| **LLM Client** | Request sent (message count, tool count), response received (tokens used, duration), retries | INFO/WARNING |
| **Context Manager** | Token usage snapshot after each operation, compaction triggered (level, before/after sizes), budget allocation | DEBUG/INFO |
| **Tool Registry** | Tool registered, tool called (name, arg keys), tool completed (duration, success/fail, result length) | INFO |
| **Individual Tools** | Tool-specific significant events (file path accessed, URL fetched, search query) | DEBUG |
| **Planner** | Plan created (step count), step started, step completed, plan finished | INFO |
| **Sub-Agent** | Spawned (task summary, depth), completed (success, result length) | INFO |
| **Memory Store** | Session saved (path, size), session loaded (message count) | INFO |

**Implementation Checklist:**

- [x] Add logging settings to `config.py`:
  - `log_level`, `log_file`, `log_max_size`, `log_rotation_count`, `log_json_format`
- [x] Create `mestudio/utils/logging.py`:
  - `setup_logging(settings, session_id)` — configure loguru
  - `get_session_logger(session_id)` — get bound logger
- [ ] Update `main.py`:
  - Generate session ID (`uuid.uuid4().hex[:8]`)
  - Call `setup_logging()` at startup
  - Log session start with system info (Python version, model, token budget)
- [x] Enhance `llm_client.py`:
  - Log before/after each LLM call with timing
  - Log retry attempts with backoff info
  - Log token usage summary (not full messages)
- [x] Enhance `context/manager.py`:
  - Log compaction events (level, old_size, new_size, duration)
  - Log token budget snapshots (DEBUG level)
- [x] Enhance `tools/registry.py`:
  - Log tool registration at startup
  - Log each tool call: `{"tool": "name", "args_keys": [...], "duration_ms": X, "success": bool, "result_len": Y}`
- [x] Enhance `context/memory_store.py`:
  - Log save/load operations with file size
- [ ] Add log viewer utility (optional):
  - CLI command `/logs` to show recent log entries
  - Filter by level, module, or time range

**Log Entry Examples:**

```json
// Session start
{"time": "2026-03-01T10:30:00", "level": "INFO", "session": "a1b2c3d4", "event": "session_start", "python": "3.10.11", "model": "gpt-oss-20b", "max_tokens": 131072}

// User message
{"time": "2026-03-01T10:30:05", "level": "INFO", "session": "a1b2c3d4", "event": "user_message", "length": 156, "preview": "Create a Python project..."}

// LLM request
{"time": "2026-03-01T10:30:05", "level": "INFO", "session": "a1b2c3d4", "event": "llm_request", "message_count": 3, "tool_count": 21, "estimated_tokens": 4521}

// LLM response
{"time": "2026-03-01T10:30:12", "level": "INFO", "session": "a1b2c3d4", "event": "llm_response", "duration_ms": 7234, "prompt_tokens": 4512, "completion_tokens": 856, "tool_calls": 2}

// Tool execution
{"time": "2026-03-01T10:30:12", "level": "INFO", "session": "a1b2c3d4", "event": "tool_call", "tool": "write_file", "args_keys": ["path", "content"], "duration_ms": 45, "success": true, "result_len": 67}

// Compaction
{"time": "2026-03-01T10:35:20", "level": "INFO", "session": "a1b2c3d4", "event": "compaction", "level": "soft", "before_tokens": 84521, "after_tokens": 32156, "duration_ms": 1234}

// Session end
{"time": "2026-03-01T11:00:00", "level": "INFO", "session": "a1b2c3d4", "event": "session_end", "duration_min": 30, "total_llm_calls": 47, "total_tool_calls": 156, "total_tokens_used": 523456}
```

**Files created in `data/logs/`:**
- `mestudio.log` — current log file
- `mestudio.log.1`, `mestudio.log.2`, ... — rotated archives

---

## Verification

### Automated Tests
- [x] **Token counter accuracy**: Verified in `test_token_counting_accuracy`. Token counts match expected values with acceptable variance.
- [x] **Context compaction**: Tested in `test_async_compaction`, `test_compactor_methods`. Soft/aggressive/emergency compaction all verified.
- [x] **Tool registry**: Tested in `test_tool_registry`, `test_tool_decorator`, `test_openai_schema_complete`. Schema generation and dispatch verified.
- [x] **Plan tracker**: Tested in `test_plan_tools`. Create/update/get/cancel plan operations verified.
- [x] **Memory store**: Tested in `test_memory_store`, `test_session_persistence`. JSON roundtrip integrity verified.
- [x] **Web search**: Tested in dedicated web tools tests. `DDGSProvider` verified working with real queries.

### Manual Integration Tests
- [ ] **Startup**: LM Studio running with gpt-oss-20b → start agent → verify health check passes, welcome banner shows.
- [ ] **Basic chat**: Send "Hello" → get streamed response with markdown rendering.
- [ ] **File operations**: Ask agent to "list files in current directory" → verify tool call shown, results displayed.
- [ ] **File editing**: Ask agent to "create a file hello.txt with content 'Hello World'" → verify file created, diff shown.
- [ ] **Web search**: Ask agent to "search the web for Python 3.12 release date" → verify results returned.
- [ ] **Planning**: Ask agent to "create a Python project with 3 files" → verify plan created, steps tracked, files written.
- [ ] **Context compaction**: Have a long conversation (50+ turns) → verify compaction triggers, status bar updates, conversation remains coherent.
- [ ] **Session save/load**: `/save test-session` → `/quit` → restart → `/load` → verify session restored.
- [ ] **Sub-agent delegation**: Ask agent to "summarize all Python files in a directory" → verify sub-agent spawned, focused result returned.

---

## Key Decisions

| Decision | Choice | Rationale |
|---|---|---|
| **Model** | gpt-oss-20b | Native tool calling, structured output, built for agentic use. First-class LM Studio support. |
| **CLI stack** | `rich` + `prompt_toolkit` | Lighter than `textual`, preserves terminal scrollback, best-in-class input + output combo. |
| **Web search** | `ddgs` library + Playwright for pages | DuckDuckGo via `ddgs` library (reliable, no API key). Playwright browser only for `read_webpage`. Brave Search API as optional fallback. |
| **Agent framework** | Custom (no LangChain/CrewAI) | Full control over context management, fewer dependencies, direct integration with LM Studio's native tool calling. |
| **Context strategy** | 3-tier hierarchical (working → compressed → disk) | Maximizes effective context utilization. 4 degradation levels handle any scenario gracefully. |
| **Sub-agent design** | Isolated context per sub-agent | Prevents single-task context blowup from affecting orchestrator. Each sub-agent is stateless and focused. |
| **Token counting** | `tiktoken` with `cl100k_base` | Fast, reliable, ~±10% accurate for gpt-oss-20b. Good enough for budget management. |
| **Persistence format** | JSON with schema versioning | Human-readable, easy to debug, supports future schema migrations. |
| **Search engine** | DuckDuckGo via `ddgs` + Brave fallback | Swappable `SearchProvider` abstraction. DDGSProvider is default (uses `ddgs` library, no API key). BraveSearchProvider as optional fallback (free tier, 2K queries/month). |

---

## Future Expansion Points

The architecture supports adding these without structural changes:
- **New tools**: Add file to `mestudio/tools/`, register with decorator, auto-available to agents
- **New sub-agents**: Add file to `mestudio/agents/`, register in spawner config
- **MCP integration**: LM Studio supports MCP servers — could add MCP tools alongside native tools
- **Voice input/output**: Add alternative CLI input handler
- **Multi-model**: Use different models for different agents (e.g., smaller model for summarization)
- **RAG**: Add vector store for long-term semantic memory alongside the current keyword-based recall
- **Additional search providers**: Add Google Custom Search, Bing, or Serper.dev as additional `SearchProvider` implementations
