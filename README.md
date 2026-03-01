# MeStudio Agent

**Local AI Agent Orchestrator** powered by gpt-oss-20b via LM Studio.

## Quick Start

1. **Install Python 3.11+** and create a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   # or: source .venv/bin/activate  # macOS/Linux
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   playwright install chromium
   ```

3. **Configure environment:**
   ```bash
   copy .env.example .env   # Windows
   # or: cp .env.example .env  # macOS/Linux
   ```
   Edit `.env` to match your LM Studio setup.

4. **Start LM Studio** with gpt-oss-20b loaded.

5. **Run the agent:**
   ```bash
   python -m mestudio.main
   ```

## Architecture

- **Three-tier context management** (working memory → compressed summaries → disk)
- **Sub-agent delegation** for focused tasks (file ops, web search, summarization)
- **Rich terminal UI** with streaming markdown, tool indicators, and context status
- **Browser-based web search** via Playwright (DuckDuckGo + Brave fallback)

See `meStudioAgent-plan.md` for the full implementation plan.
