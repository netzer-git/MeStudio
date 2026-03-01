"""Settings, token budgets, model configuration."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.
    
    All environment variables use the MESTUDIO_ prefix.
    Example: MESTUDIO_LM_STUDIO_URL=http://localhost:1234/v1
    """

    # LM Studio connection
    lm_studio_url: str = "http://localhost:1234/v1"
    lm_studio_model: str = "gpt-oss-20b"
    lm_studio_api_key: str = "lm-studio"

    # Context window
    max_context_tokens: int = 131_072
    safety_margin_tokens: int = 11_072  # total budget = max - margin = 120K

    # Compaction thresholds (percentages of usable budget)
    compaction_soft_pct: float = 0.65
    compaction_preemptive_pct: float = 0.80
    compaction_aggressive_pct: float = 0.90
    compaction_emergency_pct: float = 0.97

    # Token budgets (must sum to max_context_tokens - safety_margin_tokens = 120K)
    system_prompt_budget: int = 2_000
    compressed_history_budget: int = 8_000
    recent_messages_budget: int = 16_000
    tool_results_budget: int = 78_000
    response_budget: int = 16_000

    # File operations
    working_directory: str = "."
    data_directory: str = "./data"

    # Web / browser
    browser_headless: bool = True
    web_page_timeout: int = 15_000  # milliseconds
    max_webpage_tokens: int = 4_000

    # Agent limits
    max_tool_iterations: int = 20
    max_sub_agent_depth: int = 2
    max_sub_agent_turns: int = 10

    # Tool execution
    tool_timeout: int = 30  # seconds, per-tool execution timeout

    # Search providers
    brave_search_api_key: str = ""  # optional, enables Brave Search fallback

    # Logging
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_file: str = "./data/logs/mestudio.log"
    log_max_size: str = "50 MB"  # log rotation size
    log_rotation_count: int = 5  # number of rotated files to keep
    log_json_format: bool = True  # JSON for file (human-readable for console)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="MESTUDIO_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def total_budget(self) -> int:
        """Total token budget after safety margin."""
        return self.max_context_tokens - self.safety_margin_tokens

    @property
    def usable_budget(self) -> int:
        """Usable budget for prompt content (total minus response reservation)."""
        return self.total_budget - self.response_budget

    @property
    def working_path(self) -> Path:
        """Working directory as a Path object."""
        return Path(self.working_directory).resolve()

    @property
    def data_path(self) -> Path:
        """Data directory as a Path object."""
        return Path(self.data_directory).resolve()

    @property
    def log_path(self) -> Path:
        """Log file as a Path object."""
        return Path(self.log_file).resolve()


# Global settings instance (lazy-loaded)
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the global settings instance, creating it if needed."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Force reload settings from environment."""
    global _settings
    _settings = Settings()
    return _settings
