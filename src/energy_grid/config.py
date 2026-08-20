from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "local"
    database_url: str = "sqlite:///data/runtime/energy.db"
    kafka_bootstrap_servers: str = "localhost:29092"
    mlflow_tracking_uri: str = "http://localhost:5000"
    entsoe_token: str | None = None
    event_watermark_minutes: int = Field(default=120, ge=1)
    forecast_stale_after_minutes: int = Field(default=30, ge=1)
    api_p95_target_ms: int = Field(default=250, ge=1)


@lru_cache
def get_settings() -> Settings:
    return Settings()
