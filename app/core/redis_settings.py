from pydantic_settings import BaseSettings, SettingsConfigDict


class RedisSettings(BaseSettings):
    redis_url: str = "redis://localhost:6379/0"

    model_config = SettingsConfigDict(
        env_prefix="",
        extra="ignore",
    )

    job_state_ttl_seconds: int = 48 * 60 * 60


redis_settings = RedisSettings()
