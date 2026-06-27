from pydantic_settings import BaseSettings, SettingsConfigDict


class RedisSettings(BaseSettings):
    redis_url: str = "redis://localhost:6379/0"
    job_state_ttl_seconds: int = 3600

    model_config = SettingsConfigDict(
        env_prefix="",
        extra="ignore",
    )


redis_settings = RedisSettings()
