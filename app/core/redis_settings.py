from pydantic_settings import BaseSettings, SettingsConfigDict


class RedisSettings(BaseSettings):
    """
    Read Redis connection and job-state expiry settings from the environment.

    Attributes
    ----------
    redis_url : str
        Redis connection URL. Default is ``'redis://localhost:6379/0'``.
    job_state_ttl_seconds : int
        Lifetime of Redis job-state keys, in seconds. Default is ``48 * 60 * 60``.
    """
    redis_url: str = "redis://localhost:6379/0"

    model_config = SettingsConfigDict(
        env_prefix="",
        extra="ignore",
    )

    job_state_ttl_seconds: int = 48 * 60 * 60


redis_settings = RedisSettings()
