import redis

from app.core.redis_settings import redis_settings


redis_client = redis.Redis.from_url(
    redis_settings.redis_url,
    decode_responses=True,
)
