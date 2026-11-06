import hashlib
import json
import secrets

import redis

from app.config import REDIS_URL

_client = None

SLIDING_WINDOW_LUA = """
local key = KEYS[1]
local window_ms = tonumber(ARGV[1])
local limit = tonumber(ARGV[2])
local now_parts = redis.call('TIME')
local now = (tonumber(now_parts[1]) * 1000) + math.floor(tonumber(now_parts[2]) / 1000)
redis.call('ZREMRANGEBYSCORE', key, 0, now - window_ms)
local count = redis.call('ZCARD', key)
if count >= limit then
    local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
    local retry_ms = window_ms
    if oldest[2] then retry_ms = math.max(1, window_ms - (now - tonumber(oldest[2]))) end
    redis.call('PEXPIRE', key, window_ms)
    return {0, count, retry_ms}
end
redis.call('ZADD', key, now, tostring(now) .. ':' .. ARGV[3])
redis.call('PEXPIRE', key, window_ms)
return {1, count + 1, 0}
"""


def get_redis():
    global _client
    if not REDIS_URL:
        return None
    if _client is None:
        _client = redis.Redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=2, socket_timeout=2)
    return _client


def allow_request(user_id, action, limit, window_seconds=60):
    client = get_redis()
    if client is None:
        return True, 0
    key = f"pdfqna:ratelimit:{action}:{user_id}"
    allowed, count, retry_ms = client.eval(
        SLIDING_WINDOW_LUA, 1, key, int(window_seconds * 1000),
        int(limit), secrets.token_hex(8)
    )
    return bool(int(allowed)), int(retry_ms)


def cache_get(key):
    client = get_redis()
    if client is None:
        return None
    value = client.get(key)
    return json.loads(value) if value else None


def cache_set(key, value, ttl=300):
    client = get_redis()
    if client is not None:
        client.setex(key, ttl, json.dumps(value, separators=(",", ":")))


def cache_key(*parts):
    raw = ":".join(str(part) for part in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"pdfqna:cache:{digest}"
