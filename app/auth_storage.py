import hashlib
import json

from app.config import APP_ENV
from app.redis_store import get_redis


class RedisAuthStorage:
    """Namespaced, expiring Supabase auth storage for one OAuth PKCE round trip."""

    def __init__(self, flow_id: str):
        if not flow_id or len(flow_id) != 48:
            raise ValueError("Invalid OAuth flow identifier.")
        self.flow_id = flow_id
        self.client = get_redis()
        if self.client is None:
            if APP_ENV == "production":
                raise RuntimeError("Redis is required for persistent OAuth PKCE storage.")
            raise RuntimeError("Configure Redis to use Google sign-in.")

    def _key(self, key: str) -> str:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return f"documind:oauth:{self.flow_id}:{digest}"

    def get_item(self, key: str):
        value = self.client.get(self._key(key))
        return value if value is not None else None

    def set_item(self, key: str, value: str) -> None:
        # PKCE verifiers must survive only long enough for the OAuth callback.
        # The session itself is retained for up to 30 days and refreshed by Supabase.
        ttl = 600 if "code-verifier" in key.lower() else 60 * 60 * 24 * 30
        self.client.setex(self._key(key), ttl, value)

    def remove_item(self, key: str) -> None:
        self.client.delete(self._key(key))

    def clear(self) -> None:
        pattern = f"documind:oauth:{self.flow_id}:*"
        keys = list(self.client.scan_iter(match=pattern, count=100))
        if keys:
            self.client.delete(*keys)
