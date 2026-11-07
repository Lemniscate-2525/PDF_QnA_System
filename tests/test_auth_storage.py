import pytest

from app.auth_storage import RedisAuthStorage


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.ttls = {}

    def get(self, key):
        return self.values.get(key)

    def setex(self, key, ttl, value):
        self.values[key] = value
        self.ttls[key] = ttl

    def delete(self, *keys):
        for key in keys:
            self.values.pop(key, None)
            self.ttls.pop(key, None)

    def scan_iter(self, match, count=100):
        prefix = match[:-1]
        return iter([key for key in self.values if key.startswith(prefix)])


def test_storage_round_trip_and_ttl(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr("app.auth_storage.get_redis", lambda: fake)
    storage = RedisAuthStorage("a" * 48)
    storage.set_item("code-verifier", "secret-verifier")
    storage.set_item("session", '{"access_token":"example"}')
    assert storage.get_item("code-verifier") == "secret-verifier"
    assert fake.ttls[storage._key("code-verifier")] == 600
    assert fake.ttls[storage._key("session")] == 60 * 60 * 24 * 30


def test_storage_namespaces_flows(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr("app.auth_storage.get_redis", lambda: fake)
    first = RedisAuthStorage("a" * 48)
    second = RedisAuthStorage("b" * 48)
    first.set_item("code-verifier", "first")
    second.set_item("code-verifier", "second")
    assert first.get_item("code-verifier") == "first"
    assert second.get_item("code-verifier") == "second"


def test_storage_clear_only_removes_its_flow(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr("app.auth_storage.get_redis", lambda: fake)
    first = RedisAuthStorage("a" * 48)
    second = RedisAuthStorage("b" * 48)
    first.set_item("code-verifier", "first")
    second.set_item("code-verifier", "second")
    first.clear()
    assert first.get_item("code-verifier") is None
    assert second.get_item("code-verifier") == "second"


def test_invalid_flow_id_rejected(monkeypatch):
    monkeypatch.setattr("app.auth_storage.get_redis", lambda: FakeRedis())
    with pytest.raises(ValueError):
        RedisAuthStorage("not-a-valid-flow")
