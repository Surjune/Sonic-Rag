"""Tests for API key discovery and rotation."""

from __future__ import annotations

import httpx
import pytest

from app.credentials import ROTATABLE_STATUSES, KeyRing, collect_keys
from app.exceptions import TranscriptionError
from app.stt_service import SarvamSttService


class TestDiscovery:
    def test_finds_primary_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_KEY", "primary")
        assert collect_keys("TEST_KEY") == ["primary"]

    def test_finds_backup_suffix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_KEY", "primary")
        monkeypatch.setenv("TEST_KEY_BACKUP", "second")
        assert collect_keys("TEST_KEY") == ["primary", "second"]

    def test_backup_suffix_is_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Windows env vars are case-insensitive and Linux ones are not.

        A key written as FOO_Backup works locally and silently disappears on a
        Linux host, which is exactly the bug that only shows up in production.
        """
        monkeypatch.setenv("TEST_KEY", "primary")
        monkeypatch.setenv("TEST_KEY_Backup", "second")
        assert collect_keys("TEST_KEY") == ["primary", "second"]

    def test_accepts_fallback_naming(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_KEY", "primary")
        monkeypatch.setenv("TEST_KEY_FALLBACK", "second")
        assert collect_keys("TEST_KEY") == ["primary", "second"]

    def test_duplicate_keys_collapse(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A repeated key would waste a rotation on a credential already failing."""
        monkeypatch.setenv("TEST_KEY", "same")
        monkeypatch.setenv("TEST_KEY_BACKUP", "same")
        assert collect_keys("TEST_KEY") == ["same"]

    def test_blank_values_dropped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_KEY", "primary")
        monkeypatch.setenv("TEST_KEY_BACKUP", "   ")
        assert collect_keys("TEST_KEY") == ["primary"]

    def test_missing_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TEST_KEY", raising=False)
        assert collect_keys("TEST_KEY") == []


class TestKeyRing:
    def test_starts_on_primary(self) -> None:
        ring = KeyRing("K", ["a", "b"])
        assert ring.active == "a"
        assert ring.label == "K[1/2]"

    def test_rotate_advances(self) -> None:
        ring = KeyRing("K", ["a", "b"])
        assert ring.rotate() is True
        assert ring.active == "b"
        assert ring.label == "K[2/2]"

    def test_rotate_returns_false_when_exhausted(self) -> None:
        ring = KeyRing("K", ["a"])
        assert ring.rotate() is False

    def test_reset_returns_to_primary(self) -> None:
        ring = KeyRing("K", ["a", "b"])
        ring.rotate()
        ring.reset()
        assert ring.active == "a"

    def test_unconfigured_ring(self) -> None:
        ring = KeyRing("K", [])
        assert not ring.configured
        assert ring.active == ""
        assert ring.label == "K[none]"

    def test_label_never_contains_the_key(self) -> None:
        """Labels are logged; keys must never be."""
        ring = KeyRing("K", ["super-secret-value"])
        assert "super-secret" not in ring.label


class TestRotationTriggers:
    @pytest.mark.parametrize("status", sorted(ROTATABLE_STATUSES))
    def test_auth_and_rate_limit_rotate(self, status: int) -> None:
        assert status in ROTATABLE_STATUSES

    @pytest.mark.parametrize("status", [400, 404, 500, 502, 503])
    def test_other_failures_do_not_rotate(self, status: int) -> None:
        """A 500 means the upstream is unwell; another key will not persuade it."""
        assert status not in ROTATABLE_STATUSES


class TestProviderRotation:
    @pytest.mark.asyncio
    async def test_rotates_to_backup_on_rate_limit(self) -> None:
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            key = request.headers.get("api-subscription-key", "")
            seen.append(key)
            if key == "first":
                return httpx.Response(429, json={"error": "rate limited"})
            return httpx.Response(200, json={"transcript": "ok", "language_code": "hi-IN"})

        service = SarvamSttService(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        service._keys = KeyRing("SARVAM_API_KEY", ["first", "second"])

        result = await service.transcribe(b"RIFF" + b"\x00" * 100, include_native=False)
        assert result.english_text == "ok"
        assert "first" in seen and "second" in seen
        await service.aclose()

    @pytest.mark.asyncio
    async def test_gives_up_when_all_keys_rejected(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"error": "bad key"})

        service = SarvamSttService(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        service._keys = KeyRing("SARVAM_API_KEY", ["first", "second"])

        with pytest.raises(TranscriptionError):
            await service.transcribe(b"RIFF" + b"\x00" * 100, include_native=False)
        assert service._keys.index == 1  # both were tried
        await service.aclose()

    @pytest.mark.asyncio
    async def test_server_error_does_not_burn_the_backup(self) -> None:
        """A 500 is not a credential problem; the spare key must stay unused."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "boom"})

        service = SarvamSttService(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        service._keys = KeyRing("SARVAM_API_KEY", ["first", "second"])

        with pytest.raises(TranscriptionError):
            await service.transcribe(b"RIFF" + b"\x00" * 100, include_native=False)
        assert service._keys.index == 0
        await service.aclose()
