"""Ordered API keys with rotation on auth failure or rate limiting.

A free tier's binding limit is usually the key, not the vendor: quota is
per-key, so a second key on the same provider recovers from a 429 that a
provider-level failover cannot. This sits underneath that failover -- keys
rotate first, and only when every key is exhausted does the caller fall back to
a different vendor.

Rotation triggers on 401, 403 and 429 only. A 500 means the upstream is
unwell, and presenting a different key will not change its mind.

Environment lookup is deliberately case-insensitive for the backup suffix.
Windows treats environment variable names case-insensitively while Linux does
not, so a key written as SARVAM_API_KEY_Backup works locally and silently
vanishes on a Linux host -- exactly the kind of bug that only appears in
production.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# Statuses where a different key plausibly succeeds.
ROTATABLE_STATUSES: frozenset[int] = frozenset({401, 403, 429})

# Suffixes accepted for additional keys, in the order they are tried.
BACKUP_SUFFIXES: tuple[str, ...] = ("_BACKUP", "_FALLBACK", "_2")


def _lookup(name: str) -> str:
    """Case-insensitive environment lookup.

    os.getenv is case-sensitive on Linux, so an exact match is tried first and
    a scan only runs when that fails.
    """
    direct = os.getenv(name)
    if direct:
        return direct.strip()
    wanted = name.casefold()
    for key, value in os.environ.items():
        if key.casefold() == wanted and value.strip():
            return value.strip()
    return ""


def collect_keys(base_name: str) -> list[str]:
    """Primary key followed by any backups, de-duplicated, blanks dropped."""
    candidates = [_lookup(base_name)]
    candidates.extend(_lookup(f"{base_name}{suffix}") for suffix in BACKUP_SUFFIXES)

    keys: list[str] = []
    for candidate in candidates:
        # A repeated key would burn a rotation attempt on a credential that has
        # already failed.
        if candidate and candidate not in keys:
            keys.append(candidate)
    return keys


@dataclass
class KeyRing:
    """Holds one provider's keys and tracks which is in use."""

    name: str
    keys: list[str] = field(default_factory=list)
    index: int = 0

    @classmethod
    def from_env(cls, base_name: str) -> "KeyRing":
        return cls(name=base_name, keys=collect_keys(base_name))

    @classmethod
    def of(cls, name: str, key: str | None) -> "KeyRing":
        """Single explicit key, for tests and direct construction."""
        return cls(name=name, keys=[key] if key else [])

    @property
    def configured(self) -> bool:
        return bool(self.keys)

    @property
    def active(self) -> str:
        return self.keys[self.index] if self.keys else ""

    @property
    def count(self) -> int:
        return len(self.keys)

    @property
    def label(self) -> str:
        """Which key is in use, safe to log. Never the key itself."""
        return f"{self.name}[{self.index + 1}/{self.count}]" if self.keys else f"{self.name}[none]"

    def rotate(self) -> bool:
        """Advance to the next key. False when none remain untried."""
        if self.index + 1 >= len(self.keys):
            return False
        self.index += 1
        return True

    def reset(self) -> None:
        """Return to the primary, e.g. once a rate-limit window has passed."""
        self.index = 0
