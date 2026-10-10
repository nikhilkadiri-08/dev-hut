"""Persistence for the laboratory's generator profile.

The admin interface saves the generator setup once; every subsequent test request
reuses it. A JSON file is deliberate: the profile is a single small record that an
engineer should be able to read, diff and correct by hand, and a database would add
operational weight a two-day prototype cannot justify.

Until an admin saves anything, the profile is the set of values taken from the supplied
problem statement and workbook, reported as ``is_factory_default``.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from backend.schemas import GeneratorProfile, StoredProfile

PROFILE_PATH = Path(__file__).resolve().parent / "data" / "generator_profile.json"


def factory_default() -> StoredProfile:
    """The supplied-material values, used whenever nothing has been saved."""
    return StoredProfile(profile=GeneratorProfile(), is_factory_default=True)


def load_profile(path: Path | None = None) -> StoredProfile:
    """Read the saved profile, falling back to the supplied-material defaults.

    A corrupt or schema-invalid file falls back rather than raising, so a bad edit
    cannot take the whole application down mid-demo. The caller can tell the difference
    via ``is_factory_default``.
    """
    target = path or PROFILE_PATH
    if not target.exists():
        return factory_default()
    try:
        return StoredProfile.model_validate_json(target.read_text())
    except (ValueError, OSError):
        return factory_default()


def save_profile(
    profile: GeneratorProfile, updated_by: str = "", path: Path | None = None
) -> StoredProfile:
    """Persist a profile atomically.

    Written to a temporary file in the same directory and then renamed, so a crash
    partway through cannot leave a half-written profile behind.
    """
    target = path or PROFILE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    stored = StoredProfile(
        profile=profile,
        updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        updated_by=updated_by.strip() or "unidentified",
        is_factory_default=False,
    )

    handle, temporary = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w") as stream:
            stream.write(stored.model_dump_json(indent=2) + "\n")
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise

    return stored


def reset_profile(path: Path | None = None) -> StoredProfile:
    """Discard the saved profile and return to the supplied-material values."""
    target = path or PROFILE_PATH
    target.unlink(missing_ok=True)
    return factory_default()
