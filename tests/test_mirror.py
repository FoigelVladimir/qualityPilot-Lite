import hashlib
from pathlib import Path

import pytest

from qualitypilot.data.mirror import checked_path, verify_payload


@pytest.mark.parametrize(
    "relative", ["../a", "/bottle/a", "bottle/../../a", "other/a", "bottle/C:a", "bottle\\a"]
)
def test_mirror_paths(tmp_path: Path, relative: str):
    with pytest.raises(ValueError):
        checked_path(tmp_path, "bottle", relative)


def test_payload_sha256():
    payload = b"image fixture"
    entry = {
        "path": "bottle/test/good/000.png",
        "size": len(payload),
        "lfs": {"oid": hashlib.sha256(payload).hexdigest()},
    }
    assert verify_payload(payload, entry) == entry["lfs"]["oid"]
    with pytest.raises(ValueError, match="integrity"):
        verify_payload(b"IMAGE fixture", entry)
    with pytest.raises(ValueError, match="Size"):
        verify_payload(b"", entry)


def test_payload_git_blob():
    payload = b"license text"
    blob = f"blob {len(payload)}\0".encode() + payload
    entry = {
        "path": "bottle/license.txt",
        "size": len(payload),
        "oid": hashlib.sha1(blob, usedforsecurity=False).hexdigest(),
    }
    assert len(verify_payload(payload, entry)) == 64
