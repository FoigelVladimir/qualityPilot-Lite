"""Explicit public mirror fallback; never silently replace the official source."""

import hashlib
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from urllib.parse import quote
from urllib.request import Request, urlopen

REPOSITORY = "foersben/mvtec-ad"
REVISION = "c75b39616f84db43677bcc8228caaafaf5096d7f"
LOGGER = logging.getLogger(__name__)


def checked_path(destination: Path, category: str, relative: str) -> Path:
    path = PurePosixPath(relative)
    if (
        not path.parts
        or path.parts[0] != category
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in relative
        or ":" in relative
    ):
        raise ValueError(f"Unsafe mirror path: {relative}")
    target = destination.joinpath(*path.parts).resolve()
    if not target.is_relative_to((destination / category).resolve()):
        raise ValueError("Mirror path escapes selected category")
    return target


def verify_payload(payload: bytes, entry: dict) -> str:
    if len(payload) != entry["size"]:
        raise ValueError(f"Size mismatch: {entry['path']}")
    digest = hashlib.sha256(payload).hexdigest()
    if entry.get("lfs"):
        expected = entry["lfs"]["oid"]
        actual = digest
    else:
        expected = entry["oid"]
        header = f"blob {len(payload)}\0".encode()
        actual = hashlib.sha1(header + payload, usedforsecurity=False).hexdigest()
    if actual != expected:
        raise ValueError(f"Mirror integrity mismatch: {entry['path']}")
    return digest


def download_mirror(destination: Path, category: str = "bottle") -> Path:
    """Download raw files at a pinned revision and verify mirror LFS/Git hashes.

    These hashes verify transfer integrity against the mirror, NOT equivalence
    to the unavailable official MVTec archive. No repository code is executed.
    Completed files can be reused after an interrupted download, but must match.
    """
    if not re.fullmatch(r"[a-z][a-z0-9_]*", category):
        raise ValueError("Invalid category")
    url = (
        f"https://huggingface.co/api/datasets/{REPOSITORY}/tree/{REVISION}/"
        f"{quote(category)}?recursive=true&limit=1000"
    )
    with urlopen(Request(url, headers={"User-Agent": "QualityPilot-Lite/0.1"}), timeout=30) as r:
        tree = json.load(r)
        if r.headers.get("Link"):
            raise ValueError("Paginated category listing is not supported; refusing partial data")
    entries = [entry for entry in tree if entry["type"] == "file"]
    if not entries or sum(e["size"] for e in entries) > 1_000_000_000:
        raise ValueError("Mirror category is empty or exceeds the 1 GB limit")
    for entry in entries:
        checked_path(destination, category, entry["path"])
        if Path(entry["path"]).suffix.lower() not in {".png", ".txt"}:
            raise ValueError(f"Unexpected mirror file: {entry['path']}")
    LOGGER.info(
        "Explicit unofficial mirror %s @ %s: %s files, %.1f MB",
        REPOSITORY,
        REVISION,
        len(entries),
        sum(e["size"] for e in entries) / 1e6,
    )

    def fetch(entry: dict) -> dict:
        target = checked_path(destination, category, entry["path"])
        if target.exists():
            payload = target.read_bytes()
        else:
            source = (
                f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/"
                f"{quote(entry['path'], safe='/')}"
            )
            with urlopen(source, timeout=60) as response:
                payload = response.read(entry["size"] + 1)
        digest = verify_payload(payload, entry)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".part")
            temporary.write_bytes(payload)
            temporary.replace(target)
        return {"path": entry["path"], "size": len(payload), "sha256": digest}

    with ThreadPoolExecutor(max_workers=4) as pool:
        files = []
        for index, item in enumerate(pool.map(fetch, entries), 1):
            files.append(item)
            if index % 50 == 0 or index == len(entries):
                LOGGER.info("Verified %s/%s files", index, len(entries))
    provenance = {
        "source_kind": "third_party_mirror",
        "repository": REPOSITORY,
        "revision": REVISION,
        "source_url": f"https://huggingface.co/datasets/{REPOSITORY}/tree/{REVISION}/{category}",
        "original_source": "https://www.mvtec.com/research-teaching/datasets/mvtec-ad",
        "license": "CC-BY-NC-SA-4.0",
        "downloaded_at": datetime.now(UTC).isoformat(),
        "integrity": "LFS SHA256 / Git blob hashes verified against pinned mirror revision",
        "limitation": "Not independently matched against an official MVTec archive checksum.",
        "files": files,
    }
    path = destination / f"provenance-{category}.json"
    path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return path
