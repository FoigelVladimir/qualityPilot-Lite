"""Official access check and safe, category-only archive import."""

import hashlib
import tarfile
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen

OFFICIAL_PAGE = "https://www.mvtec.com/research-teaching/datasets/mvtec-ad"


def check_access() -> str:
    request = Request(OFFICIAL_PAGE, headers={"User-Agent": "QualityPilot-Lite/0.1"})
    with urlopen(request, timeout=30) as response:
        page = response.read(4_000_000).decode("utf-8", errors="replace")
    if "Fill in the form" in page:
        return (
            "Manual access required: fill in the official form at "
            f"{OFFICIAL_PAGE}. See docs/data.md"
        )
    return f"Visit {OFFICIAL_PAGE} for current access instructions. No archive URL is assumed."


def import_archive(archive: Path, destination: Path, category: str) -> str:
    """Import a trusted vendor tar archive, rejecting links, traversal and overwrites.

    Validate every member before writing. A full archive is allowed as a local input,
    but only the requested category is extracted. Downloading is intentionally not
    attempted by guessing old URLs behind the current access form.
    """
    if category in {"", ".", ".."} or "/" in category or "\\" in category:
        raise ValueError("Invalid category")
    target = destination / category
    if target.exists():
        raise ValueError(f"Refusing to overwrite existing category: {target}")
    with tarfile.open(archive, "r:*") as bundle:
        members = []
        total = 0
        seen = set()
        for member in bundle.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or "\\" in member.name:
                raise ValueError(f"Unsafe archive path: {member.name}")
            if not (member.isdir() or member.isfile()):
                raise ValueError(f"Archive links and special files are forbidden: {member.name}")
            if not path.parts or path.parts[0] != category:
                continue
            if ":" in member.name:
                raise ValueError(f"Unsafe Windows archive path: {member.name}")
            resolved = destination.joinpath(*path.parts).resolve()
            if not resolved.is_relative_to(destination.resolve()):
                raise ValueError("Archive escapes destination")
            if resolved in seen:
                raise ValueError(f"Duplicate archive member: {member.name}")
            seen.add(resolved)
            total += member.size
            if total > 5_000_000_000:
                raise ValueError("Category exceeds the 5 GB extraction limit")
            members.append(member)
        if not any(m.isfile() and f"{category}/train/good/" in m.name for m in members):
            raise ValueError(f"Archive must contain {category}/train/good/")
        destination.mkdir(parents=True, exist_ok=True)
        bundle.extractall(destination, members=members, filter="data")
    with archive.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
