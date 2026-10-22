import io
import shutil
import tarfile
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from qualitypilot.config import Config, load_config
from qualitypilot.data.download import import_archive
from qualitypilot.data.pipeline import inventory, prepare, read_rgb


def test_config_roundtrip(config: Config, tmp_path: Path):
    config.save(tmp_path / "config.yaml")
    assert load_config(tmp_path / "config.yaml") == config


def test_relative_paths_follow_config(config: Config, tmp_path: Path):
    path = tmp_path / "nested" / "config.yaml"
    path.parent.mkdir()
    values = config.to_dict()
    values["data_root"] = "../data"
    import yaml

    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    assert load_config(path).data_root == tmp_path / "data"


def test_reproducible_disjoint_splits(config: Config):
    first, summary = prepare(config)
    second, repeated = prepare(config)
    assert first == second
    assert summary == repeated
    groups = {
        key: {s.path for s in first if s.split == key} for key in ("train", "calibration", "test")
    }
    assert all(groups.values())
    assert not groups["train"] & groups["calibration"]
    assert not groups["train"] & groups["test"]
    assert not groups["calibration"] & groups["test"]
    assert all(s.label == 0 for s in first if s.split != "test")
    assert groups["test"] == {s.path for s in inventory(config) if s.split == "test"}
    changed, _ = prepare(replace(config, seed=7))
    assert {s.path for s in changed if s.split == "calibration"} != groups["calibration"]


def test_missing_structure(config: Config):
    with pytest.raises(ValueError, match="Missing MVTec"):
        prepare(replace(config, category="missing"))


def test_corrupt_image(config: Config):
    path = config.data_root / "bottle/train/good/000.png"
    path.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="Cannot decode"):
        prepare(config)


def test_duplicate_pixels_across_splits(config: Config):
    source = config.data_root / "bottle/train/good/000.png"
    duplicate = config.data_root / "bottle/test/good/duplicate.bmp"
    read_rgb(source).save(duplicate)
    with pytest.raises(ValueError, match="Duplicate image content"):
        prepare(config)


@pytest.mark.parametrize("failure", ["missing", "shape", "empty", "nonbinary", "orphan"])
def test_invalid_masks(config: Config, failure: str):
    mask = config.data_root / "bottle/ground_truth/scratch/000_mask.png"
    if failure == "missing":
        mask.unlink()
    elif failure == "shape":
        Image.new("L", (4, 4), 255).save(mask)
    elif failure == "empty":
        Image.new("L", (24, 24), 0).save(mask)
    elif failure == "nonbinary":
        Image.new("L", (24, 24), 42).save(mask)
    else:
        shutil.copyfile(mask, mask.with_name("orphan_mask.png"))
    with pytest.raises(ValueError):
        prepare(config)


def test_safe_archive_import(config: Config, tmp_path: Path):
    archive = tmp_path / "bottle.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(config.data_root / "bottle", arcname="bottle")
    destination = tmp_path / "imported"
    checksum = import_archive(archive, destination, "bottle")
    assert len(checksum) == 64
    assert len(inventory(replace(config, data_root=destination))) == 23
    with pytest.raises(ValueError, match="overwrite"):
        import_archive(archive, destination, "bottle")


@pytest.mark.parametrize(
    "name", ["../escape.txt", "/escape.txt", "bottle/../../escape.txt", "bottle/C:escape.txt"]
)
def test_archive_traversal_rejected(tmp_path: Path, name: str):
    archive = tmp_path / "unsafe.tar"
    with tarfile.open(archive, "w") as bundle:
        member = tarfile.TarInfo(name)
        member.size = 1
        bundle.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(ValueError):
        import_archive(archive, tmp_path / "out", "bottle")
    assert not (tmp_path / "out").exists()


def test_archive_symlink_rejected(tmp_path: Path):
    archive = tmp_path / "unsafe.tar"
    with tarfile.open(archive, "w") as bundle:
        member = tarfile.TarInfo("bottle/link")
        member.type = tarfile.SYMTYPE
        member.linkname = "../../outside"
        bundle.addfile(member)
    with pytest.raises(ValueError, match="links"):
        import_archive(archive, tmp_path / "out", "bottle")
