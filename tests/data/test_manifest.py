import hashlib
import json

import pytest

from goalline.data.manifest import (
    ChecksumError,
    Manifest,
    download,
    read_manifest,
    sha256_of,
    verify,
)


def write(tmp_path, text="a,b\n1,2\n"):
    path = tmp_path / "source.csv"
    path.write_text(text)
    return path


def test_sha256_of_matches_hashlib(tmp_path):
    path = write(tmp_path)
    assert sha256_of(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_read_manifest(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"url": "u", "commit": "c", "sha256": "s", "retrieved": "r"}))
    assert read_manifest(path) == Manifest(url="u", commit="c", sha256="s", retrieved="r")


def test_verify_accepts_matching_file(tmp_path):
    path = write(tmp_path)
    verify(path, Manifest("u", "c", sha256_of(path), "r"))


def test_verify_rejects_changed_file(tmp_path):
    path = write(tmp_path)
    with pytest.raises(ChecksumError, match="SHA-256"):
        verify(path, Manifest("u", "c", "0" * 64, "r"))


def test_download_fetches_and_verifies(tmp_path):
    source = write(tmp_path)
    dest = tmp_path / "raw" / "Matches.csv"
    manifest = Manifest(source.as_uri(), "c", sha256_of(source), "r")
    assert download(manifest, dest) == dest
    assert dest.read_text() == source.read_text()


def test_download_with_wrong_checksum_leaves_no_file(tmp_path):
    source = write(tmp_path)
    dest = tmp_path / "raw" / "Matches.csv"
    with pytest.raises(ChecksumError):
        download(Manifest(source.as_uri(), "c", "0" * 64, "r"), dest)
    assert not dest.exists()
    assert list(dest.parent.iterdir()) == []


def test_repository_manifest_is_pinned():
    m = read_manifest(__import__("pathlib").Path("data/manifest.json"))
    assert m.commit == "25882a58a736daf7ece3781940eac17ae1117a66"
    assert m.commit in m.url
    assert m.sha256 == "ef224cf2c252f07a842b3bcfd4ba5c718c25cedd8937ffa174a74b86b5ba4221"
