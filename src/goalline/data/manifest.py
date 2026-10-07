"""Pinned source file: where it comes from and how to check it is the same file."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path


class ChecksumError(RuntimeError):
    pass


@dataclass(frozen=True)
class Manifest:
    url: str
    commit: str
    sha256: str
    retrieved: str


def read_manifest(path: Path) -> Manifest:
    data = json.loads(Path(path).read_text())
    return Manifest(data["url"], data["commit"], data["sha256"], data["retrieved"])


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, manifest: Manifest) -> None:
    actual = sha256_of(path)
    if actual != manifest.sha256:
        raise ChecksumError(
            f"SHA-256 of {path} is {actual}, the manifest pins {manifest.sha256}. "
            "Delete the file and run `goalline download` again."
        )


def download(manifest: Manifest, dest: Path) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, suffix=".part")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        urllib.request.urlretrieve(manifest.url, tmp)
        verify(tmp, manifest)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest
