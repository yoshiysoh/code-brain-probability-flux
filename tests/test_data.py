"""The data files in data/ are the published ones.

data/SHA256SUMS lists every file the notebooks read with its SHA-256 hash
(`shasum -a 256 -c SHA256SUMS` inside data/ checks the same).
"""

import hashlib

import pytest

from brain_probability_flux.plotting import DATA_DIR

MANIFEST = DATA_DIR / "SHA256SUMS"


def manifest():
    return [
        tuple(line.split())
        for line in MANIFEST.read_text().splitlines()
        if line.strip()
    ]


def test_manifest_lists_every_data_file():
    listed = {name for _, name in manifest()}
    shipped = {p.name for p in DATA_DIR.glob("*.h5")}
    assert listed == shipped


@pytest.mark.parametrize("digest, name", manifest(), ids=lambda v: v[:16])
def test_data_file_matches_manifest(digest, name):
    h = hashlib.sha256()
    with open(DATA_DIR / name, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    assert h.hexdigest() == digest, name
