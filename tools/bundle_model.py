#!/usr/bin/env python3
"""Rebuild/check bundled data from the checksum-pinned nagisa 0.2.11 files."""
import argparse
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re
import struct

ROOT = Path(__file__).resolve().parents[1]
MAGIC = b"NAGISA\x00\x01"
SOURCES = {
    "nagisa_v001.dict": "968ac9e6c7a53051ef24d8561673dd31de81b2feb9b5bff01b1d3b6b2473113c",
    "nagisa_v001.model": "9db9abc06a927c56e18af8d485e20a14908138752be459a83f0dc7ac85368c1b",
    "LICENSE-nagisa.txt": "fa2fb3121bd2250c66169e769cdf32836394c5f60306b814462c06881a06f0cd",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def pack_model(text):
    source = io.BytesIO(text)
    packed = bytearray(MAGIC)
    count = 0
    while header := source.readline():
        match = re.fullmatch(rb"#(?:Lookup)?Parameter# (\S+) \{([0-9,]+)\} ([0-9]+) ZERO_GRAD\n", header)
        if not match:
            raise SystemExit("unexpected DyNet parameter header")
        dims = [int(dim) for dim in match[2].split(b",")]
        values = [float(value) for value in source.read(int(match[3])).split()]
        if len(values) != math.prod(dims) or not all(map(math.isfinite, values)):
            raise SystemExit("invalid model values")
        packed.extend(header)
        packed.extend(struct.pack("<" + "f" * len(values), *values))
        count += 1
    if count != 28:
        raise SystemExit(f"expected 28 model parameters, found {count}")
    return bytes(packed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?", default=ROOT / "models/nagisa-0.2.11")
    parser.add_argument("--check", action="store_true", help="verify committed assets without writing")
    args = parser.parse_args()
    source = {name: (args.source / name).read_bytes() for name in SOURCES}
    for name, expected in SOURCES.items():
        if digest(source[name]) != expected:
            raise SystemExit(f"upstream checksum mismatch: {name}")
    weights = pack_model(source["nagisa_v001.model"])
    dictionary = ROOT / "data/nagisa_v001.dict"
    model = ROOT / "model/data/nagisa_v001.bin.gz"
    license_path = ROOT / "model/LICENSE"
    manifest_path = ROOT / "data/manifest.json"
    if args.check:
        if dictionary.read_bytes() != source["nagisa_v001.dict"]:
            raise SystemExit("bundled dictionary differs from upstream")
        if gzip.decompress(model.read_bytes()) != weights:
            raise SystemExit("bundled model differs from upstream f32 weights")
        if license_path.read_bytes() != source["LICENSE-nagisa.txt"]:
            raise SystemExit("model license differs from upstream")
    else:
        dictionary.parent.mkdir(parents=True, exist_ok=True)
        model.parent.mkdir(parents=True, exist_ok=True)
        dictionary.write_bytes(source["nagisa_v001.dict"])
        compressed = io.BytesIO()
        with gzip.GzipFile(fileobj=compressed, mode="wb", filename="", mtime=0, compresslevel=9) as output:
            output.write(weights)
        model.write_bytes(compressed.getvalue())
        license_path.write_bytes(source["LICENSE-nagisa.txt"])
    manifest = {
        "upstream": "https://github.com/taishi-i/nagisa/tree/0.2.11",
        "version": "0.2.11", "license": "MIT", "source_sha256": SOURCES,
        "format": "NAGISA-v1: original DyNet header lines followed by little-endian f32 values",
        "parameters": 28, "uncompressed_weights_sha256": digest(weights),
        "assets": {str(path.relative_to(ROOT)): {"sha256": digest(path.read_bytes()), "bytes": path.stat().st_size}
                   for path in (dictionary, model)},
    }
    if args.check:
        if json.loads(manifest_path.read_text()) != manifest:
            raise SystemExit("bundled data manifest differs")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Verified original dictionary, all 28 lossless f32 parameters, license and asset hashes")


if __name__ == "__main__":
    main()
