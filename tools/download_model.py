#!/usr/bin/env python3
"""Download the original nagisa 0.3.0 data without installing Python packages."""
import argparse
import hashlib
import io
from pathlib import Path
import tarfile
import urllib.request

URL = "https://files.pythonhosted.org/packages/7c/ae/4a05cf192d1656fae4dbd71367ccd86ba4a75a175131e489a4527ecab125/nagisa-0.3.0.tar.gz"
SHA256 = "7f62f478f40facc2e8b87835a4d8cbe78759dc0598f7f79ded50b0b117b9651a"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=Path("models/nagisa-0.3.0"))
    args = parser.parse_args()
    with urllib.request.urlopen(URL, timeout=120) as response:
        archive = response.read()
    if hashlib.sha256(archive).hexdigest() != SHA256:
        raise SystemExit("nagisa source archive checksum mismatch")
    members = {
        "nagisa/data/nagisa_v001.dict": "nagisa_v001.dict",
        "nagisa/data/nagisa_v001.model": "nagisa_v001.model",
        "LICENSE.txt": "LICENSE-nagisa.txt",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as source:
        for member, filename in members.items():
            entry = source.getmember("nagisa-0.3.0/" + member)
            if not entry.isfile():
                raise SystemExit("expected a regular archive member: " + member)
            # Extract only these named files, never arbitrary paths or symlinks.
            with source.extractfile(entry) as contents:
                data = contents.read()
            path = args.output / filename
            path.write_bytes(data)
            print(f"{hashlib.sha256(data).hexdigest()}  {path}")


if __name__ == "__main__":
    main()
