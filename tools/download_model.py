#!/usr/bin/env python3
"""Download the original nagisa 0.2.11 data without installing Python packages."""
import argparse
import hashlib
import io
from pathlib import Path
import tarfile
import urllib.request

URL = "https://files.pythonhosted.org/packages/e1/71/1cc63f4fde1c1ca491ea69b46c3c5bd6bb06f3efa8262f34b227fe25dbf5/nagisa-0.2.11.tar.gz"
SHA256 = "213e8f837c6f7391ea1f56f4fa6047612d117e3f0e9c4f6b93a3a77f78ba6e6f"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=Path("models/nagisa-0.2.11"))
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
            entry = source.getmember("nagisa-0.2.11/" + member)
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
