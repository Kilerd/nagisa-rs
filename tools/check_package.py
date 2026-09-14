#!/usr/bin/env python3
"""Check crate upload sizes and run a consumer built offline from the archives."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
LIMIT = 10 * 1024 * 1024


def main():
    metadata = json.loads(subprocess.check_output(
        ["cargo", "metadata", "--no-deps", "--offline", "--format-version=1"], cwd=ROOT))
    target = Path(metadata["target_directory"])
    packages = {package["name"]: package["version"] for package in metadata["packages"]}
    env = {key: value for key, value in os.environ.items() if not key.startswith("NAGISA_RS_")}
    with tempfile.TemporaryDirectory(prefix="nagisa-package-") as directory:
        work = Path(directory).resolve()
        extracted = {}
        for name in ("nagisa-rs", "nagisa-rs-model"):
            version = packages[name]
            archive = target / "package" / f"{name}-{version}.crate"
            size = archive.stat().st_size
            if size >= LIMIT:
                raise SystemExit(f"{archive.name} exceeds the default 10 MiB upload limit")
            print(f"{archive.name}: {size:,} bytes ({size / 1024**2:.2f} MiB)", flush=True)
            with tarfile.open(archive) as contents:
                for member in contents:
                    destination = (work / member.name).resolve()
                    if work not in destination.parents or not (member.isfile() or member.isdir()):
                        raise SystemExit(f"unexpected archive member: {member.name}")
                    if member.isdir():
                        destination.mkdir(parents=True, exist_ok=True)
                    else:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with contents.extractfile(member) as source, destination.open("wb") as output:
                            shutil.copyfileobj(source, output)
            extracted[name] = work / f"{name}-{version}"
        assert (extracted["nagisa-rs"] / "data/nagisa_v001.dict").is_file()
        assert (extracted["nagisa-rs"] / "licenses/nagisa-MIT.txt").is_file()
        assert (extracted["nagisa-rs-model"] / "data/nagisa_v001.bin.gz").is_file()
        assert (extracted["nagisa-rs-model"] / "LICENSE").is_file()

        consumer = work / "consumer"
        (consumer / "src").mkdir(parents=True)
        (consumer / "Cargo.toml").write_text(
            '[package]\nname = "nagisa-package-smoke"\nversion = "0.0.0"\nedition = "2024"\n'
            '[dependencies]\nnagisa-rs = { path = ' + json.dumps(str(extracted["nagisa-rs"])) + ' }\n'
            '[patch.crates-io]\nnagisa-rs-model = { path = ' + json.dumps(str(extracted["nagisa-rs-model"])) + ' }\n',
            encoding="utf-8")
        (consumer / "src/main.rs").write_text('''fn main() -> Result<(), nagisa_rs::JaError> {
    let segmenter = nagisa_rs::JaSegmenter::new()?;
    let tagger = nagisa_rs::Tagger::new()?;
    let text = "Pythonで簡単に使えるツールです";
    let expected = ["Python", "で", "簡単", "に", "使える", "ツール", "です"];
    assert_eq!(segmenter.words(text), expected);
    let tagged = tagger.tagging(text);
    assert_eq!(tagged.words, expected);
    assert_eq!(tagged.postags, ["名詞", "助詞", "形状詞", "助動詞", "動詞", "名詞", "助動詞"]);
    assert_eq!(tagger.postagging(&tagged.words)?, tagged.postags);
    let tagger = tagger.with_single_word_list(["東京大学"]);
    assert_eq!(tagger.words("東京大学"), ["東京大学"]);
    assert_eq!(tagger.words_with_options("Python", nagisa_rs::TextOptions { lower: true }), ["python"]);
    println!("Bundled segmentation and POS work without model files or package sources");
    Ok(())
}
''', encoding="utf-8")
        build = target / "package-smoke"
        subprocess.run(["cargo", "build", "--offline", "--manifest-path", str(consumer / "Cargo.toml"),
                        "--target-dir", str(build)], cwd=consumer, env=env, check=True)
        empty = work / "empty"
        empty.mkdir()
        binary = "nagisa-package-smoke" + (".exe" if os.name == "nt" else "")
        executable = empty / binary
        shutil.copy2(build / "debug" / binary, executable)
        # A CARGO_MANIFEST_DIR runtime lookup would now fail, as would any
        # attempt to open files relative to the consumer's working directory.
        for path in [consumer, *extracted.values()]:
            shutil.rmtree(path)
        subprocess.run([str(executable)], cwd=empty, env=env, check=True)


if __name__ == "__main__":
    main()
