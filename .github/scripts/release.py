"""Validate and publish the two Cargo packages, only from a version-tag workflow."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tomllib
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ("ragisa-model", "ragisa")
SEMVER = re.compile(r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)"
                    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?")


def versions(root=ROOT, tag=""):
    main = tomllib.loads((root / "Cargo.toml").read_text())
    model = tomllib.loads((root / "model/Cargo.toml").read_text())
    if main["package"]["name"] != "ragisa" or model["package"]["name"] != "ragisa-model":
        raise ValueError("unexpected package names")
    result = {"ragisa-model": model["package"]["version"], "ragisa": main["package"]["version"]}
    if any(not SEMVER.fullmatch(version) for version in result.values()):
        raise ValueError("invalid package version")
    expected_tag = "v" + result["ragisa"]
    if tag and tag != expected_tag:
        raise ValueError(f"release tag must equal {expected_tag}, got {tag!r}")
    dependency = main["dependencies"]["ragisa-model"]
    if dependency["version"] != "=" + result["ragisa-model"] or dependency["path"] != "model":
        raise ValueError("ragisa must pin the exact workspace ragisa-model version")
    return result


def registry_status(name, version):
    request = urllib.request.Request(
        f"https://crates.io/api/v1/crates/{name}",
        headers={"User-Agent": "ragisa-release (https://github.com/Kilerd/ragisa)"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return {"crate_exists": False, "published": False}
        raise
    if data["crate"]["name"] != name:
        raise ValueError("registry returned an unexpected crate")
    matching = [item for item in data["versions"] if item["num"] == version]
    if any(item["yanked"] for item in matching):
        raise ValueError(f"{name} {version} is yanked; choose a new release version")
    return {"crate_exists": True, "published": bool(matching)}


def plan(package_versions):
    return {name: dict(version=package_versions[name], **registry_status(name, package_versions[name]))
            for name in PACKAGES}


def publish(package_versions, tag):
    if (os.environ.get("GITHUB_ACTIONS") != "true"
            or os.environ.get("GITHUB_EVENT_NAME") != "push"
            or os.environ.get("GITHUB_REF") != "refs/tags/" + tag
            or tag != "v" + package_versions["ragisa"]):
        raise ValueError("publishing is only allowed in the matching version-tag push workflow")
    if not os.environ.get("CARGO_REGISTRY_TOKEN"):
        raise ValueError("crates.io authentication is required")
    for name, state in plan(package_versions).items():
        if state["published"]:
            print(f"Already published: {name} {state['version']}", flush=True)
            continue
        # Cargo waits for index propagation before returning, so the parent can resolve the model.
        subprocess.run(["cargo", "publish", "--package", name, "--locked", "--registry", "crates-io"],
                       cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "plan", "publish"))
    parser.add_argument("--tag", default="", help="must equal v<ragisa version>; optional for dry-run checks")
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    package_versions = versions(tag=args.tag)
    if args.command == "publish":
        publish(package_versions, args.tag)
        return
    if args.command == "check":
        print(json.dumps({"tag": "v" + package_versions["ragisa"], "versions": package_versions}, indent=2))
        return
    state = plan(package_versions)
    print(json.dumps(state, indent=2))
    outputs = {"needs_publish": any(not item["published"] for item in state.values()),
               "needs_token": any(not item["crate_exists"] for item in state.values())}
    if args.github_output:
        with args.github_output.open("a") as output:
            for key, value in outputs.items():
                output.write(f"{key}={str(value).lower()}\n")


if __name__ == "__main__":
    main()
