#!/usr/bin/env python3
"""Compare word segmentation or full POS tagging in separate Rust/Python processes."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import unicodedata
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
THREAD_ENV = {name: "1" for name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
)}


def python_worker():
    request = json.load(sys.stdin)
    start = time.perf_counter_ns()
    import nagisa
    load_ms = (time.perf_counter_ns() - start) / 1e6
    backend = type(nagisa.tagger._model).__module__
    if backend != "nagisa.np_model":
        raise SystemExit("expected nagisa 0.3.0's NumPy/Cython inference backend")
    tagging = request["mode"] == "tagging"
    measurements = []
    for text in request["texts"]:
        for _ in range(request["warmup"]):
            tagged = nagisa.tagging(text)
            tagged.words
            if tagging:
                tagged.postags
            del tagged
        samples = []
        for _ in range(request["iterations"]):
            start = time.perf_counter_ns()
            tagged = nagisa.tagging(text)
            words = tagged.words
            postags = tagged.postags if tagging else []
            elapsed = time.perf_counter_ns() - start
            samples.append(elapsed / 1e3)
            del tagged, words, postags
        tagged = nagisa.tagging(text)
        measurements.append({"text": text, "words": tagged.words,
                             "postags": tagged.postags if tagging else [],
                             "samples_us": samples})
    json.dump({"load_ms": load_ms, "backend": backend,
               "dynet_imported": any(name.startswith(("dynet", "_dynet")) for name in sys.modules),
               "measurements": measurements}, sys.stdout, ensure_ascii=False)


def command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def cpu_name():
    if sys.platform == "darwin":
        return command("sysctl", "-n", "machdep.cpu.brand_string")
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            key, sep, value = line.partition(":")
            if sep and key.strip() == "model name":
                return value.strip()
    return platform.processor()


def cpu_resources():
    """Record CPU constraints without hostnames, addresses or cluster metadata."""
    result = {"logical_cpus": os.cpu_count()}
    if hasattr(os, "sched_getaffinity"):
        result["affinity"] = sorted(os.sched_getaffinity(0))
    cgroup = Path("/sys/fs/cgroup")
    for name in ("cpu.max", "cpuset.cpus.effective", "memory.max"):
        path = cgroup / name
        if path.exists():
            result[name] = path.read_text().strip()
    return result


def cpu_accounting():
    path = Path("/sys/fs/cgroup/cpu.stat")
    if path.exists():
        return {key: int(value) for key, value in
                (line.split() for line in path.read_text().splitlines())}
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/benchmark.json")
    parser.add_argument("--runs", type=positive, default=6)
    parser.add_argument("--warmup", type=positive, default=10)
    parser.add_argument("--iterations", type=positive, default=400)
    parser.add_argument("--mode", choices=("words", "tagging"), default="words")
    parser.add_argument("--cpu", type=int, help="pin workers to this Linux logical CPU")
    parser.add_argument("--external-model", action="store_true",
                        help="load the Python package's original model instead of bundled data")
    args = parser.parse_args()
    if (sys.implementation.name, sys.version_info[:2], unicodedata.unidata_version) != (
        "cpython", (3, 12), "15.0.0"
    ):
        raise SystemExit("use CPython 3.12 / Unicode 15.0.0")
    if importlib.metadata.version("nagisa") != "0.3.0":
        raise SystemExit("install tools/requirements-reference.txt first")
    # Locate package data without importing/initializing nagisa in the driver.
    model_dir = Path(importlib.metadata.distribution("nagisa").locate_file("nagisa/data"))
    env = dict(os.environ, **THREAD_ENV)
    subprocess.run(["cargo", "build", "--release", "--locked", "--example", "bench"],
                   cwd=ROOT, env=env, check=True)
    metadata = json.loads(command("cargo", "metadata", "--no-deps", "--format-version=1"))
    binary = Path(metadata["target_directory"]) / "release/examples/bench"
    if os.name == "nt":
        binary = binary.with_suffix(".exe")
    if args.cpu is not None:
        if not hasattr(os, "sched_getaffinity") or args.cpu not in os.sched_getaffinity(0):
            raise SystemExit("--cpu must be an allowed Linux logical CPU")
        # Set affinity after building; worker processes inherit it, including model allocation.
        os.sched_setaffinity(0, {args.cpu})
    base = "令和6年4月1日から、東京都渋谷区で新しいサービスが始まります。"
    request = {"mode": args.mode, "warmup": args.warmup, "iterations": args.iterations,
               "texts": [(base * (length // len(base) + 1))[:length] for length in (20, 100, 400)]}
    runs = []
    accounting_before = cpu_accounting()
    for run in range(args.runs):
        order = ["rust", "python"] if run % 2 == 0 else ["python", "rust"]
        result = {"order": order}
        for engine in order:
            argv = [str(binary)] + ([str(model_dir)] if args.external_model else []) if engine == "rust" else [
                sys.executable, str(Path(__file__).resolve()), "--python-worker"]
            completed = subprocess.run(argv, input=json.dumps(request), capture_output=True,
                                       text=True, env=env, cwd=ROOT, check=True)
            result[engine] = json.loads(completed.stdout)
        for rust, python in zip(result["rust"]["measurements"], result["python"]["measurements"], strict=True):
            if any(rust[key] != python[key] for key in ("text", "words", "postags")):
                raise SystemExit("word/POS parity failed; benchmark is not comparable")
        runs.append(result)
        print(f"completed run {run + 1}/{args.runs}: {' → '.join(order)}", file=sys.stderr)
    accounting_after = cpu_accounting()
    summary = []
    for index, text in enumerate(request["texts"]):
        row = {"chars": len(text)}
        for engine in ("rust", "python"):
            samples = sorted(s for run in runs for s in run[engine]["measurements"][index]["samples_us"])
            medians = [statistics.median(run[engine]["measurements"][index]["samples_us"]) for run in runs]
            row[engine] = {"median_us": statistics.median(samples),
                           "p95_us": samples[math.ceil(len(samples) * 0.95) - 1],
                           "run_medians_us": medians}
        row["speedup"] = row["python"]["median_us"] / row["rust"]["median_us"]
        summary.append(row)
    files = [ROOT / name for name in ("Cargo.toml", "Cargo.lock", "build.rs", "model/Cargo.toml")]
    for folder, pattern in (("src", "*.rs"), ("model/src", "*.rs"), ("examples", "*.rs"), ("tools", "*.py")):
        files.extend(sorted((ROOT / folder).glob(pattern)))
    report = {"schema_version": 2, "measured_at": datetime.now(timezone.utc).isoformat(),
              "rust_model": "external" if args.external_model else "bundled",
              "environment": {"platform": platform.platform(), "cpu": cpu_name(),
                              "cpu_resources": cpu_resources(),
                              "cpu_accounting_before": accounting_before,
                              "cpu_accounting_after": accounting_after,
                              "python": sys.version, "unicode": unicodedata.unidata_version,
                              "rustc": command("rustc", "--version"),
                              "packages": {name: importlib.metadata.version(name) for name in
                                           ("nagisa", "DyNet38", "numpy", "Cython", "six")},
                              "thread_env": THREAD_ENV, "rustflags": os.environ.get("RUSTFLAGS", "")},
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
              "bundled_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                 for name in ("data/nagisa_v001.dict", "model/data/nagisa_v001.bin.gz")},
              "model_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(model_dir.glob("nagisa_v001.*"))},
              "request": request, "summary": summary, "runs": runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("| Characters | Python median / p95 (µs) | Rust median / p95 (µs) | Speedup |")
    print("|---:|---:|---:|---:|")
    for row in summary:
        py, rs = row["python"], row["rust"]
        print(f"| {row['chars']} | {py['median_us']:.1f} / {py['p95_us']:.1f} | "
              f"{rs['median_us']:.1f} / {rs['p95_us']:.1f} | {row['speedup']:.2f}× |")
    print(f"Raw samples: {args.output}")


if __name__ == "__main__":
    if sys.argv[1:] == ["--python-worker"]:
        python_worker()
    else:
        main()
