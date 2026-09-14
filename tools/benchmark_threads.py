#!/usr/bin/env python3
"""Shared-model throughput: Rust, CPython 3.12, and 3.14t with/without the GIL."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import sysconfig
import threading
import time
import unicodedata
import warnings

from benchmark import ROOT, THREAD_ENV, command, cpu_accounting, cpu_name, cpu_resources, positive

FIXTURE = ROOT / "tests/fixtures/ja_parity_subset.jsonl"
ENGINES = ("rust", "python312", "python314t_gil", "python314t_nogil")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gil_enabled():
    return sys._is_gil_enabled() if hasattr(sys, "_is_gil_enabled") else True


def python_worker():
    request = json.load(sys.stdin)
    before_import = gil_enabled()
    import nagisa
    import nagisa_utils
    expected_gil = request["engine"] != "python314t_nogil"
    free_threaded = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
    version = (3, 12) if request["engine"] == "python312" else (3, 14)
    if (sys.implementation.name != "cpython" or sys.version_info[:2] != version
            or free_threaded != (version == (3, 14))):
        raise RuntimeError("wrong interpreter build for this engine")
    after_import = gil_enabled()
    if before_import != expected_gil or after_import != expected_gil:
        raise RuntimeError("GIL state does not match the requested benchmark")
    if importlib.metadata.version("nagisa") != "0.3.0":
        raise RuntimeError("expected nagisa 0.3.0")
    backend = type(nagisa.tagger._model).__module__
    if backend != "nagisa.np_model":
        raise RuntimeError("expected nagisa's NumPy/Cython inference backend")
    # One initialized model shared by every thread. Each call gets a fresh lazy Token.
    tagging = request["mode"] == "tagging"
    tagger = nagisa.tagger

    def infer(text):
        result = tagger.tagging(text)
        return result.words, result.postags if tagging else []

    expected = [(case["words"], case["postags"] if tagging else []) for case in request["cases"]]
    for case, reference in zip(request["cases"], expected, strict=True):
        if infer(case["text"]) != reference:
            raise RuntimeError(f"serial reference parity failed: {case['text']!r}")
    measurements = []
    for count in request["threads"]:
        ready, start, done, validate = barriers = [threading.Barrier(count + 1) for _ in range(4)]

        def worker(worker_index):
            try:
                indices = request["indices"][worker_index * len(request["indices"]) // count:
                                             (worker_index + 1) * len(request["indices"]) // count]
                outputs = [None] * len(indices)
                for i in range(request["warmup"]):
                    infer(request["cases"][indices[i % len(indices)]]["text"])
                ready.wait()
                start.wait()
                for i, index in enumerate(indices):
                    outputs[i] = infer(request["cases"][index]["text"])
                done.wait()
                validate.wait()
                for index, output in zip(indices, outputs, strict=True):
                    if output != expected[index]:
                        raise RuntimeError(f"concurrent reference parity failed at case {index}")
                return len(outputs), threading.get_ident()
            except BaseException:
                for barrier in barriers:
                    barrier.abort()
                raise

        with ThreadPoolExecutor(max_workers=count) as pool:
            futures = [pool.submit(worker, index) for index in range(count)]
            try:
                ready.wait()
                timer = time.perf_counter_ns()
                start.wait()
                done.wait()
                elapsed_s = (time.perf_counter_ns() - timer) / 1e9
                validate.wait()
            except BaseException:
                for barrier in barriers:
                    barrier.abort()
                # Surface the inference error instead of only BrokenBarrierError.
                for future in futures:
                    future.result()
                raise
            results = [future.result() for future in futures]
        if len({identity for _, identity in results}) != count:
            raise RuntimeError("expected distinct concurrent worker threads")
        if gil_enabled() != expected_gil:
            raise RuntimeError("GIL state changed during inference")
        measurements.append({"threads": count, "elapsed_s": elapsed_s,
                             "verified_lines": sum(lines for lines, _ in results),
                             "worker_lines": [lines for lines, _ in results]})
    model_dir = Path(nagisa.__file__).parent / "data"
    distribution = importlib.metadata.distribution("nagisa")
    json.dump({"measurements": measurements, "environment": {
        "python": sys.version, "unicode": unicodedata.unidata_version,
        "free_threaded_build": free_threaded,
        "gil_before_import": before_import, "gil_after_import": after_import,
        "gil_after_inference": gil_enabled(), "gil_override": sys._xoptions.get("gil"),
        "backend": backend,
        "dynet_imported": any(name.startswith(("dynet", "_dynet")) for name in sys.modules),
        "packages": {name: importlib.metadata.version(name) for name in ("nagisa", "numpy", "six")},
        "nagisa_wheel": distribution.read_text("WHEEL"),
        "extension_sha256": sha256(Path(nagisa_utils.__file__)),
        "model_sha256": {path.name: sha256(path) for path in sorted(model_dir.glob("nagisa_v001.*"))},
    }}, sys.stdout, ensure_ascii=False)


def default_gil_probe():
    before = gil_enabled()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import nagisa  # noqa: F401
    json.dump({"free_threaded_build": bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
               "gil_before_import": before, "gil_after_import": gil_enabled(),
               "gil_warning": any("global interpreter lock" in str(item.message) for item in caught)},
              sys.stdout)


def execute(argv, env, request=None):
    completed = subprocess.run(argv, input=json.dumps(request) if request is not None else None,
                               capture_output=True, text=True, cwd=ROOT, env=env, timeout=600)
    if completed.returncode:
        raise RuntimeError(f"worker failed ({completed.returncode}):\n{completed.stderr}")
    return json.loads(completed.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python312", type=Path, default=Path(sys.executable))
    parser.add_argument("--python314t", type=Path, required=True)
    parser.add_argument("--mode", choices=("words", "tagging", "both"), default="both")
    parser.add_argument("--threads", type=positive, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--lines", type=positive, default=4000, help="total lines per batch, across all threads")
    parser.add_argument("--runs", type=positive, default=4)
    parser.add_argument("--warmup", type=positive, default=10, help="untimed calls per worker")
    parser.add_argument("--cpus", help="comma-separated Linux CPU affinity pool, e.g. 16,17,18,19,20,21,22,23")
    parser.add_argument("--output", type=Path, default=ROOT / "results/threads.json")
    args = parser.parse_args()
    if 1 not in args.threads or len(set(args.threads)) != len(args.threads) or max(args.threads) > args.lines:
        parser.error("--threads must be unique, include 1, and not exceed --lines")
    env = dict(os.environ, **THREAD_ENV)
    # Only the explicit per-engine -X flag may select GIL state.
    env.pop("PYTHON_GIL", None)
    subprocess.run(["cargo", "build", "--release", "--locked", "--example", "bench_threads"],
                   cwd=ROOT, env=env, check=True)
    metadata = json.loads(command("cargo", "metadata", "--no-deps", "--format-version=1"))
    binary = Path(metadata["target_directory"]) / "release/examples/bench_threads"
    if os.name == "nt":
        binary = binary.with_suffix(".exe")
    if args.cpus:
        cpus = {int(value) for value in args.cpus.split(",")}
        if not hasattr(os, "sched_getaffinity") or not cpus <= os.sched_getaffinity(0):
            parser.error("--cpus must be allowed Linux logical CPUs")
        os.sched_setaffinity(0, cpus)
    script = str(Path(__file__).resolve())
    argv = {"rust": [str(binary)],
            "python312": [str(args.python312.absolute()), script, "--python-worker"],
            "python314t_gil": [str(args.python314t.absolute()), "-X", "gil=1", script, "--python-worker"],
            "python314t_nogil": [str(args.python314t.absolute()), "-X", "gil=0", script, "--python-worker"]}
    probe = execute([str(args.python314t.absolute()), script, "--default-gil-probe"], env)
    if not probe["free_threaded_build"]:
        parser.error("--python314t must be a free-threaded build")
    # Reuse all Wikipedia sentence/multi-sentence cases with their existing attribution.
    cases = [case for line in FIXTURE.read_text(encoding="utf-8").split("\n")
             if line and (case := json.loads(line)).get("cat") in ("wiki", "wiki_multi")]
    indices = [i % len(cases) for i in range(args.lines)]
    random.Random(0).shuffle(indices)
    request = {"cases": cases, "indices": indices, "warmup": args.warmup}
    modes = ("words", "tagging") if args.mode == "both" else (args.mode,)
    runs, environments = [], {}
    accounting_before = cpu_accounting()
    for mode in modes:
        for run in range(args.runs):
            offset = run % len(ENGINES)
            order = ENGINES[offset:] + ENGINES[:offset]
            offset = run % len(args.threads)
            threads = args.threads[offset:] + args.threads[:offset]
            row = {"mode": mode, "run": run + 1, "order": order, "thread_order": threads, "engines": {}}
            for engine in order:
                result = execute(argv[engine], env, dict(request, mode=mode, threads=threads, engine=engine))
                if "environment" in result:
                    current = result.pop("environment")
                    previous = environments.setdefault(engine, current)
                    if previous != current:
                        raise RuntimeError("environment changed between runs")
                measurements = result["measurements"]
                if [cell["threads"] for cell in measurements] != threads:
                    raise RuntimeError("worker returned incorrect thread counts")
                for cell in measurements:
                    n = cell["threads"]
                    counts = [(i + 1) * args.lines // n - i * args.lines // n for i in range(n)]
                    if cell["verified_lines"] != args.lines or cell["worker_lines"] != counts or cell["elapsed_s"] <= 0:
                        raise RuntimeError("worker did not verify the complete batch")
                row["engines"][engine] = measurements
                print(f"{mode} run {run + 1}/{args.runs}: {engine} ({threads})", file=sys.stderr)
            runs.append(row)
    accounting_after = cpu_accounting()
    model_hashes = [value["model_sha256"] for value in environments.values()]
    if not all(hashes == model_hashes[0] for hashes in model_hashes):
        raise RuntimeError("Python interpreters loaded different models")
    if model_hashes[0]["nagisa_v001.dict"] != sha256(ROOT / "data/nagisa_v001.dict"):
        raise RuntimeError("Python and Rust dictionaries differ")
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    for name in ("nagisa_v001.dict", "nagisa_v001.model"):
        if model_hashes[0][name] != manifest["source_sha256"][name]:
            raise RuntimeError("Python model differs from the bundled model's pinned source")
    for name, asset in manifest["assets"].items():
        if sha256(ROOT / name) != asset["sha256"]:
            raise RuntimeError("bundled model does not match its manifest")
    summary = []
    for mode in modes:
        for engine in ENGINES:
            baseline = None
            for count in sorted(args.threads):
                seconds = [cell["elapsed_s"] for run in runs if run["mode"] == mode
                           for cell in run["engines"][engine] if cell["threads"] == count]
                rates = [args.lines / elapsed for elapsed in seconds]
                median = statistics.median(rates)
                if count == 1:
                    baseline = median
                summary.append({"mode": mode, "engine": engine, "threads": count,
                                "median_lines_s": median, "min_lines_s": min(rates), "max_lines_s": max(rates),
                                "speedup_vs_one_thread": median / baseline, "run_lines_s": rates})
    files = [ROOT / name for name in ("Cargo.toml", "Cargo.lock", "build.rs", "model/Cargo.toml")]
    for folder, pattern in (("src", "*.rs"), ("model/src", "*.rs"), ("examples", "*.rs"), ("tools", "*.py")):
        files.extend(sorted((ROOT / folder).glob(pattern)))
    resources = cpu_resources()
    if sys.platform == "darwin":
        resources["performance_cores"] = int(command("sysctl", "-n", "hw.perflevel0.physicalcpu"))
        resources["efficiency_cores"] = int(command("sysctl", "-n", "hw.perflevel1.physicalcpu"))
    elif args.cpus:
        resources["cpu_topology"] = {
            str(cpu): {name: Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/{name}").read_text().strip()
                       for name in ("physical_package_id", "core_id", "thread_siblings_list")}
            for cpu in sorted(os.sched_getaffinity(0))}
    report = {"schema_version": 1, "benchmark": "shared_model_threads",
              "measured_at": datetime.now(timezone.utc).isoformat(),
              "environment": {"platform": platform.platform(), "cpu": cpu_name(), "cpu_resources": resources,
                              "rustc": command("rustc", "--version"), "rustflags": os.environ.get("RUSTFLAGS", ""),
                              "thread_env": THREAD_ENV, "python": environments, "default_314t_import": probe,
                              "cpu_accounting_before": accounting_before, "cpu_accounting_after": accounting_after},
              "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in files},
              "bundled_sha256": {name: sha256(ROOT / name) for name in
                                 ("data/nagisa_v001.dict", "model/data/nagisa_v001.bin.gz")},
              "fixture_sha256": sha256(FIXTURE), "request": request,
              "summary": summary, "runs": runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("| Task | Engine | Threads | Lines/s | Scaling vs 1 thread |")
    print("|---|---|---:|---:|---:|")
    for row in summary:
        print(f"| {row['mode']} | {row['engine']} | {row['threads']} | {row['median_lines_s']:.1f} | "
              f"{row['speedup_vs_one_thread']:.2f}× |")
    print(f"Raw measurements: {args.output}")


if __name__ == "__main__":
    if sys.argv[1:] == ["--python-worker"]:
        python_worker()
    elif sys.argv[1:] == ["--default-gil-probe"]:
        default_gil_probe()
    else:
        main()
