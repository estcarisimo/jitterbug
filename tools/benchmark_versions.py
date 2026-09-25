#!/usr/bin/env python3
"""Compare the runtime of Jitterbug releases on the PAM 2022 dataset.

Every release runs in its own virtual environment, built with ``uv`` on the same
interpreter, against ``examples/network_analysis/data/raw.csv``. Two dependency sets:

``era``
    Dependencies resolved as of the release date (``uv pip install --exclude-newer``),
    with the Bayesian change point back end the release used at the time. This is what
    a user of that release got.
``current``
    Every release on the same, current dependency set (``bayesian-changepoint`` 1.2),
    which isolates changes in Jitterbug's own code from changes in its dependencies.

Two measurements per release and configuration:

``cli``
    Wall-clock time of the command-line tool in a fresh process: interpreter start-up,
    imports, analysis and writing the results file.
``stage``
    In-process time of each analysis stage (import, load, change points, latency
    jumps, jitter analysis, inference), from a probe script that wraps the stage
    functions of each release.

Usage::

    uv run python tools/benchmark_versions.py setup --workdir /tmp/jb-bench
    uv run python tools/benchmark_versions.py run --workdir /tmp/jb-bench --repeats 5
    uv run python tools/benchmark_versions.py report --workdir /tmp/jb-bench

The script only needs the standard library; ``uv`` and ``git`` must be on ``PATH``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("benchmark_versions")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET = REPO_ROOT / "examples" / "network_analysis" / "data" / "raw.csv"
DEFAULT_PYTHON = "3.12"

# The numpy implementation that the 2022 paper and Jitterbug 1.0 used (installed from git,
# as the 1.0 README instructed) and the first PyTorch rewrite that Jitterbug 2.0 used.
BCP_NUMPY = "bayescd @ git+https://github.com/hildensia/bayesian_changepoint_detection@9ecc0ec"
BCP_TORCH_1_0 = (
    "bayesian_changepoint_detection @ "
    "git+https://github.com/estcarisimo/bayesian_changepoint_detection@fdb332b"
)
BCP_CURRENT = "bayesian-changepoint>=1.2"


@dataclass(frozen=True)
class Release:
    """A Jitterbug release and how to install it."""

    tag: str
    cutoff: str  # --exclude-newer for the "era" dependency set (UTC, right after the tag)
    source: str  # "pypi" (jitterbug-inference) or "git" (tag checkout)
    api: str  # "v1" (argparse script) or "v2" (Typer CLI and JitterbugAnalyzer)
    era_requirements: tuple[str, ...]

    @property
    def version(self) -> str:
        return self.tag.removeprefix("v")


RELEASES = (
    Release("v1.0.0", "2024-03-19T00:00:00Z", "git", "v1", ("numpy", "pandas", "scipy", BCP_NUMPY)),
    Release(
        "v2.0.0", "2025-07-20T00:00:00Z", "git", "v2", ("-r", "requirements-new.txt", BCP_TORCH_1_0)
    ),
    Release("v2.1.0", "2026-09-24T01:10:00Z", "git", "v2", ()),
    Release("v2.1.1", "2026-09-24T01:39:00Z", "pypi", "v2", ()),
    Release("v2.2.0", "2026-09-24T16:25:00Z", "pypi", "v2", ()),
    Release("v2.3.0", "2026-09-24T17:07:00Z", "pypi", "v2", ()),
)

# name -> (change point algorithm, jitter method). v1.0.0 only implements BCP.
CONFIGS = {
    "bcp_ks": ("bcp", "ks_test"),
    "ruptures_ks": ("ruptures", "ks_test"),
    "bcp_jd": ("bcp", "jitter_dispersion"),
    "ruptures_jd": ("ruptures", "jitter_dispersion"),
}
V1_METHODS = {"ks_test": "ks", "jitter_dispersion": "jd"}

KEY_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "torch",
    "ruptures",
    "pydantic",
    "bayescd",
    "bayesian-changepoint",
    "bayesian-changepoint-detection",
)

PROBE_V2 = r"""
import json, sys, time
t0 = time.perf_counter()
from jitterbug.analyzer import JitterbugAnalyzer
from jitterbug.models.config import JitterbugConfig
stages = {"import": time.perf_counter() - t0}
algorithm, method, path = sys.argv[1], sys.argv[2], sys.argv[3]
config = JitterbugConfig()
config.change_point_detection.algorithm = algorithm
config.jitter_analysis.method = method
analyzer = JitterbugAnalyzer(config)

def timed(obj, name, stage):
    func = getattr(obj, name)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            stages[stage] = stages.get(stage, 0.0) + time.perf_counter() - start
    setattr(obj, name, wrapper)

timed(analyzer.data_loader, "load_from_file", "load")
timed(analyzer.change_point_detector, "detect", "change_points")
timed(analyzer.latency_jump_analyzer, "analyze", "latency_jumps")
timed(analyzer.jitter_analyzer, "analyze_ks_test", "jitter")
timed(analyzer.jitter_analyzer, "analyze_jitter_dispersion", "jitter")
timed(analyzer.congestion_inference_analyzer, "infer", "inference")
start = time.perf_counter()
results = analyzer.analyze_from_file(path)
total = time.perf_counter() - start
stages["other"] = total - sum(v for k, v in stages.items() if k != "import")
inferences = results.inferences
print(json.dumps({
    "stages": stages,
    "periods": len(inferences),
    "congested": sum(bool(i.is_congested) for i in inferences),
}))
"""

PROBE_V1 = r"""
import json, sys, time
t0 = time.perf_counter()
import tools.jitterbug as cli
stages = {"import": time.perf_counter() - t0}
method = {"ks_test": "ks", "jitter_dispersion": "jd"}[sys.argv[2]]
path = sys.argv[3]

def timed(module, name, stage):
    func = getattr(module, name)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            stages[stage] = stages.get(stage, 0.0) + time.perf_counter() - start
    setattr(module, name, wrapper)

def timed_fit(cls, stage):
    fit = cls.fit
    def wrapper(self, *args, **kwargs):
        start = time.perf_counter()
        try:
            return fit(self, *args, **kwargs)
        finally:
            stages[stage] = stages.get(stage, 0.0) + time.perf_counter() - start
    cls.fit = wrapper

timed(cli, "load_and_process_rtt_data", "load")
timed_fit(cli.BCP, "change_points")
timed_fit(cli.LatencyJumps, "latency_jumps")
timed(cli, "compute_ks_test", "jitter")
timed(cli, "compute_jitter_dispersion", "jitter")
timed_fit(cli.CongestionInference, "inference")
start = time.perf_counter()
rtts, mins = cli.load_and_process_rtt_data(path)
out = cli.jitterbug_analysis(
    rtts["epoch"].values, rtts["values"].values, mins["epoch"].values, mins["values"].values,
    method, "bcp", cli.DEFAULT_LATENCY_JUMP_THRESHOLD, cli.DEFAULT_JITTER_DISPERSION_THRESHOLD,
    cli.DEFAULT_MOVING_AVERAGE_ORDER, cli.DEFAULT_MOVING_IQR_ORDER, cli.DEFAULT_CPD_THRESHOLD,
)
total = time.perf_counter() - start
stages["other"] = total - sum(v for k, v in stages.items() if k != "import")
print(json.dumps({
    "stages": stages,
    "periods": len(out),
    "congested": int(out["congestion"].astype(bool).sum()),
}))
"""

# Prepended to the CLI and probe scripts with --hide-mps. BCP 1.0 picked Apple's MPS
# device by default when PyTorch reported it (Jitterbug 2.0 did not pass a device); this
# makes PyTorch report no MPS, so the same code runs on the CPU.
HIDE_MPS = """
try:
    import torch
    torch.backends.mps.is_available = lambda: False
except ImportError:
    pass
"""

CLI_V2 = """
import sys
from jitterbug.cli import main
sys.argv = ["jitterbug", *sys.argv[1:]]
main()
"""

CLI_V1 = """
import runpy, sys
sys.argv = ["jitterbug", *sys.argv[1:]]
runpy.run_module("tools.jitterbug", run_name="__main__")
"""

RESULT_FIELDS = (
    "release",
    "deps",
    "config",
    "kind",
    "stage",
    "repeat",
    "seconds",
    "max_rss_mb",
    "periods",
    "congested",
)


def run_command(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    """Run a command, fail loudly, and return its standard output."""
    logger.debug("$ %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{proc.stderr[-4000:]}")
    return proc.stdout


def env_dir(workdir: Path, release: Release, deps: str) -> Path:
    return workdir / "env" / f"{deps}-{release.tag}"


def src_dir(workdir: Path, release: Release) -> Path:
    return workdir / "src" / release.tag


def env_python(venv: Path) -> Path:
    return venv / "bin" / "python"


def checkout(workdir: Path, release: Release) -> Path:
    """Extract the release tag into ``workdir/src/<tag>`` (git releases only)."""
    target = src_dir(workdir, release)
    if not target.exists():
        target.mkdir(parents=True)
        archive = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "archive", release.tag],
            capture_output=True,
            check=True,
        )
        subprocess.run(["tar", "-x", "-C", str(target)], input=archive.stdout, check=True)
    return target


def install(workdir: Path, release: Release, deps: str, python: str) -> None:
    """Create the virtual environment for one release and dependency set."""
    venv = env_dir(workdir, release, deps)
    if venv.exists():
        logger.info("%s (%s): environment exists, skipping", release.tag, deps)
        return
    logger.info("%s (%s): building environment", release.tag, deps)
    run_command(["uv", "venv", "--quiet", "--python", python, str(venv)])
    pip = ["uv", "pip", "install", "--quiet", "--python", str(env_python(venv))]
    if deps == "era":
        pip += ["--exclude-newer", release.cutoff]

    src = checkout(workdir, release) if release.source == "git" else None
    if release.api == "v1":
        requirements = (
            list(release.era_requirements)
            if deps == "era"
            else [
                "numpy",
                "pandas",
                "scipy",
                BCP_CURRENT,
            ]
        )
        run_command(pip + requirements)
    elif release.tag == "v2.0.0":
        assert src is not None
        bcp = BCP_TORCH_1_0 if deps == "era" else BCP_CURRENT
        run_command(pip + ["-r", str(src / "requirements-new.txt"), bcp], cwd=src)
        run_command(pip + ["--no-deps", str(src)])
    elif src is not None:
        run_command(pip + [f"{src}[bcp]"])
    else:
        run_command(pip + [f"jitterbug-inference[bcp]=={release.version}"])


def package_versions(venv: Path) -> dict[str, str]:
    """Return the installed versions of the packages that drive the runtime."""
    out = run_command(["uv", "pip", "list", "--format", "json", "--python", str(env_python(venv))])
    installed = {pkg["name"].lower(): pkg["version"] for pkg in json.loads(out)}
    return {name: installed[name] for name in KEY_PACKAGES if name in installed}


def cli_command(
    workdir: Path, release: Release, venv: Path, config: str, output: Path, hide_mps: bool
) -> tuple[list[str], Path | None]:
    algorithm, method = CONFIGS[config]
    if release.api == "v1":
        args = ["-r", str(DATASET), "-i", V1_METHODS[method], "-c", algorithm, "-o", str(output)]
        if hide_mps:
            cmd = [str(env_python(venv)), "-c", HIDE_MPS + CLI_V1]
        else:
            cmd = [str(env_python(venv)), "-m", "tools.jitterbug"]
        return cmd + args, src_dir(workdir, release)
    args = ["analyze", str(DATASET), "-a", algorithm, "-m", method, "-o", str(output)]
    if hide_mps:
        return [str(env_python(venv)), "-c", HIDE_MPS + CLI_V2, *args], None
    return [str(venv / "bin" / "jitterbug"), *args], None


def count_results(release: Release, output: Path) -> tuple[int, int]:
    """Return (periods, congested periods) from a CLI results file."""
    if release.api == "v1":
        with output.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        return len(rows), sum(row["congestion"] in ("True", "1", "1.0") for row in rows)
    inferences = json.loads(output.read_text())["inferences"]
    return len(inferences), sum(bool(inf["is_congested"]) for inf in inferences)


def run_timed(cmd: list[str], cwd: Path | None, timeout: float | None) -> tuple[str, float, float]:
    """Run a command and return (stdout, wall-clock seconds, peak RSS in MiB) of that child.

    Raises ``TimeoutError`` (after killing the child) when it runs longer than ``timeout``.
    """
    logger.debug("$ %s", " ".join(cmd))
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        start = time.perf_counter()
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=out, stderr=err)
        while True:
            pid, status, usage = os.wait4(proc.pid, os.WNOHANG)
            seconds = time.perf_counter() - start
            if pid:
                break
            if timeout is not None and seconds > timeout:
                proc.kill()
                os.wait4(proc.pid, 0)
                raise TimeoutError(f"{' '.join(cmd)} ran longer than {timeout:.0f} s")
            time.sleep(0.01)
        proc.returncode = os.waitstatus_to_exitcode(status)
        out.seek(0)
        err.seek(0)
        if proc.returncode != 0:
            stderr = err.read().decode(errors="replace")
            raise RuntimeError(f"{' '.join(cmd)} failed:\n{stderr[-4000:]}")
        stdout = out.read().decode()
    # ru_maxrss is in bytes on macOS and in KiB on Linux.
    scale = 1 / 2**20 if sys.platform == "darwin" else 1 / 2**10
    return stdout, seconds, usage.ru_maxrss * scale


def time_cli(
    workdir: Path,
    release: Release,
    venv: Path,
    config: str,
    timeout: float | None,
    hide_mps: bool = False,
) -> dict[str, float | int]:
    """Time one CLI run in a fresh process and count the periods it reported."""
    suffix = ".csv" if release.api == "v1" else ".json"
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / f"results{suffix}"
        cmd, cwd = cli_command(workdir, release, venv, config, output, hide_mps)
        _, seconds, rss = run_timed(cmd, cwd, timeout)
        periods, congested = count_results(release, output)
    return {"seconds": seconds, "max_rss_mb": rss, "periods": periods, "congested": congested}


def time_stages(
    workdir: Path,
    release: Release,
    venv: Path,
    config: str,
    timeout: float | None,
    hide_mps: bool = False,
) -> dict[str, Any]:
    algorithm, method = CONFIGS[config]
    probe = PROBE_V1 if release.api == "v1" else PROBE_V2
    if hide_mps:
        probe = HIDE_MPS + probe
    cwd = src_dir(workdir, release) if release.api == "v1" else None
    cmd = [str(env_python(venv)), "-c", probe, algorithm, method, str(DATASET)]
    stdout, _, _ = run_timed(cmd, cwd, timeout)
    probe_result: dict[str, Any] = json.loads(stdout.strip().splitlines()[-1])
    return probe_result


def selected(names: list[str] | None) -> list[Release]:
    if not names:
        return list(RELEASES)
    by_tag = {r.tag: r for r in RELEASES}
    return [by_tag[n if n.startswith("v") else f"v{n}"] for n in names]


def cmd_setup(args: argparse.Namespace) -> None:
    for deps in args.deps:
        for release in selected(args.releases):
            install(args.workdir, release, deps, args.python)


def cmd_run(args: argparse.Namespace) -> None:
    results_path = args.workdir / "results.csv"
    new_file = not results_path.exists()
    packages: dict[str, dict[str, str]] = {}
    environment: dict[str, object] = {
        "python": args.python,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": cpu_name(),
        "cpu_count": os.cpu_count(),
        "thread_env": {k: v for k, v in os.environ.items() if k.endswith("_NUM_THREADS")},
    }
    with results_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        if new_file:
            writer.writeheader()
        for deps in args.deps:
            for release in selected(args.releases):
                venv = env_dir(args.workdir, release, deps)
                if not venv.exists():
                    logger.warning("%s (%s): no environment, run setup first", release.tag, deps)
                    continue
                packages[f"{deps}-{release.tag}"] = package_versions(venv)
                # Warm up: compile bytecode and load the dataset into the page cache.
                warm = (
                    "import tools.jitterbug" if release.api == "v1" else "import jitterbug.cli.main"
                )
                cwd = src_dir(args.workdir, release) if release.api == "v1" else None
                run_command([str(env_python(venv)), "-c", warm], cwd=cwd)
                DATASET.read_bytes()
                for config in args.configs:
                    if release.api == "v1" and CONFIGS[config][0] != "bcp":
                        continue
                    for repeat in range(args.repeats):
                        label = f"{deps}-cpu" if args.hide_mps else deps
                        row = {"release": release.tag, "deps": label, "config": config}
                        try:
                            cli = time_cli(
                                args.workdir, release, venv, config, args.timeout, args.hide_mps
                            )
                        except TimeoutError:
                            # Recorded as a lower bound; the remaining repeats would time out too.
                            logger.warning(
                                "%s %s %s: timed out after %.0f s",
                                release.tag,
                                deps,
                                config,
                                args.timeout,
                            )
                            timed_out = {"seconds": args.timeout, "max_rss_mb": float("nan")}
                            writer.writerow(
                                row
                                | {"kind": "cli", "stage": "timeout", "repeat": repeat}
                                | timed_out
                                | {"periods": "", "congested": ""}
                            )
                            handle.flush()
                            break
                        writer.writerow(
                            row | {"kind": "cli", "stage": "total", "repeat": repeat} | cli
                        )
                        handle.flush()
                        logger.info(
                            "%s %s %s cli #%d: %.2f s (%d periods, %d congested)",
                            release.tag,
                            deps,
                            config,
                            repeat,
                            cli["seconds"],
                            cli["periods"],
                            cli["congested"],
                        )
                        if repeat < args.stage_repeats:
                            probe = time_stages(
                                args.workdir, release, venv, config, args.timeout, args.hide_mps
                            )
                            for stage, seconds in probe["stages"].items():
                                writer.writerow(
                                    row
                                    | {"kind": "stage", "stage": stage, "repeat": repeat}
                                    | {"seconds": seconds, "max_rss_mb": float("nan")}
                                    | {"periods": probe["periods"], "congested": probe["congested"]}
                                )
                            handle.flush()
    env_path = args.workdir / "environment.json"
    previous = json.loads(env_path.read_text()) if env_path.exists() else {}
    previous_packages = previous.get("packages", {})
    environment["packages"] = previous_packages | packages
    env_path.write_text(json.dumps(environment, indent=2) + "\n")
    logger.info("Results appended to %s", results_path)


def cpu_name() -> str:
    if sys.platform == "darwin":
        try:
            return run_command(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
        except RuntimeError:
            pass
    return platform.processor() or "unknown"


def cmd_report(args: argparse.Namespace) -> None:
    """Print Markdown tables: CLI time per release, and stage medians."""
    with (args.workdir / "results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    groups: dict[tuple[str, str, str, str, str], list[float]] = {}
    rss: dict[tuple[str, str, str], list[float]] = {}
    outcome: dict[tuple[str, str, str], tuple[str, str]] = {}
    for row in rows:
        key = (row["deps"], row["config"], row["kind"], row["stage"], row["release"])
        groups.setdefault(key, []).append(float(row["seconds"]))
        if row["kind"] == "cli" and row["stage"] == "total":
            run = (row["deps"], row["config"], row["release"])
            outcome[run] = (row["periods"], row["congested"])
            rss.setdefault(run, []).append(float(row["max_rss_mb"]))
    order = [r.tag for r in RELEASES]
    for deps in dict.fromkeys(row["deps"] for row in rows):
        for config in CONFIGS:
            cli = {k[4]: v for k, v in groups.items() if k[:4] == (deps, config, "cli", "total")}
            timeouts = {
                k[4]: max(v) for k, v in groups.items() if k[:4] == (deps, config, "cli", "timeout")
            }
            if not cli:
                continue
            releases = sorted(set(cli) | set(timeouts), key=order.index)
            reference = next(tag for tag in releases if tag in cli)
            baseline = statistics.median(cli[reference])
            print(f"\n### CLI, {config}, {deps} dependencies\n")
            print(
                "| Release | Median (s) | Min–max (s) | Runs | Peak RSS (MiB) "
                f"| Speedup vs {reference} | Periods / congested |"
            )
            print("|---|---:|---:|---:|---:|---:|---:|")
            for tag in releases:
                if tag not in cli:
                    limit = timeouts[tag]
                    print(
                        f"| {tag} | > {fmt(limit)} | – | 1 (timed out) | – "
                        f"| < {speedup(baseline / limit)} | – |"
                    )
                    continue
                times = cli[tag]
                med = statistics.median(times)
                periods, congested = outcome[(deps, config, tag)]
                print(
                    f"| {tag} | {fmt(med)} | {fmt(min(times))}–{fmt(max(times))} | {len(times)} "
                    f"| {statistics.median(rss[(deps, config, tag)]):,.0f} "
                    f"| {speedup(baseline / med)} | {periods} / {congested} |"
                )
            stages = sorted(
                {k[3] for k in groups if k[:3] == (deps, config, "stage")}, key=stage_order
            )
            if not stages:
                continue
            print(f"\n### Stages (median seconds), {config}, {deps} dependencies\n")
            print("| Release | " + " | ".join(stages) + " |")
            print("|---|" + "---:|" * len(stages))
            for tag in releases:
                cells = []
                for stage in stages:
                    values = groups.get((deps, config, "stage", stage, tag))
                    cells.append(fmt(statistics.median(values)) if values else "–")
                print(f"| {tag} | " + " | ".join(cells) + " |")


def speedup(ratio: float) -> str:
    if ratio < 0.01:
        return f"{ratio:,.3f}×"
    return f"{ratio:,.2f}×" if ratio < 10 else f"{ratio:,.0f}×"


def stage_order(stage: str) -> int:
    known = ["import", "load", "change_points", "latency_jumps", "jitter", "inference", "other"]
    return known.index(stage) if stage in known else len(known)


def fmt(seconds: float) -> str:
    return (
        f"{seconds:,.3f}"
        if seconds < 1
        else f"{seconds:,.2f}"
        if seconds < 100
        else f"{seconds:,.0f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--workdir", type=Path, required=True, help="environments and results")
    common.add_argument("--releases", nargs="*", help="tags to include (default: all)")
    common.add_argument("--deps", nargs="*", default=["era"], choices=["era", "current"])
    common.add_argument("-v", "--verbose", action="store_true")

    setup = sub.add_parser("setup", parents=[common], help="build one environment per release")
    setup.add_argument("--python", default=DEFAULT_PYTHON, help="interpreter for every environment")
    setup.set_defaults(func=cmd_setup)

    run = sub.add_parser("run", parents=[common], help="time the CLI and the analysis stages")
    run.add_argument("--configs", nargs="*", default=list(CONFIGS), choices=list(CONFIGS))
    run.add_argument("--repeats", type=int, default=5, help="CLI runs per release and config")
    run.add_argument("--stage-repeats", type=int, default=3, help="stage probe runs (<= repeats)")
    run.add_argument("--python", default=DEFAULT_PYTHON, help="recorded in environment.json")
    run.add_argument(
        "--hide-mps",
        action="store_true",
        help="make PyTorch report no MPS device (runs are labeled <deps>-cpu)",
    )
    run.add_argument(
        "--timeout", type=float, default=None, help="seconds per run; slower runs are a lower bound"
    )
    run.set_defaults(func=cmd_run)

    report = sub.add_parser("report", parents=[common], help="print Markdown tables")
    report.set_defaults(func=cmd_report)

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")
    if shutil.which("uv") is None:
        parser.error("uv is required")
    args.workdir = args.workdir.resolve()
    args.workdir.mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
