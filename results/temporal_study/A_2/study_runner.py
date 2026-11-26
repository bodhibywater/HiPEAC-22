#!/usr/bin/env python3
"""Run matched HiPEAC temporal experiments; also works on A with --suite a.
All variants use a single exact input-major trace. No simulated speedup is
reported for failed correctness checks. No third-party runner dependencies.
"""
import argparse
import configparser
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import time

BASE = ["sparse_scalar_im", "dense_rvv_fused", "sparse_rvv_fused",
        "sparse_rvv_fused_acc2", "sparse_rvv_fused_acc4"]
SPARSE = BASE[2:]
A = ["sparse_rvv_temporal4_weights", "sparse_rvv_temporal4"]

def tile(b, m, shared, resident):
    return f"sparse_rvv_tile{b}_m{m}_{'shared' if shared else 'separate'}_{'reg' if resident else 'mem'}"

FULL = [tile(b, m, s, r) for b, m, s, r in itertools.product((2, 4, 8), (1, 2), (False, True), (False, True))]
ONE = [tile(1, m, False, True) for m in (1, 2)]
CORE = [tile(b, m, True, r) for b, m, r in itertools.product((2, 4, 8), (1, 2), (False, True))]
CORE += [tile(4, 1, False, r) for r in (False, True)]
SUITES = {"a": BASE + A, "core": BASE + A + ONE + CORE, "full": BASE + A + ONE + FULL}
METRICS = {
    "cycles": "board.processor.cores.core.numCycles",
    "instructions": "simInsts", "sim_ops": "simOps", "ticks": "simTicks",
    "branch_mispredicts": "board.processor.cores.core.commit.branchMispredicts",
    "lsq_full_events": "board.processor.cores.core.iew.lsqFullEvents",
    "l1d_accesses": "board.cache_hierarchy.l1d-cache-0.overallAccesses::total",
    "l1d_misses": "board.cache_hierarchy.l1d-cache-0.overallMisses::total",
    "l1i_accesses": "board.cache_hierarchy.l1i-cache-0.overallAccesses::total",
    "l1i_misses": "board.cache_hierarchy.l1i-cache-0.overallMisses::total",
    "l2_accesses": "board.cache_hierarchy.l2-cache-0.overallAccesses::total",
    "l2_misses": "board.cache_hierarchy.l2-cache-0.overallMisses::total",
    "l1d_miss_latency_ticks": "board.cache_hierarchy.l1d-cache-0.overallMissLatency::total",
    "l2_miss_latency_ticks": "board.cache_hierarchy.l2-cache-0.overallMissLatency::total",
}
HEADER = struct.Struct("<8sIIIIIff")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def capture(cmd, cwd=None):
    return subprocess.check_output([str(x) for x in cmd], cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()


def logged(cmd, log, cwd=None):
    with Path(log).open("w") as f:
        return subprocess.run([str(x) for x in cmd], cwd=cwd, stdout=f, stderr=subprocess.STDOUT).returncode


def require_run(cmd, log, cwd=None):
    if logged(cmd, log, cwd):
        print("\n".join(Path(log).read_text(errors="replace").splitlines()[-25:]), file=sys.stderr)
        raise RuntimeError(f"Command failed; see {log}")


def validate_trace(path):
    with path.open("rb") as f:
        magic, n, steps, ni, nh, no, beta, threshold = HEADER.unpack(f.read(HEADER.size))
    if magic != b"SNNMNST2" or not (0 < n <= 10000 and 0 < steps <= 1024 and ni == 784 and 0 < nh <= 8192 and no == 10):
        raise ValueError("Expected a valid input-major SNNMNST2 trace")
    expected = HEADER.size + 4 * (ni*nh + nh + nh*no + no) + n * (2 + steps*(ni+nh+no))
    if path.stat().st_size != expected:
        raise ValueError("Trace length does not match its header")
    return dict(samples=n, timesteps=steps, inputs=ni, hidden=nh, outputs=no, beta=beta, threshold=threshold, sha256=sha(path))


def elf_check(path):
    with path.open("rb") as f:
        h = f.read(20)
    if len(h) < 20 or h[:6] != b"\x7fELF\x02\x01" or struct.unpack_from("<H", h, 18)[0] != 243:
        raise RuntimeError(f"Not a little-endian RV64 ELF: {path}")


def roi(path):
    # First COMPLETE block, never sum the exit dump or replace missing with zero.
    active = False
    found = {}
    for line in path.read_text(errors="replace").splitlines():
        if "Begin Simulation Statistics" in line:
            if active:
                raise ValueError("First ROI block is incomplete")
            active = True
            continue
        if not active:
            continue
        if "End Simulation Statistics" in line:
            return found
        fields = line.split()
        if len(fields) >= 2 and not fields[0].startswith("#"):
            found[fields[0]] = fields[1]
    raise ValueError("No complete first ROI statistics block")


def stat_int(stats, key):
    value = stats.get(key)
    return int(value) if value is not None and re.fullmatch(r"[0-9]+", value) else None


def metadata(mode):
    match = re.fullmatch(r"sparse_rvv_tile([1248])_m([12])_(shared|separate)_(reg|mem)", mode)
    if match:
        b, m, s, r = match.groups()
        return dict(family="tile", window=int(b), lmul=int(m), weight_sharing=int(s == "shared"), membrane_resident=int(r == "reg"))
    if mode in A:
        return dict(family="A", window=4, lmul=1, weight_sharing=1, membrane_resident=int(mode == A[1]))
    return dict(family="baseline", window=1, lmul=1 if mode != BASE[0] else "", weight_sharing=0, membrane_resident=0)


def config_info(path, expected_vlen):
    c = configparser.ConfigParser(interpolation=None, strict=False)
    c.read(path)
    core = "board.processor.cores.core"
    isa = core + ".isa"
    if c.get(core, "type", fallback="") not in ("RiscvO3CPU", "BaseO3CPU", "DerivO3CPU"):
        # The exact name varies by version; require O3 in the name.
        if "O3" not in c.get(core, "type", fallback=""):
            raise ValueError("Expected an O3 CPU in config.ini")
    if c.getint(isa, "vlen", fallback=-1) != expected_vlen:
        raise ValueError("Configured VLEN differs from requested VLEN")
    if c.has_option(isa, "enable_rvv") and not c.getboolean(isa, "enable_rvv"):
        raise ValueError("RVV is disabled")
    # Fingerprint every model parameter except the workload subtree, where
    # command-line mode and trace/output paths legitimately differ.
    model = {}
    for section in c.sections():
        if section == "root" or section.startswith("board"):
            if "workload" in section or "process" in section and "processor" not in section:
                continue
            model[section] = {k: v for k, v in c.items(section)
                              if k not in ("workload", "output", "errout", "input", "executable", "cmd", "cwd", "pid")}
    # A concise config for human review; archive the original config regardless.
    clocks = {section: c.get(section, "clock") for section in c.sections() if c.has_option(section, "clock")}
    sizes = {section: c.get(section, "size") for section in c.sections() if "cache" in section and c.has_option(section, "size")}
    return dict(cpu=c.get(core, "type"), vlen=expected_vlen, clocks=clocks, cache_sizes=sizes,
                model_sha256=hashlib.sha256(json.dumps(model, sort_keys=True).encode()).hexdigest())


def run_mode(repo, out, gem5, runner, binary, trace, trace_meta, vlen, mode):
    dest = out / mode
    dest.mkdir()
    row = dict(mode=mode, **metadata(mode), samples=trace_meta["samples"], vlen=vlen,
               trace_sha256=trace_meta["sha256"], status="FAIL")
    cmd = [gem5, "--outdir=" + str(dest), runner, "--binary", binary, "--vlen", str(vlen), "--", "--trace", trace, "--mode", mode]
    (dest / "command.json").write_text(json.dumps([str(x) for x in cmd], indent=2) + "\n")
    start = time.monotonic()
    print(f"Running {mode}", flush=True)
    try:
        rc = logged(cmd, dest / "run.log", repo)
        log = (dest / "run.log").read_text(errors="replace")
        version = re.search(r"^gem5 version (.+)$", log, re.M)
        row["gem5_version"] = version.group(1).strip() if version else "unknown"
        if rc:
            raise RuntimeError(f"gem5 exit code {rc}")
        check = re.search(r"^reference_check=PASS hidden_spike_mismatches=0 output_spike_mismatches=0 prediction_mismatches=0\s*$", log, re.M)
        if not check:
            raise RuntimeError("Missing exact PyTorch spike/prediction PASS")
        final = re.search(r"^weight_layout=.*$", log, re.M)
        if not final:
            raise RuntimeError("Missing final inference completion line")
        fields = dict(re.findall(r"(\w+)=([^\s]+)", final.group(0)))
        if (fields.get("first_mode") != mode or fields.get("second_mode") != mode or
                fields.get("weight_layout") != "pretransposed_input_major" or
                fields.get("samples") != str(trace_meta["samples"]) or fields.get("timesteps") != str(trace_meta["timesteps"])):
            raise RuntimeError("Wrong measured mode/trace")
        for k in ("hidden_spike_checksum", "output_spike_checksum", "images_correct"):
            row[k] = int(fields[k])
        stats = roi(dest / "stats.txt")
        row.update({k: stat_int(stats, key) for k, key in METRICS.items()})
        if not row["cycles"] or not row["instructions"]:
            raise RuntimeError("Missing required first-ROI counters")
        cfg = config_info(dest / "config.ini", vlen)
        (dest / "model_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        row["model_sha256"] = cfg["model_sha256"]
        row["status"] = "PASS"
    except (OSError, ValueError, RuntimeError, KeyError, configparser.Error) as e:
        row["error"] = str(e)
        print(f"  FAIL: {e}; see {dest / 'run.log'}", flush=True)
    row["wall_seconds"] = round(time.monotonic() - start, 3)
    if row["status"] == "PASS":
        print(f"  PASS: cycles={row['cycles']} instructions={row['instructions']}", flush=True)
    return row


def write_summary(rows, out):
    # Only use same-config, correct conventional sparse baselines. No failed
    # row may receive a speedup. If any of the three are missing, leave best
    # sparse speedup blank rather than silently using a weaker baseline.
    valid = [r for r in rows if r["status"] == "PASS"]
    for r in rows:
        r["speedup_vs_acc1"] = r["speedup_vs_best_sparse"] = ""
        r["best_sparse_mode"] = ""
        if r["status"] != "PASS":
            continue
        group = [x for x in valid if x["model_sha256"] == r["model_sha256"] and x["gem5_version"] == r["gem5_version"]]
        acc1 = next((x for x in group if x["mode"] == SPARSE[0]), None)
        if acc1:
            r["speedup_vs_acc1"] = round(acc1["cycles"] / r["cycles"], 6)
        baseline = [x for x in group if x["mode"] in SPARSE]
        if len(baseline) == 3:
            best = min(baseline, key=lambda x: x["cycles"])
            r["best_sparse_mode"] = best["mode"]
            r["speedup_vs_best_sparse"] = round(best["cycles"] / r["cycles"], 6)
    keys = ["mode", "family", "window", "lmul", "weight_sharing", "membrane_resident", "samples", "vlen", "status",
            *METRICS, "speedup_vs_acc1", "speedup_vs_best_sparse", "best_sparse_mode", "hidden_spike_checksum", "output_spike_checksum",
            "images_correct", "gem5_version", "trace_sha256", "model_sha256", "wall_seconds", "error"]
    tmp = out / "summary.csv.tmp"
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    tmp.replace(out / "summary.csv")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", choices=(*SUITES, "custom"), default="full")
    p.add_argument("--modes", nargs="+", help="Explicit modes for --suite custom")
    p.add_argument("--list", action="store_true", help="Print selected modes without running")
    p.add_argument("--repo", type=Path, default=Path.cwd())
    p.add_argument("--out", type=Path)
    p.add_argument("--samples", type=int, help="Default 16 if exporting a fresh trace; inferred if --trace")
    p.add_argument("--seed", type=int, default=20261115)
    p.add_argument("--trace", type=Path, help="Use an existing exact SNNMNST2 trace")
    p.add_argument("--checkpoint", type=Path, default=Path("artifacts/mnist_snn.pt"))
    p.add_argument("--data", type=Path)
    p.add_argument("--build-dir", type=Path, default=Path("build-riscv-m5"))
    p.add_argument("--gem5-root", type=Path, default=Path(os.environ.get("GEM5_ROOT", str(Path.home() / "gem5-sim"))))
    p.add_argument("--gem5-bin", type=Path, default=os.environ.get("GEM5_BIN"))
    p.add_argument("--vlen", type=int, choices=(128, 256, 512), default=256)
    p.add_argument("--qemu", default="qemu-riscv64")
    p.add_argument("--objdump", default="riscv64-linux-gnu-objdump")
    p.add_argument("--jobs", type=int, default=2, help="Parallel CMake build jobs, not simulation jobs")
    p.add_argument("--no-build", action="store_true", help="Use a prebuilt RV64 binary; provenance must be checked manually")
    p.add_argument("--skip-qemu", action="store_true", help="Only if QEMU was validated separately, e.g. in a container")
    p.add_argument("--allow-dirty", action="store_true", help="Archive local tracked source changes rather than refusing them")
    p.add_argument("--fail-fast", action="store_true")
    a = p.parse_args()
    if a.suite == "custom":
        if not a.modes or any(m not in SUITES["full"] for m in a.modes):
            p.error("--suite custom requires recognised --modes")
        modes = list(dict.fromkeys(a.modes))
    else:
        if a.modes: p.error("--modes requires --suite custom")
        modes = SUITES[a.suite]
    if a.list:
        print(f"{len(modes)} modes:"); print("\n".join(modes)); return 0
    if a.out is None: p.error("--out is required")
    if a.samples is not None and not 1 <= a.samples <= 10000: p.error("invalid sample count")
    repo = a.repo.expanduser().resolve()
    out = a.out.expanduser().resolve()
    build = (repo / a.build_dir).resolve()
    root = a.gem5_root.expanduser().resolve()
    gem5 = (a.gem5_bin or root / "build/RISCV/gem5.opt").expanduser().resolve()
    runner = repo / "gem5/run_o3.py"
    if not runner.is_file() or not gem5.is_file():
        raise FileNotFoundError(f"Missing simulator or runner: {gem5}, {runner}")
    if any(m.startswith("sparse_rvv_tile") for m in modes) and not (repo / "src/temporal_tile.cpp").is_file():
        raise RuntimeError("Tile modes require the A+B branch; use --suite a on A")
    revision = capture(["git", "rev-parse", "HEAD"], repo)
    branch = capture(["git", "branch", "--show-current"], repo)
    diff = capture(["git", "diff", "HEAD", "--"], repo)
    if diff and not a.allow_dirty:
        raise RuntimeError("Commit tracked changes first or explicitly use --allow-dirty")
    out.mkdir(parents=True, exist_ok=False)
    (out / "source.diff").write_text(diff + "\n")
    (out / "git_status.txt").write_text(capture(["git", "status", "--short"], repo) + "\n")
    shutil.copy2(Path(__file__).resolve(), out / "study_runner.py")
    shutil.copy2(runner, out / "run_o3.py")
    cache = build / "CMakeCache.txt"
    if not cache.is_file():
        raise RuntimeError(f"Configure the RISC-V CMake build first: {build}")
    cache_text = cache.read_text(errors="replace")
    home = re.search(r"^CMAKE_HOME_DIRECTORY:INTERNAL=(.+)$", cache_text, re.M)
    if home and Path(home.group(1)).resolve() != repo and not a.no_build:
        raise RuntimeError("CMake cache belongs to a different source directory")
    if "SNN_GEM5:BOOL=ON" not in cache_text:
        raise RuntimeError("CMake build must enable SNN_GEM5")
    if not a.no_build:
        require_run(["cmake", "--build", build, "--target", "snn_test", "snn_mnist", f"-j{a.jobs}"], out / "build.log", repo)
    binary = build / "snn_mnist"; test = build / "snn_test"
    elf_check(binary); elf_check(test)
    shutil.copy2(cache, out / "CMakeCache.txt")
    for target in ("snn_kernels", "snn_mnist", "snn_test"):
        flags = build / "CMakeFiles" / (target + ".dir") / "flags.make"
        if flags.is_file(): shutil.copy2(flags, out / (target + "_flags.make"))
    for name in ("compile_commands.json",):
        f = build / name
        if f.is_file(): shutil.copy2(f, out / name)
    manifest = dict(branch=branch, revision=revision, runner_sha256=sha(runner), study_sha256=sha(Path(__file__)),
                    binary=str(binary), binary_sha256=sha(binary), test_binary_sha256=sha(test),
                    gem5_binary=str(gem5), gem5_sha256=sha(gem5), vlen=a.vlen, suite=a.suite, modes=modes,
                    qemu_skipped=a.skip_qemu, prebuilt=a.no_build, dirty_tracked=bool(diff))
    try:
        manifest["gem5_revision"] = capture(["git", "rev-parse", "HEAD"], root)
        (out / "gem5_source.diff").write_text(capture(["git", "diff", "HEAD", "--"], root) + "\n")
        (out / "gem5_git_status.txt").write_text(capture(["git", "status", "--short"], root) + "\n")
    except (OSError, subprocess.CalledProcessError):
        manifest["gem5_revision"] = "unavailable"
    compiler = re.search(r"^CMAKE_CXX_COMPILER:[^=]+=(.+)$", cache.read_text(errors="replace"), re.M)
    if compiler:
        manifest["compiler"] = compiler.group(1)
        try:
            manifest["compiler_version"] = capture([compiler.group(1), "--version"], repo)
        except (OSError, subprocess.CalledProcessError):
            manifest["compiler_version"] = "unavailable (possibly a prebuilt container binary)"
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not a.skip_qemu:
        (out / "qemu_version.txt").write_text(capture([a.qemu, "--version"], repo) + "\n")
        require_run([a.qemu, "-cpu", f"max,vlen={a.vlen}", test], out / "qemu.log", repo)
        if "All C++ LIF kernel tests passed" not in (out / "qemu.log").read_text(errors="replace"):
            raise RuntimeError("Missing QEMU kernel test PASS")
        print("QEMU correctness: PASS", flush=True)
    if shutil.which(a.objdump):
        rc = logged([a.objdump, "-d", "-C", binary], out / "snn_mnist.disassembly.txt", repo)
        manifest["disassembly_ok"] = rc == 0
    else:
        manifest["disassembly_ok"] = False
        print("Note: objdump unavailable; inspect final vector assembly separately.", flush=True)
    trace = out / "trace_im.bin"
    if a.trace:
        shutil.copy2(a.trace.expanduser().resolve(), trace)
        manifest["trace_source"] = str(a.trace.expanduser().resolve())
    else:
        checkpoint = (repo / a.checkpoint).resolve()
        manifest["checkpoint_sha256"] = sha(checkpoint)
        cmd = [sys.executable, "-m", "python.export_mnist_trace", "--checkpoint", checkpoint,
               "--samples", str(a.samples or 16), "--seed", str(a.seed), "--input-major", "--output", trace]
        if a.data: cmd.extend(["--data", a.data.expanduser().resolve()])
        require_run(cmd, out / "trace_export.log", repo)
        manifest["seed"] = a.seed
    tm = validate_trace(trace)
    if a.samples is not None and a.samples != tm["samples"]:
        raise ValueError("Requested sample count does not match existing trace")
    manifest["trace"] = tm
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if (repo / "temporal_reuse.py").is_file():
        require_run([sys.executable, repo / "temporal_reuse.py", trace, "--vlen", str(a.vlen), "--csv", out / "temporal_reuse.csv"], out / "temporal_reuse.log", repo)
    rows = []
    anchor = None
    for mode in modes:
        row = run_mode(repo, out, gem5, runner, binary, trace, tm, a.vlen, mode)
        if row["status"] == "PASS":
            signature = (row["gem5_version"], row["model_sha256"], row["hidden_spike_checksum"], row["output_spike_checksum"], row["images_correct"])
            if anchor is None: anchor = signature
            elif signature != anchor:
                row["status"] = "FAIL"
                row["error"] = "Model config or output checksum differs from the first correct run"
                print(f"  FAIL: {row['error']}", flush=True)
        rows.append(row)
        write_summary(rows, out)
        if row["status"] != "PASS" and a.fail_fast: break
    ok = sum(r["status"] == "PASS" for r in rows)
    print(f"Completed {ok}/{len(modes)} correct modes. Results: {out / 'summary.csv'}")
    return 0 if ok == len(modes) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as e:
        print(f"Study failed: {e}", file=sys.stderr)
        sys.exit(2)
