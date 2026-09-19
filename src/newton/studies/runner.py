"""Resumable study runner with isolated live logs and explicit timing provenance."""

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import importlib
import json
import logging
import os
from pathlib import Path
import platform
import signal
import sys
import time
import traceback

import numpy as np
import scipy
import casadi
import newton as N
from ..baselines.common import initial_guess, pack_traj
from .spec import ROOT, audit, digest, jobs, load_config, source_hash


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, default=float, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def progress_logger(output):
    logger = logging.getLogger("swing.progress")
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    for handler in (
        logging.StreamHandler(),
        logging.FileHandler(output / "progress.log"),
    ):
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)
    logger.propagate = False
    return logger


@contextmanager
def scenario_log(path):
    """Capture Python and native Ipopt output, line buffered, without interleaving."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = [os.dup(fd) for fd in (1, 2)]
    with open(path, "a", buffering=1, encoding="utf-8") as stream:
        try:
            for fd in (1, 2):
                os.dup2(stream.fileno(), fd)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            for fd, old in zip((1, 2), saved):
                os.dup2(old, fd)
                os.close(old)


def execute(job, output, identity, resume):
    output = Path(output)
    logger = progress_logger(output)
    record_path = output / "records" / f"{job['id']}.json"
    expected = dict(
        schema=3, job_id=job["id"], config_sha256=job["config_sha256"], **identity
    )
    if resume and record_path.exists():
        row = json.loads(record_path.read_text())
        if all(row.get(k) == v for k, v in expected.items()) and "error" not in row:
            audit(row)
            logger.info(
                "CACHED %s KKT=%.3e stop=%s", job["id"], row["kkt"], row["stop_reason"]
            )
            return row
    logger.info("START %s log=logs/%s.log", job["id"], job["id"])
    with scenario_log(output / "logs" / f"{job['id']}.log"):
        print(
            f"\n=== {job['method']} N={job['problem']['N']} M={job['M']} "
            f"b={job['b']} seed={job['seed']} ===",
            flush=True,
        )
        print(
            f"config_sha256={job['config_sha256']}\nsource_sha256={identity['source_sha256']}\n"
            f"execution_mode={identity['execution_mode']} started={datetime.now(timezone.utc).isoformat()}",
            flush=True,
        )
        try:
            row = solve_job(job)
            row.update(expected)
            row["timing_valid"] = identity["execution_mode"] == "uncontended"
            audit(row)
            print(
                f"[final] converged={row['converged']} stop={row['stop_reason']} "
                f"outer_iters={row['outer_iters']} kkt={row['kkt']:.12e} "
                f"feas={row['feas']:.12e} inner_iters={row['inner_iters']} "
                f"flops={row['flops']:.12e} wall={row['wall']:.6f} "
                f"timing_valid={row['timing_valid']}",
                flush=True,
            )
        except Exception as exc:
            traceback.print_exc()
            row = dict(
                expected,
                method=job["method"],
                M=job["M"],
                b=job["b"],
                seed=job["seed"],
                config=job["config"],
                error=f"{type(exc).__name__}: {exc}",
            )
        atomic_json(record_path, row)
    return row


def solve_job(job):
    p = job["problem"].copy()
    noise, sketch_seed = p.pop("noise_std"), p.pop("sketch_seed")
    initial = N.make_swing_problem(**p)
    x0 = N.swing_x0(initial.params, kick=p["kick"]) + noise * np.random.default_rng(
        job["seed"]
    ).standard_normal(2 * N.N_BUS)
    problem = N.make_swing_problem(**p, x0=x0)
    importlib.import_module("newton.AOTDsolver")._SKETCH_RNG = np.random.default_rng(
        sketch_seed
    )
    method = job["method"]
    row = dict(
        method=method,
        method_label=method,
        N=p["N"],
        M=job["M"],
        b=job["b"],
        seed=job["seed"],
        config=job["config"],
        problem=job["problem"],
    )
    start = time.perf_counter()
    if method.startswith(("AOTD", "FOTD")):
        x, u = initial_guess(problem)
        result = N.run_algorithm(
            problem,
            N.AlgorithmConfig(**job["config"]),
            pack_traj(problem, x, u),
            np.zeros(N.n_lam(problem)),
        )
        row.update(
            family="newton",
            converged=bool(result["converged"]),
            stop_reason=result["stop_reason"],
            kkt=float(result["final_grad_L_norm"]),
            feas=float(result["final_feas"]),
            cost=float(result["final_cost"]),
            flops=float(result["total_flops"]),
            outer_iters=sum(t["step_applied"] for t in result["trajectory"]),
            outer_checks=result["outer_iters"],
            inner_iters=result["total_inner_iters"],
            matvecs=result["total_matvecs"],
            b_list=result["b_list"],
            eps_i_list=result["eps_i_list"],
            trajectory=result["trajectory"],
        )
        serial, parallel = result["t_subsolve_serial"], result["t_subsolve_parallel"]
    else:
        settings = job["config"]
        if method == "IPOPT":
            result = N.solve_ipopt(problem, **settings)
        elif method == "Schwarz":
            result = N.solve_schwarz(problem, M=job["M"], b=job["b"], **settings)
        elif method == "MultiShoot":
            result = N.solve_multishoot(problem, M=job["M"], **settings)
        else:
            result = N.solve_admm(problem, M=job["M"], **settings)
        converged = bool(result.final_grad_L_norm <= settings["tol_kkt"])
        row.update(
            family="baseline",
            converged=converged,
            stop_reason="kkt"
            if converged
            else result.extra.get("stop_reason", "stopped"),
            kkt=float(result.final_grad_L_norm),
            feas=float(result.final_feas),
            cost=float(result.final_cost),
            flops=float(result.extra["flops"]),
            outer_iters=result.iters,
            inner_iters=result.extra.get("ipopt_iters", -1),
            history=list(result.history or []),
        )
        serial = result.extra.get("t_subsolve_serial", 0.0)
        parallel = result.extra.get("t_subsolve_parallel", 0.0)
    row["wall"] = time.perf_counter() - start
    row["par_wall"] = row["wall"] - serial + parallel
    return row


def main(study):
    parser = argparse.ArgumentParser(
        description=f"Run the corrected IEEE39 {study} study."
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / f"{study}.json"
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--uncontended",
        action="store_true",
        help="Sequential scenarios; one BLAS thread; timing eligible.",
    )
    modes.add_argument(
        "--parallel",
        action="store_true",
        help="Concurrent scenarios; timings excluded from tables.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of concurrent scenarios (parallel only; default 4).",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only matching verified records, including unsuccessful solves.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and list jobs without running solvers.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Run the first n jobs for validation; result remains partial.",
    )
    args = parser.parse_args()
    if args.uncontended and args.workers is not None:
        parser.error("--workers is only valid with --parallel")
    workers = args.workers or (1 if args.uncontended else 4)
    if (
        workers < 1
        or (args.workers is not None and args.workers < 1)
        or (args.limit is not None and args.limit < 1)
    ):
        parser.error("workers and limit must be positive")
    config = load_config(args.config, study)
    scenarios = jobs(config)
    selected = scenarios[: args.limit] if args.limit else scenarios
    mode = "uncontended" if args.uncontended else "parallel"
    output = (args.output or ROOT / "results/production" / study / mode).resolve()
    if not output.is_relative_to(ROOT / "results"):
        parser.error("Output must be inside the dev repository results/ directory")
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    study=study,
                    mode=mode,
                    workers=workers,
                    output=str(output),
                    expected=len(scenarios),
                    selected=len(selected),
                    jobs=[j["id"] for j in selected],
                ),
                indent=2,
            )
        )
        return
    output.mkdir(parents=True, exist_ok=True)
    for directory in ("records", "logs"):
        (output / directory).mkdir(exist_ok=True)
    # Coordinate our own experiments: uncontended excludes ALL other study runners.
    # External machine load still needs to be controlled by the operator.
    with (
        open(ROOT / "results/.study_execution.lock", "a") as machine_lock,
        open(output / ".lock", "a") as out_lock,
    ):
        try:
            fcntl.flock(
                machine_lock,
                (fcntl.LOCK_EX if args.uncontended else fcntl.LOCK_SH) | fcntl.LOCK_NB,
            )
            fcntl.flock(out_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(
                "Another study conflicts with this output or uncontended execution"
            )
        environment = dict(
            python=sys.version,
            numpy=np.__version__,
            scipy=scipy.__version__,
            casadi=casadi.__version__,
            platform=platform.platform(),
            hostname=platform.node(),
            blas_threads=1,
        )
        identity = dict(
            source_sha256=source_hash(),
            study_sha256=digest(config),
            execution_mode=mode,
            environment_sha256=digest(environment),
        )
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if any(previous.get(k) != v for k, v in identity.items()):
                parser.error(
                    "Output contains another configuration/source/environment/mode; choose a new --output"
                )
            if not args.resume:
                parser.error(
                    "Output already exists; use --resume or choose a new --output"
                )
        manifest = dict(
            schema=3,
            study=study,
            config=config,
            **identity,
            workers=workers,
            environment=environment,
            started=datetime.now(timezone.utc).isoformat(),
            expected=len(scenarios),
            completed=0,
            job_ids=[j["id"] for j in scenarios],
            status="running",
            records=[],
        )
        atomic_json(manifest_path, manifest)
        logger = progress_logger(output)
        logger.info(
            "STUDY %s mode=%s workers=%d scenarios=%d/%d",
            study,
            mode,
            workers,
            len(selected),
            len(scenarios),
        )
        rows = []

        def record_finished(job, row):
            rows.append(row)
            manifest.update(
                completed=len(rows), records=sorted(rows, key=lambda r: r["job_id"])
            )
            atomic_json(manifest_path, manifest)
            logger.info(
                "DONE %d/%d %s %s",
                len(rows),
                len(selected),
                job["id"],
                row.get(
                    "error",
                    f"stop={row.get('stop_reason')} KKT={row.get('kkt', float('nan')):.3e}",
                ),
            )

        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Interrupted by signal {signum}")

        signal.signal(signal.SIGTERM, interrupted)
        try:
            if args.uncontended:
                # Finish progress/manifest I/O before starting the next timed scenario.
                for job in selected:
                    record_finished(
                        job, execute(job, str(output), identity, args.resume)
                    )
            else:
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    remaining = iter(selected)
                    pending = {}

                    def submit_next():
                        job = next(remaining, None)
                        if job is not None:
                            pending[
                                pool.submit(
                                    execute, job, str(output), identity, args.resume
                                )
                            ] = job

                    for _ in range(workers):
                        submit_next()
                    while pending:
                        done, _ = wait(pending, return_when=FIRST_COMPLETED)
                        for future in done:
                            job = pending.pop(future)
                            try:
                                row = future.result()
                            except Exception as exc:
                                row = dict(
                                    job_id=job["id"], error=f"Worker failed: {exc}"
                                )
                            record_finished(job, row)
                            submit_next()
        except BaseException:
            manifest["status"] = "interrupted"
            atomic_json(manifest_path, manifest)
            raise
        manifest["status"] = (
            "failed"
            if any("error" in r for r in rows)
            else "complete"
            if len(rows) == len(scenarios)
            else "partial"
        )
        manifest["finished"] = datetime.now(timezone.utc).isoformat()
        manifest["converged"] = sum(r.get("converged", False) for r in rows)
        atomic_json(manifest_path, manifest)
        logger.info(
            "FINISHED status=%s converged=%d/%d manifest=%s",
            manifest["status"],
            manifest["converged"],
            len(rows),
            manifest_path,
        )
        if manifest["status"] == "failed":
            raise SystemExit(1)
