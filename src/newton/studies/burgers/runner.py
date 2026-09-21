"""Resumable Burgers study runner with explicit timing provenance."""

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import fcntl
import importlib
import json
from pathlib import Path
import platform
import signal
import sys
import time
import traceback

import casadi
import numpy as np
import scipy

import newton as N
from ... import burgers_setting as bs
from ...baselines.common import initial_guess, pack_traj
from ..common import (
    add_execution_arguments,
    worker_count,
    atomic_json,
    progress_logger,
    scenario_log,
)
from .spec import ROOT, audit, digest, jobs, load_config, source_hash


def method_label(job):
    if job["b"] is None:
        return job["method"]
    return f"{job['method']}-b{job['b']}"


def make_problem(job):
    problem = job["problem"]
    nx, nu = int(problem["nx"]), int(problem["nu"])
    bs.set_dims(nx, nu=nu)
    params = N.BurgersParams(
        nu=float(problem["viscosity"]),
        amp=float(problem["amplitude"]),
        alpha=float(problem["control_penalty"]),
        target=problem["target"],
    )
    grid = N.burgers_grid()
    rng = np.random.default_rng(job["seed"])
    amplitude = params.amp * (
        1.0 + float(problem["amplitude_noise"]) * rng.standard_normal()
    )
    perturbation = float(problem["perturbation_scale"]) * (
        rng.uniform(-1, 1) * np.sin(2 * np.pi * grid)
        + rng.uniform(-1, 1) * np.sin(3 * np.pi * grid)
    )
    x0 = amplitude * np.sin(np.pi * grid) + perturbation
    horizon = job["N"]
    dt = float(problem["dt"])
    target = np.tile(N.burgers_target(params), (horizon + 1, 1))
    return N.Problem(
        horizon,
        dt,
        dt * np.eye(nx),
        dt * params.alpha * np.eye(nu),
        np.eye(nx),
        x0=x0,
        x_des=target,
        params=params,
    )


def solve_job(job):
    problem = make_problem(job)
    method = job["method"]
    row = dict(
        method=method,
        method_label=method_label(job),
        N=job["N"],
        M=job["M"],
        b=job["b"],
        seed=job["seed"],
        sketch_seed=job["sketch_seed"],
        config=job["config"],
        problem=job["problem"],
    )
    start = time.perf_counter()
    if method.startswith(("AOTD", "FOTD")):
        if job["sketch_seed"] is not None:
            solver = importlib.import_module("newton.AOTDsolver")
            solver._SKETCH_RNG = np.random.default_rng(job["sketch_seed"])
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
            stationarity=float(result["final_stationarity"]),
            cost=float(result["final_cost"]),
            flops=float(result["total_flops"]),
            outer_iters=sum(step["step_applied"] for step in result["trajectory"]),
            outer_checks=result["outer_iters"],
            inner_iters=result["total_inner_iters"],
            matvecs=result["total_matvecs"],
            b_list=result["b_list"],
            eps_i_list=result["eps_i_list"],
            trajectory=result["trajectory"],
        )
        serial = result["t_subsolve_serial"]
        parallel = result["t_subsolve_parallel"]
    else:
        result = N.solve_schwarz(
            problem,
            M=job["M"],
            b=job["b"],
            **job["config"],
        )
        converged = bool(result.final_grad_L_norm <= job["config"]["tol_kkt"])
        row.update(
            family="baseline",
            converged=converged,
            stop_reason="kkt" if converged else "max_iters",
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


def execute(job, output, identity, resume):
    output = Path(output)
    logger = progress_logger(output)
    record_path = output / "records" / f"{job['id']}.json"
    expected = dict(
        schema=3,
        job_id=job["id"],
        config_sha256=job["config_sha256"],
        **identity,
    )
    if resume and record_path.exists():
        row = json.loads(record_path.read_text())
        if all(row.get(key) == value for key, value in expected.items()) and (
            "error" not in row
        ):
            audit(row)
            logger.info(
                "CACHED %s KKT=%.3e stop=%s",
                job["id"],
                row["kkt"],
                row["stop_reason"],
            )
            return row
    logger.info("START %s log=logs/%s.log", job["id"], job["id"])
    with scenario_log(output / "logs" / f"{job['id']}.log"):
        print(
            f"\n=== {method_label(job)} N={job['N']} M={job['M']} "
            f"seed={job['seed']} ===",
            flush=True,
        )
        print(
            f"config_sha256={job['config_sha256']}\n"
            f"source_sha256={identity['source_sha256']}\n"
            f"execution_mode={identity['execution_mode']} "
            f"started={datetime.now(timezone.utc).isoformat()}",
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
                method_label=method_label(job),
                N=job["N"],
                M=job["M"],
                b=job["b"],
                seed=job["seed"],
                sketch_seed=job["sketch_seed"],
                config=job["config"],
                problem=job["problem"],
                error=f"{type(exc).__name__}: {exc}",
            )
        atomic_json(record_path, row)
    return row


def _parser():
    parser = argparse.ArgumentParser(description="Run the configured Burgers study.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "burgers" / "main.json",
    )
    add_execution_arguments(parser)
    return parser


def main():
    parser = _parser()
    args = parser.parse_args()
    workers = worker_count(parser, args)
    config = load_config(args.config)
    scenarios = jobs(config)
    selected = scenarios[: args.limit] if args.limit else scenarios
    mode = "uncontended" if args.uncontended else "parallel"
    output = (args.output or ROOT / "runs" / "burgers" / mode).resolve()
    if not output.is_relative_to(ROOT / "runs"):
        parser.error("Output must be inside the repository runs/ directory")
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    study="burgers",
                    mode=mode,
                    workers=workers,
                    output=str(output),
                    expected=len(scenarios),
                    selected=len(selected),
                    jobs=[job["id"] for job in selected],
                ),
                indent=2,
            )
        )
        return

    output.mkdir(parents=True, exist_ok=True)
    for directory in ("records", "logs"):
        (output / directory).mkdir(exist_ok=True)
    with (
        open(ROOT / "runs/.study_execution.lock", "a") as machine_lock,
        open(output / ".lock", "a") as output_lock,
    ):
        try:
            fcntl.flock(
                machine_lock,
                (fcntl.LOCK_EX if args.uncontended else fcntl.LOCK_SH) | fcntl.LOCK_NB,
            )
            fcntl.flock(output_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
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
            if any(previous.get(key) != value for key, value in identity.items()):
                parser.error(
                    "Output contains another configuration/source/environment/mode; "
                    "choose a new --output"
                )
            if not args.resume:
                parser.error(
                    "Output already exists; use --resume or choose a new --output"
                )
        manifest = dict(
            schema=3,
            study="burgers",
            config=config,
            **identity,
            workers=workers,
            environment=environment,
            started=datetime.now(timezone.utc).isoformat(),
            expected=len(scenarios),
            completed=0,
            job_ids=[job["id"] for job in scenarios],
            status="running",
            records=[],
        )
        atomic_json(manifest_path, manifest)
        logger = progress_logger(output)
        logger.info(
            "STUDY burgers mode=%s workers=%d scenarios=%d/%d",
            mode,
            workers,
            len(selected),
            len(scenarios),
        )
        rows = []

        def record_finished(job, row):
            rows.append(row)
            manifest.update(
                completed=len(rows),
                records=sorted(rows, key=lambda item: item["job_id"]),
            )
            atomic_json(manifest_path, manifest)
            logger.info(
                "DONE %d/%d %s %s",
                len(rows),
                len(selected),
                job["id"],
                row.get(
                    "error",
                    f"stop={row.get('stop_reason')} "
                    f"KKT={row.get('kkt', float('nan')):.3e}",
                ),
            )

        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Interrupted by signal {signum}")

        signal.signal(signal.SIGTERM, interrupted)
        try:
            if args.uncontended:
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
                            future = pool.submit(
                                execute, job, str(output), identity, args.resume
                            )
                            pending[future] = job

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
                                    job_id=job["id"],
                                    error=f"Worker failed: {exc}",
                                )
                            record_finished(job, row)
                            submit_next()
        except BaseException:
            manifest["status"] = "interrupted"
            atomic_json(manifest_path, manifest)
            raise
        manifest["status"] = (
            "failed"
            if any("error" in row for row in rows)
            else "complete"
            if len(rows) == len(scenarios)
            else "partial"
        )
        manifest["finished"] = datetime.now(timezone.utc).isoformat()
        manifest["converged"] = sum(row.get("converged", False) for row in rows)
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
