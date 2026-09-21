"""Burgers experiment specification, scenario expansion, and identities."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from ...AOTD import AlgorithmConfig


ROOT = Path(__file__).resolve().parents[4]


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def source_hash():
    core = [
        path
        for path in sorted((ROOT / "src/newton").rglob("*.py"))
        if "studies" not in path.parts
    ]
    study = [
        Path(__file__),
        Path(__file__).with_name("runner.py"),
        Path(__file__).parents[1] / "common.py",
    ]
    paths = sorted(core + study)
    return hashlib.sha256(
        b"".join(
            str(path.relative_to(ROOT)).encode() + path.read_bytes() for path in paths
        )
    ).hexdigest()


def _positive_unique_ints(values, name):
    if (
        not isinstance(values, list)
        or not values
        or len(values) != len(set(values))
        or any(not isinstance(value, int) or value < 1 for value in values)
    ):
        raise ValueError(f"{name} must contain unique positive integers")


def load_config(path):
    config = json.loads(Path(path).read_text())
    required = {
        "schema",
        "study",
        "description",
        "problem",
        "grid",
        "algorithm",
        "variants",
        "baselines",
    }
    if set(config) != required or config["schema"] != 1 or config["study"] != "burgers":
        raise ValueError("Unexpected Burgers configuration schema, fields, or study")

    problem = config["problem"]
    if set(problem) != {
        "nx",
        "nu",
        "dt",
        "viscosity",
        "control_penalty",
        "amplitude",
        "target",
        "amplitude_noise",
        "perturbation_scale",
    }:
        raise ValueError("Unexpected Burgers problem fields")
    if (
        not isinstance(problem["nx"], int)
        or not isinstance(problem["nu"], int)
        or problem["nx"] < 1
        or problem["nu"] < 1
    ):
        raise ValueError("Burgers state and control dimensions must be positive")
    for name in (
        "dt",
        "viscosity",
        "control_penalty",
        "amplitude",
        "amplitude_noise",
        "perturbation_scale",
    ):
        if not isinstance(problem[name], (int, float)) or problem[name] <= 0:
            raise ValueError(f"problem.{name} must be positive")

    grid = config["grid"]
    if set(grid) != {"N", "M", "seeds"}:
        raise ValueError("Unexpected Burgers grid fields")
    for name in ("N", "M", "seeds"):
        _positive_unique_ints(grid[name], f"grid.{name}")
    if any(horizon % subproblems for horizon in grid["N"] for subproblems in grid["M"]):
        raise ValueError("Every Burgers horizon must be divisible by every M")

    if set(config["algorithm"]) != {"adaptive", "fixed"}:
        raise ValueError("Require exactly adaptive and fixed algorithm settings")
    adaptive = AlgorithmConfig(**config["algorithm"]["adaptive"])
    fixed = AlgorithmConfig(**config["algorithm"]["fixed"])
    if not adaptive.adaptive or fixed.adaptive:
        raise ValueError("Require adaptive=true and fixed.adaptive=false")

    variants = config["variants"]
    if set(variants) != {
        "fotd_overlaps",
        "fotd_qlp_tolerance",
        "fotd_qlp_rank_tol",
        "fotd_qlp_max_inner_iters",
        "fotd_lu_max_inner_iters",
        "sketch_seed",
        "sketch_restart",
        "sketch_max_inner_iters",
    }:
        raise ValueError("Unexpected Burgers variant fields")
    _positive_unique_ints(variants["fotd_overlaps"], "variants.fotd_overlaps")
    for name in ("fotd_qlp_tolerance", "fotd_qlp_rank_tol"):
        if not isinstance(variants[name], (int, float)) or variants[name] <= 0:
            raise ValueError(f"variants.{name} must be positive")
    for name in (
        "fotd_qlp_max_inner_iters",
        "fotd_lu_max_inner_iters",
        "sketch_restart",
        "sketch_max_inner_iters",
    ):
        if not isinstance(variants[name], int) or variants[name] < 1:
            raise ValueError(f"variants.{name} must be a positive integer")
    if not isinstance(variants["sketch_seed"], int) or variants["sketch_seed"] < 0:
        raise ValueError("variants.sketch_seed must be a nonnegative integer")
    if set(config["baselines"]) != {"schwarz"}:
        raise ValueError("Require exactly the Schwarz baseline")
    schwarz = config["baselines"]["schwarz"]
    if set(schwarz) != {"N", "b", "mu_pen", "max_outer", "tol_kkt"}:
        raise ValueError("Unexpected Schwarz settings")
    if schwarz["N"] not in grid["N"]:
        raise ValueError("Schwarz N must be in the Burgers horizon grid")
    if not isinstance(schwarz["b"], int) or schwarz["b"] < 0:
        raise ValueError("Schwarz overlap must be a nonnegative integer")
    if not isinstance(schwarz["max_outer"], int) or schwarz["max_outer"] < 1:
        raise ValueError("Schwarz max_outer must be a positive integer")
    for name in ("mu_pen", "tol_kkt"):
        if not isinstance(schwarz[name], (int, float)) or schwarz[name] <= 0:
            raise ValueError(f"Schwarz {name} must be positive")

    config["algorithm"]["adaptive"] = asdict(adaptive)
    config["algorithm"]["fixed"] = asdict(fixed)
    return config


def jobs(config):
    result = []
    variants = config["variants"]

    def add(method, horizon, subproblems, seed, b=None, settings=None):
        if method.startswith(("AOTD", "FOTD")):
            algorithm = dict(settings, M=subproblems, verbose=True)
            algorithm = asdict(AlgorithmConfig(**algorithm))
        else:
            algorithm = dict(settings, verbose=True)
        job = dict(
            method=method,
            N=horizon,
            M=subproblems,
            seed=seed,
            b=b,
            sketch_seed=(variants["sketch_seed"] if method == "AOTD-sketch" else None),
            config=algorithm,
            problem=config["problem"],
        )
        job["config_sha256"] = digest(job)
        job["id"] = (
            f"{method}_N{horizon}_M{subproblems}_b{b}_seed{seed}_"
            f"{job['config_sha256'][:12]}"
        )
        result.append(job)

    for horizon in config["grid"]["N"]:
        for subproblems in config["grid"]["M"]:
            for seed in config["grid"]["seeds"]:
                adaptive = config["algorithm"]["adaptive"]
                add("AOTD-rebuild", horizon, subproblems, seed, settings=adaptive)
                add(
                    "AOTD-sketch",
                    horizon,
                    subproblems,
                    seed,
                    settings=dict(
                        adaptive,
                        inner_solver="sketch",
                        sketch_restart=variants["sketch_restart"],
                        max_inner_iters=variants["sketch_max_inner_iters"],
                    ),
                )
                fixed = config["algorithm"]["fixed"]
                for overlap in variants["fotd_overlaps"]:
                    add(
                        "FOTD-LU",
                        horizon,
                        subproblems,
                        seed,
                        overlap,
                        dict(
                            fixed,
                            b0=overlap,
                            inner_solver="direct",
                            use_preconditioner=False,
                            max_inner_iters=variants["fotd_lu_max_inner_iters"],
                        ),
                    )
                    add(
                        "FOTD-QLP",
                        horizon,
                        subproblems,
                        seed,
                        overlap,
                        dict(
                            fixed,
                            b0=overlap,
                            eps_i_0=variants["fotd_qlp_tolerance"],
                            inner_rank_tol=variants["fotd_qlp_rank_tol"],
                            inner_solver="gmres_qlp",
                            use_preconditioner=True,
                            max_inner_iters=variants["fotd_qlp_max_inner_iters"],
                        ),
                    )
                schwarz = config["baselines"]["schwarz"]
                if horizon == schwarz["N"]:
                    add(
                        "Schwarz",
                        horizon,
                        subproblems,
                        seed,
                        schwarz["b"],
                        {
                            key: value
                            for key, value in schwarz.items()
                            if key not in ("N", "b")
                        },
                    )
    if len({job["id"] for job in result}) != len(result):
        raise ValueError("Duplicate Burgers scenarios")
    return result


def audit(row):
    """Reject Newton records that apply an uncertified step."""
    for step in row.get("trajectory", []):
        if step["step_applied"]:
            passes = step["passes"]
            if not passes:
                raise ValueError("Applied step has no local-solve certificate")
            if not all(
                solve["converged"]
                and solve["residual_norm"] <= solve["residual_threshold"]
                for pass_record in passes
                for solve in pass_record["local_solves"]
            ):
                raise ValueError("Applied step fails original-system local accuracy")
            last = passes[-1]
            if row["config"]["adaptive"] and not (
                last["outcome"] == "accept"
                and last["r_norm"] <= last["acc_rhs"]
                and last["gdir"] <= last["descent_rhs"]
            ):
                raise ValueError("Applied step fails accuracy/descent")
            if not step["accepted"] or step["alpha"] <= 0:
                raise ValueError("Applied step fails line search")
        elif step["alpha"] != 0:
            raise ValueError("Rejected step has nonzero alpha")
