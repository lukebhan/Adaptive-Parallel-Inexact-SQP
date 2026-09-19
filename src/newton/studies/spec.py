"""Experiment specifications, job expansion, and reproducible identities."""

from dataclasses import asdict
from itertools import product
import hashlib
import json
from pathlib import Path

from ..AOTD import AlgorithmConfig

ROOT = Path(__file__).resolve().parents[3]
STUDIES = ("swing", "rate_ablation", "eta_ablation")


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def source_hash():
    paths = sorted((ROOT / "src/newton").rglob("*.py"))
    paths = [
        p
        for p in paths
        if "studies" not in p.parts or p.name in ("runner.py", "spec.py")
    ]
    return hashlib.sha256(
        b"".join(str(p.relative_to(ROOT)).encode() + p.read_bytes() for p in paths)
    ).hexdigest()


def load_config(path, study):
    config = json.loads(Path(path).read_text())
    required = {
        "schema",
        "study",
        "description",
        "problem",
        "algorithm",
        "seeds",
        "M",
        "grid",
        "baselines",
    }
    if set(config) != required or config["schema"] != 1 or config["study"] != study:
        raise ValueError("Unexpected configuration schema, fields, or study")
    problem = config["problem"]
    if set(problem) != {
        "N",
        "dt",
        "mode",
        "pm_scale",
        "q_theta",
        "q_omega",
        "r_scale",
        "kick",
        "noise_std",
        "sketch_seed",
    }:
        raise ValueError("Unexpected problem fields")
    if not isinstance(problem["N"], int) or problem["N"] < 1:
        raise ValueError("N must be a positive integer")
    for field in ("seeds", "M"):
        values = config[field]
        if (
            not values
            or len(set(values)) != len(values)
            or any(
                not isinstance(x, int) or x < (1 if field == "M" else 0) for x in values
            )
        ):
            raise ValueError(f"Invalid {field}")
    if max(config["M"]) > problem["N"]:
        raise ValueError("M cannot exceed N")
    cfg = AlgorithmConfig(**config["algorithm"])
    if not cfg.adaptive:
        raise ValueError("The primary study algorithm must be adaptive")
    keys = {
        "swing": set(),
        "rate_ablation": {"varrho_b", "varrho_eps"},
        "eta_ablation": {"eta1_0", "eta2_0", "nu"},
    }[study]
    if set(config["grid"]) != keys:
        raise ValueError("Unexpected ablation grid")
    for key, values in config["grid"].items():
        if not values or len(values) != len(set(values)):
            raise ValueError(f"Empty or duplicate grid: {key}")
        for value in values:
            AlgorithmConfig(**dict(config["algorithm"], **{key: value}))
    if study == "swing":
        base = config["baselines"]
        if set(base) != {
            "fixed",
            "overlaps",
            "schwarz_overlaps",
            "ipopt",
            "schwarz",
            "multishoot",
            "admm",
        }:
            raise ValueError("Unexpected baseline specification")
        AlgorithmConfig(**base["fixed"])
        if base["fixed"]["adaptive"]:
            raise ValueError("FOTD must have adaptive=false")
    elif config["baselines"]:
        raise ValueError("Ablations do not have baseline jobs")
    if study == "eta_ablation" and len(config["seeds"]) != 1:
        raise ValueError("Eta figure requires one seed per initialization")
    if study != "swing" and len(config["M"]) != 1:
        raise ValueError("Ablations require one M")
    # Complete defaults in the saved manifest; hashes include every effective setting.
    config["algorithm"] = asdict(cfg)
    if study == "swing":
        config["baselines"]["fixed"] = asdict(
            AlgorithmConfig(**config["baselines"]["fixed"])
        )
    return config


def jobs(config):
    result = []

    def add(method, m, seed, b=None, overrides=None, settings=None):
        if method.startswith(("AOTD", "FOTD")):
            algorithm = dict(settings or config["algorithm"], M=m, verbose=True)
            algorithm.update(overrides or {})
            algorithm = asdict(AlgorithmConfig(**algorithm))
        else:
            algorithm = dict(settings, verbose=True)
        job = dict(
            method=method,
            M=m,
            seed=seed,
            b=b,
            config=algorithm,
            problem=config["problem"],
        )
        job["config_sha256"] = digest(job)
        job["id"] = f"{method}_M{m}_b{b}_seed{seed}_{job['config_sha256'][:12]}"
        result.append(job)

    grid = config["grid"]
    combos = (
        [dict(zip(grid, values)) for values in product(*grid.values())]
        if grid
        else [{}]
    )
    for overrides, m, seed in product(combos, config["M"], config["seeds"]):
        for method, solver in [("AOTD-rebuild", "gmres_qlp")] + (
            [("AOTD-sketch", "sketch")] if config["study"] == "swing" else []
        ):
            add(method, m, seed, overrides=dict(overrides, inner_solver=solver))
    if config["study"] == "swing":
        baseline = config["baselines"]
        for m, seed in product(config["M"], config["seeds"]):
            for b, method in product(baseline["overlaps"], ("FOTD-LU", "FOTD-QLP")):
                add(
                    method,
                    m,
                    seed,
                    b,
                    settings=baseline["fixed"],
                    overrides=dict(
                        b0=b,
                        inner_solver="direct" if method == "FOTD-LU" else "gmres_qlp",
                        use_preconditioner=method != "FOTD-LU",
                        max_inner_iters=400 if method == "FOTD-LU" else 100,
                    ),
                )
            for b in baseline["schwarz_overlaps"]:
                add("Schwarz", m, seed, b, settings=baseline["schwarz"])
            add("MultiShoot", m, seed, settings=baseline["multishoot"])
            add("ADMM", m, seed, settings=baseline["admm"])
        for seed in config["seeds"]:
            add("IPOPT", None, seed, settings=baseline["ipopt"])
    if len({j["id"] for j in result}) != len(result):
        raise ValueError("Duplicate scenarios")
    return result


def audit(row):
    """Reject records that apply an uncertified step, including resumed records."""
    for step in row.get("trajectory", []):
        if step["step_applied"]:
            last = step["passes"][-1]
            if not all(
                c["converged"] and c["residual_norm"] <= c["residual_threshold"]
                for c in last["local_solves"]
            ):
                raise ValueError("Applied step fails original-system local accuracy")
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
