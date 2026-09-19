#!/usr/bin/env python3
"""Run AOTD experiments and plot generated or archived data."""

import argparse
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
GROUPS = ("swing", "burgers", "rates", "globalization")
RUNNERS = {
    "swing": [
        "run_comparison.py",
        "run_baselines.py",
        "run_sketch.py",
    ],
    "burgers": [
        "run_comparison.py",
        "run_schwarz_10k.py",
        "run_sketch_10k.py",
        "run_sketch_50k.py",
    ],
    "rates": ["run_sweep.py"],
    "globalization": ["run_sweep.py"],
}
RENDERERS = {
    "swing": ["plot_results.py"],
    "burgers": ["plot_results.py"],
    "rates": ["plot_results.py", "make_table.py"],
    "globalization": ["plot_results.py"],
}
ARTIFACTS = {
    "swing": {
        "swing_convergence.pdf": "figure_2.pdf",
        "swing_adaptation.pdf": "figure_3.pdf",
        "swing_highlight_table.tex": "table_1.tex",
        "swing_appendix_table.tex": "table_5.tex",
    },
    "burgers": {
        "burgers_convergence.pdf": "figure_4.pdf",
        "burgers_adaptation.pdf": "figure_5.pdf",
        "burgers_highlight_table.tex": "table_2.tex",
        "burgers_appendix_table.tex": "table_6.tex",
    },
    "rates": {
        "rate_ablation.pdf": "figure_6.pdf",
        "rate_ablation_table.tex": "table_4.tex",
    },
    "globalization": {"ablation_eta_init.pdf": "figure_7.pdf"},
}


def archive(group):
    with gzip.open(ROOT / "data" / f"{group}.json.gz", "rt") as f:
        return json.load(f)


def environment(work, workers=1):
    env = os.environ.copy()
    env.update(
        AOTD_WORK=str(work),
        AOTD_WORKERS=str(workers),
        PYTHONPATH=str(ROOT / "src"),
        MPLBACKEND="Agg",
        MPLCONFIGDIR=str(work / ".matplotlib"),
        PYTHONDONTWRITEBYTECODE="1",
    )
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[name] = "1"
    return env


def invoke(group, script, work, workers=1):
    subprocess.run(
        [sys.executable, str(ROOT / "experiments" / group / script)],
        cwd=work,
        env=environment(work, workers),
        check=True,
    )


def verify():
    counts = {"swing": 160, "burgers": 185, "rates": 80}
    for group in GROUPS:
        data = archive(group)
        records = [v for k, v in data.items() if k.startswith("records/")]
        if group in counts and len(records) != counts[group]:
            raise RuntimeError(f"Unexpected record count for {group}")
        if any("error" in r for r in records):
            raise RuntimeError(f"Error record in {group}")
        if group == "globalization" and len(data["ablation_eta_data.json"]) != 36:
            raise RuntimeError("Expected 36 globalization runs")
        if group == "rates" and {r["M"] for r in records} != {20}:
            raise RuntimeError("Expected only M=20 rate records")
        if group == "swing" and len(data["baselines_sigma06.json"]) != 125:
            raise RuntimeError("Expected 125 Swing baseline records")
        print(
            f"Verified {group}: {len(records) if records else 36} run records",
            flush=True,
        )
    # Numerical anchors from the printed tables, checked independently of rendering.
    swing = list(archive("swing").values())
    burgers = list(archive("burgers").values())
    for rows, horizon, ms, expected, scale in [
        (swing, 1000, [4, 10, 20, 50], [10.0, 7.5, 14.4, 59.2], 1e7),
        (burgers, 10000, [10, 50, 250], [14.8, 19.9, 41.0], 1e9),
        (burgers, 50000, [10, 50, 250], [55.5, 73.4, 97.6], 1e9),
    ]:
        for m, wanted in zip(ms, expected):
            cell = [
                r
                for r in rows
                if isinstance(r, dict)
                and r.get("method_label") == "AOTD-rebuild"
                and r.get("N") == horizon
                and r.get("M") == m
            ]
            if len(cell) != 5:
                raise RuntimeError(f"Expected five seeds for N={horizon}, M={m}")
            got = round(sum(r["flops"] for r in cell) / 5 / scale, 1)
            if got != wanted:
                raise RuntimeError(
                    f"Paper table mismatch N={horizon}, M={m}: {got} != {wanted}"
                )
    print("Archived record counts and AOTD FLOP entries verified.", flush=True)


def plot(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not shutil.which("latex"):
        raise RuntimeError("Rendering needs LaTeX (see README.md for system packages).")
    for group in GROUPS:
        with tempfile.TemporaryDirectory(prefix=f"aotd-{group}-") as temp:
            work = Path(temp)
            if args.input:
                source = args.input.resolve() / group
                if not source.is_dir():
                    raise RuntimeError(f"Missing recomputed study: {source}")
                shutil.copytree(source, work, dirs_exist_ok=True)
            else:
                for name, value in archive(group).items():
                    target = work / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(json.dumps(value))
            for script in RENDERERS[group]:
                invoke(group, script, work)
            for name, dest in ARTIFACTS[group].items():
                shutil.copyfile(work / name, out / dest)
    # A small standalone document for each table; \\tablesize is used by the original renderer.
    for i in (1, 2, 4, 5, 6):
        stem = f"table_{i}"
        doc = (
            r"\documentclass[10pt]{article}"
            "\n"
            r"\usepackage[margin=0.5in,landscape]{geometry}"
            "\n"
            r"\usepackage{booktabs,amsmath,amssymb,graphicx}"
            "\n"
            r"\providecommand{\tablesize}{\footnotesize}"
            "\n"
            r"\begin{document}\thispagestyle{empty}"
            "\n"
            rf"\input{{{stem}.tex}}"
            "\n"
            r"\end{document}"
            "\n"
        )
        (out / f"{stem}_standalone.tex").write_text(doc)
        if args.compile_tables:
            with tempfile.TemporaryDirectory(prefix="aotd-tex-") as temp:
                p = Path(temp)
                (p / f"{stem}.tex").write_text((out / f"{stem}.tex").read_text())
                (p / "doc.tex").write_text(doc)
                result = subprocess.run(
                    [
                        "pdflatex",
                        "-interaction=nonstopmode",
                        "-halt-on-error",
                        "doc.tex",
                    ],
                    cwd=p,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                if result.returncode:
                    raise RuntimeError(result.stdout[-6000:])
                shutil.copyfile(p / "doc.pdf", out / f"{stem}.pdf")
    print(f"Generated Figures 2–7 and Tables 1, 2, 4, 5, 6 in {out}")


def run(args):
    groups = GROUPS if args.study == "all" else [args.study]
    for group in groups:
        work = args.output.resolve() / group
        work.mkdir(parents=True, exist_ok=True)
        for script in RUNNERS[group]:
            invoke(group, script, work, args.workers)
        if group == "globalization":
            rows = json.loads((work / "ablation_eta_data.json").read_text())
            if len(rows) != 36:
                raise RuntimeError("Incomplete globalization sweep")
        else:
            expected = {"swing": 160, "burgers": 185, "rates": 80}[group]
            rows = [
                json.loads(p.read_text()) for p in (work / "records").glob("*.json")
            ]
            if len(rows) != expected or any("error" in r for r in rows):
                raise RuntimeError(
                    f"Incomplete or failed {group} sweep; inspect {work}"
                )
            if group == "swing":
                baselines = json.loads((work / "baselines_sigma06.json").read_text())
                if len(baselines) != 125 or any("error" in r for r in baselines):
                    raise RuntimeError(
                        f"Incomplete or failed Swing baselines; inspect {work}"
                    )
    print("Completed studies:", ", ".join(groups))


def smoke():
    # Fresh paper-size case; historical work counts used incorrect acceptance tests.
    with tempfile.TemporaryDirectory(prefix="aotd-smoke-") as tmp:
        work = Path(tmp)
        code = """import importlib.util, json, sys
p=sys.argv[1]
s=importlib.util.spec_from_file_location("sweep",p)
m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
r=m.run_one(("AOTD-rebuild",None,20,1))
assert "error" not in r, r.get("error")
assert r["converged"], r
assert r["gauss_newton"] is False
for t in r["trajectory"]:
    assert t["step_applied"] and t["passes"][-1]["outcome"] == "accept"
    for p in t["passes"]:
        assert all(c["converged"] and c["residual_norm"] <= c["residual_threshold"]
                   for c in p["local_solves"])
print(json.dumps({k:r[k] for k in ("kkt","flops","outer_iters","inner_iters")}))
"""
        r = subprocess.run(
            [
                sys.executable,
                "-c",
                code,
                str(ROOT / "experiments/swing/run_comparison.py"),
            ],
            cwd=work,
            env={**environment(work), "AOTD_MAX_PASSES": "50"},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        actual = json.loads(r.stdout.strip().splitlines()[-1])
        print(
            "Corrected Swing N=1000, M=20, seed=1 converges with certified local residuals and accepted steps:",
            actual,
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "verify", help="Check archived record counts and reference work values"
    )
    sub.add_parser(
        "smoke", help="Recompute one complete deterministic paper-sized Swing case"
    )
    r = sub.add_parser(
        "plot", help="Generate figures and tables from explicitly selected data"
    )
    r.add_argument("--output", type=Path, default=ROOT / "results/paper")
    source = r.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input",
        type=Path,
        help="Root containing generated swing/burgers/rates/globalization data",
    )
    source.add_argument(
        "--archived", action="store_true", help="Use bundled paper data"
    )
    r.add_argument("--compile-tables", action="store_true")
    r = sub.add_parser(
        "run", help="Generate new experimental data; Burgers is expensive (see README)"
    )
    r.add_argument("--study", choices=(*GROUPS, "all"), default="all")
    r.add_argument("--output", type=Path, default=ROOT / "results/runs")
    r.add_argument("--workers", type=int, default=1)
    args = p.parse_args()
    if getattr(args, "workers", 1) < 1:
        p.error("--workers must be positive")
    if args.command == "verify":
        verify()
    elif args.command == "smoke":
        smoke()
    elif args.command == "plot":
        plot(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
