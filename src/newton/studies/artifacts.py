"""Read complete study manifests and render one independently reproducible artifact."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess

from .spec import ROOT, audit, digest, jobs

ARTIFACTS = {
    "swing_highlight_table": ("swing", 1),
    "swing_appendix_table": ("swing", 5),
    "swing_adaptation": ("swing", None),
    "swing_convergence": ("swing", None),
    "rate_ablation": ("rate_ablation", None),
    "rate_ablation_table": ("rate_ablation", 4),
    "ablation_eta_init": ("eta_ablation", None),
}


def read_study(path, study, timing=False):
    manifest = json.loads(Path(path).read_text())
    if (
        manifest.get("schema") != 3
        or manifest.get("study") != study
        or manifest.get("status") != "complete"
    ):
        raise ValueError(
            "Require a complete schema-3 manifest for this study; partial/error datasets cannot be rendered"
        )
    config, rows = manifest["config"], manifest["records"]
    expected = {j["id"]: j for j in jobs(config)}
    if (
        len(rows) != len(expected)
        or manifest["expected"] != len(expected)
        or {r["job_id"] for r in rows} != set(expected)
        or manifest["study_sha256"] != digest(config)
    ):
        raise ValueError("Missing, duplicate, or unexpected scenarios/configuration")
    if timing and manifest["execution_mode"] != "uncontended":
        raise ValueError(
            "Timing tables require --uncontended data; parallel elapsed times are diagnostic only"
        )
    for row in rows:
        job = expected[row["job_id"]]
        if "error" in row:
            raise ValueError(f"Scenario error: {row['job_id']}")
        for key in ("config", "problem", "method", "M", "b", "seed", "config_sha256"):
            if row[key] != job[key]:
                raise ValueError(f"Scenario {row['job_id']} has inconsistent {key}")
        for key in ("source_sha256", "study_sha256", "execution_mode"):
            if row[key] != manifest[key]:
                raise ValueError(f"Mixed record provenance: {key}")
        if row["timing_valid"] != (manifest["execution_mode"] == "uncontended"):
            raise ValueError("Incorrect timing eligibility")
        if (
            "environment_sha256" in manifest
            and row.get("environment_sha256") != manifest["environment_sha256"]
        ):
            raise ValueError("Mixed execution environments")
        audit(row)
    return manifest


def compile_table(directory, stem, number):
    build = directory / ".table_build"
    build.mkdir(exist_ok=True)
    source = build / f"{stem}_standalone.tex"
    source.write_text(
        "\n".join(
            [
                r"\documentclass[10pt]{article}",
                r"\usepackage[a4paper,landscape,margin=0.4in]{geometry}",
                r"\usepackage{booktabs,amsmath,amssymb,lmodern,graphicx}",
                r"\pagestyle{empty}",
                r"\begin{document}",
                rf"\setcounter{{table}}{{{number - 1}}}",
                rf"\input{{{stem}.tex}}",
                r"\end{document}",
            ]
        )
        + "\n"
    )
    logpath = build / f"{stem}_build.log"
    with logpath.open("w") as log:
        result = subprocess.run(
            [
                "latexmk",
                "-pdf",
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-outdir={build}",
                f"-jobname={stem}",
                str(source),
            ],
            cwd=directory,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError(f"LaTeX failed; see {logpath}")
    shutil.copyfile(build / f"{stem}.pdf", directory / f"{stem}.pdf")


def main(artifact):
    study, number = ARTIFACTS[artifact]
    parser = argparse.ArgumentParser(
        description=f"Render {artifact} from a complete, verified study."
    )
    parser.add_argument(
        "--input", required=True, type=Path, help="Study directory or manifest.json"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "dev_figures/production")
    if number:
        parser.add_argument("--tex-only", action="store_true")
    args = parser.parse_args()
    path = args.input / "manifest.json" if args.input.is_dir() else args.input
    try:
        manifest = read_study(path, study, timing=artifact == "swing_appendix_table")
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "dev_figures"):
        parser.error("Output must be inside the dev repository dev_figures/ directory")
    output.mkdir(parents=True, exist_ok=True)
    rows, config = manifest["records"], manifest["config"]
    if study == "swing":
        from . import swing_results as renderer

        renderer.configure(rows, output, config)
        if artifact == "swing_convergence":
            renderer.fig_convergence()
        elif artifact == "swing_adaptation":
            renderer.fig_adaptation()
        elif artifact == "swing_highlight_table":
            renderer.build_table(
                renderer.fmt_flops,
                "flops",
                r"FLOPs ($\times 10^{7}$)",
                renderer.HL_CAP.replace(
                    "N=1000", f"N={config['problem']['N']}"
                ).replace("5 seeds", f"{len(config['seeds'])} seeds"),
                "tab:swing_exp10_highlight",
                artifact,
            )
        else:
            renderer.build_table(
                renderer.fmt_pwall,
                "pwall",
                r"parallel wall $T_{\parallel}$ (s)",
                renderer.AP_CAP.replace("five seeds", f"{len(config['seeds'])} seeds"),
                "tab:swing_exp10_appendix",
                artifact,
                show_kkt=False,
            )
    elif artifact == "rate_ablation_table":
        from .rate_table import render

        render(rows, output, config)
    elif artifact == "rate_ablation":
        from .rate_results import render

        render(rows, output, config)
    else:
        from .eta_results import render

        render(rows, output, config)
    if number and not args.tex_only:
        compile_table(output, artifact, number)
    (output / f"{artifact}.provenance.json").write_text(
        json.dumps(
            dict(
                input=str(path.resolve()),
                config=config,
                source_sha256=manifest["source_sha256"],
                execution_mode=manifest["execution_mode"],
                study_sha256=manifest["study_sha256"],
                scenarios=len(rows),
                converged=sum(r["converged"] for r in rows),
            ),
            indent=2,
        )
        + "\n"
    )
