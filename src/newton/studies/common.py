"""Shared execution options, manifest storage, and scenario logging."""

from contextlib import contextmanager
import gzip
import json
import logging
import os
from pathlib import Path
import sys


def add_execution_arguments(parser):
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--uncontended",
        action="store_true",
        help="Run scenarios sequentially with one BLAS thread; eligible for timing tables.",
    )
    modes.add_argument(
        "--parallel",
        type=int,
        metavar="X",
        help="Run at most X scenario processes, each with one BLAS thread.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse verified records matching configuration, source, environment, and mode.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and list scenarios without solving.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Run only the first n scenarios; keep the manifest partial.",
    )


def worker_count(parser, args):
    if args.parallel is not None and args.parallel < 1:
        parser.error("--parallel must be a positive processor count")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return args.parallel if args.parallel is not None else 1


def manifest_path(path):
    path = Path(path)
    if path.is_dir():
        plain = path / "manifest.json"
        return plain if plain.exists() else path / "manifest.json.gz"
    return path


def read_manifest(path):
    path = manifest_path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def portable_path(path, root):
    path = Path(path).resolve()
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, default=float, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def progress_logger(output):
    logger = logging.getLogger(f"study.{Path(output).resolve()}")
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
    """Capture Python and native solver output in one scenario log."""
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
