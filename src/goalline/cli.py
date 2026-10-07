"""Command line: download, validate, develop, final, report."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from goalline.constants import ODDS_COLUMNS
from goalline.data.load import load_matches, prepare, read_raw
from goalline.data.manifest import ChecksumError, download, read_manifest, verify
from goalline.evaluation.develop import develop, final_run
from goalline.evaluation.outputs import home_advantage, write_outputs
from goalline.evaluation.selection import read_selected, write_selected
from goalline.evaluation.walkforward import evaluation_rows
from goalline.report.render import render
from goalline.settings import Settings


@dataclass(frozen=True)
class Paths:
    manifest: Path
    data: Path
    config: Path
    output: Path
    site: Path

    @classmethod
    def under(cls, root: Path) -> Paths:
        root = Path(root)
        return cls(
            manifest=root / "data" / "manifest.json",
            data=root / "data" / "raw" / "Matches.csv",
            config=root / "config" / "selected.json",
            output=root / "output",
            site=root / "site",
        )

    @classmethod
    def default(cls) -> Paths:
        return cls.under(Path.cwd())


class UncommittedSelectionError(RuntimeError):
    pass


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
        )
    except (FileNotFoundError, NotADirectoryError):
        return None


def _in_git(folder: Path) -> bool:
    done = _git(folder, "rev-parse", "--is-inside-work-tree")
    return done is not None and done.returncode == 0 and done.stdout.strip() == "true"


def selection_is_uncommitted(config: Path) -> bool:
    """True when the file is in a git work tree and is untracked or differs from HEAD."""
    config = Path(config)
    if not _in_git(config.parent):
        return False
    tracked = _git(config.parent, "ls-files", "--error-unmatch", "--", config.name)
    if tracked is None or tracked.returncode != 0:
        return True
    done = _git(config.parent, "status", "--porcelain", "--", config.name)
    return done is None or done.returncode != 0 or bool(done.stdout.strip())


def selected_commit(config: Path) -> str:
    """Last commit that changed the file, or "uncommitted" if it is not in git or has edits."""
    config = Path(config)
    if not _in_git(config.parent) or selection_is_uncommitted(config):
        return "uncommitted"
    done = _git(config.parent, "log", "-1", "--format=%H", "--", config.name)
    sha = done.stdout.strip() if done is not None and done.returncode == 0 else ""
    return sha or "uncommitted"


def code_state() -> tuple[str, bool]:
    """HEAD of the repository holding this code, and whether tracked files are modified."""
    folder = Path(__file__).resolve().parent
    if not _in_git(folder):
        return "uncommitted", True
    head = _git(folder, "rev-parse", "HEAD")
    status = _git(folder, "status", "--porcelain", "--untracked-files=no")
    if head is None or head.returncode != 0 or status is None or status.returncode != 0:
        return "uncommitted", True
    return head.stdout.strip(), bool(status.stdout.strip())


def environment_versions() -> dict[str, str]:
    packages = ("numpy", "pandas", "scipy", "scikit-learn", "xgboost")
    return {"python": platform.python_version(), **{p: version(p) for p in packages}}


def _verified(paths: Paths) -> str:
    manifest = read_manifest(paths.manifest)
    verify(paths.data, manifest)
    return manifest.sha256


def run_download(paths: Paths) -> None:
    manifest = read_manifest(paths.manifest)
    if paths.data.exists():
        try:
            verify(paths.data, manifest)
            print(f"{paths.data} already matches the manifest")
            return
        except ChecksumError:
            paths.data.unlink()
    download(manifest, paths.data)
    print(f"downloaded {paths.data}")


def run_validate(paths: Paths, settings: Settings) -> None:
    _verified(paths)
    _, report = prepare(read_raw(paths.data))
    paths.output.mkdir(parents=True, exist_ok=True)
    report.to_csv(paths.output / "data_quality.csv", index=False)
    print(f"{int(report['count'].sum()) if not report.empty else 0} rows dropped or odds blanked")


def run_develop(paths: Paths, settings: Settings) -> None:
    sha = _verified(paths)
    frame, _ = load_matches(paths.data, settings, mode="develop")
    result = develop(frame, settings, data_sha256=sha)
    write_selected(paths.config, result.selected)
    tables = write_outputs(paths.output / "dev", result.predictions, frame, settings)
    result.grids.to_csv(paths.output / "dev" / "grid_scores.csv", index=False)
    print(f"selected hyperparameters written to {paths.config}")
    print(tables["metrics"].query("group_type == 'overall'").to_string(index=False))


def run_final(paths: Paths, settings: Settings) -> None:
    if selection_is_uncommitted(paths.config):
        raise UncommittedSelectionError(
            f"{paths.config.name} has uncommitted changes: commit it (and tag it) before "
            "`goalline final`, so the published numbers name the commit that froze them"
        )
    selected = read_selected(paths.config)
    sha = _verified(paths)
    if sha != selected.data_sha256:
        raise ChecksumError(
            f"the data file ({sha}) is not the one used in develop ({selected.data_sha256}); "
            "a new data file needs a new develop run"
        )
    frame, _ = load_matches(paths.data, settings, mode="final")
    run = final_run(frame, selected, settings)
    write_outputs(paths.output / "final", run.predictions, frame, settings)
    home_advantage(frame).to_csv(paths.output / "home_advantage.csv", index=False)
    scored = evaluation_rows(frame, settings.final_seasons)
    max_odds = scored[list(ODDS_COLUMNS["max"])].dropna()
    code_commit, code_dirty = code_state()
    meta = {
        "selected_commit": selected_commit(paths.config),
        "code_commit": code_commit,
        "code_dirty": code_dirty,
        "environment": environment_versions(),
        "dixon_coles_fits": run.dixon_coles_fits,
        "dixon_coles_nonconverged": run.dixon_coles_nonconverged,
        "data_sha256": sha,
        "generated": dt.date.today().isoformat(),
        "seasons": list(settings.final_seasons),
        "n_matches": int(len(scored)),
        "max_odds_arbitrage": int(((1 / max_odds).sum(axis=1) < 1).sum()),
    }
    (paths.output / "final" / "final_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(pd.read_csv(paths.output / "final" / "metrics.csv").query("group_type == 'overall'"))


def run_report(paths: Paths, settings: Settings) -> None:
    path = render(paths.output, paths.site, paths.config)
    print(f"report written to {path}")


COMMANDS = {
    "download": lambda paths, settings: run_download(paths),
    "validate": run_validate,
    "develop": run_develop,
    "final": run_final,
    "report": run_report,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="goalline", description=__doc__)
    parser.add_argument("command", choices=sorted(COMMANDS))
    args = parser.parse_args(argv)
    try:
        COMMANDS[args.command](Paths.default(), Settings())
    except UncommittedSelectionError as error:
        print(f"goalline: {error}", file=sys.stderr)
        return 1
    return 0
