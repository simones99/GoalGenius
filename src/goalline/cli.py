"""Command line: download, validate, develop, final, report."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from goalline.constants import ODDS_COLUMNS
from goalline.data.load import load_matches, prepare, read_raw
from goalline.data.manifest import ChecksumError, download, read_manifest, verify
from goalline.evaluation.develop import develop, final_predictions
from goalline.evaluation.outputs import home_advantage, write_outputs
from goalline.evaluation.selection import read_selected, write_selected
from goalline.evaluation.walkforward import evaluation_rows
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


def selected_commit(config: Path) -> str:
    try:
        done = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", Path(config).name],
            cwd=Path(config).parent, capture_output=True, text=True, check=False,
        )
    except FileNotFoundError:
        return "uncommitted"
    sha = done.stdout.strip()
    return sha if done.returncode == 0 and sha else "uncommitted"


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
    selected = read_selected(paths.config)
    sha = _verified(paths)
    if sha != selected.data_sha256:
        raise ChecksumError(
            f"the data file ({sha}) is not the one used in develop ({selected.data_sha256}); "
            "a new data file needs a new develop run"
        )
    frame, _ = load_matches(paths.data, settings, mode="final")
    preds = final_predictions(frame, selected, settings)
    write_outputs(paths.output / "final", preds, frame, settings)
    home_advantage(frame).to_csv(paths.output / "home_advantage.csv", index=False)
    scored = evaluation_rows(frame, settings.final_seasons)
    max_odds = scored[list(ODDS_COLUMNS["max"])].dropna()
    meta = {
        "selected_commit": selected_commit(paths.config),
        "data_sha256": sha,
        "generated": dt.date.today().isoformat(),
        "seasons": list(settings.final_seasons),
        "n_matches": int(len(scored)),
        "max_odds_arbitrage": int(((1 / max_odds).sum(axis=1) < 1).sum()),
    }
    (paths.output / "final" / "final_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(pd.read_csv(paths.output / "final" / "metrics.csv").query("group_type == 'overall'"))


COMMANDS = {
    "download": lambda paths, settings: run_download(paths),
    "validate": run_validate,
    "develop": run_develop,
    "final": run_final,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="goalline", description=__doc__)
    parser.add_argument("command", choices=sorted(COMMANDS))
    args = parser.parse_args(argv)
    COMMANDS[args.command](Paths.default(), Settings())
    return 0
