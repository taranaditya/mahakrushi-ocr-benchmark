from pathlib import Path

import typer

from .adapters.base import AdapterRegistry
from .dataset import load_manifest, validate_dataset
from .metrics import score_prediction
from .reporting import write_report
from .runner import run_benchmark

app = typer.Typer(help="MahaKrushi OCR benchmark")
@app.command("validate-data")
def validate_data(manifest: Path = typer.Option(Path("data/manifest.jsonl"))):
    records=load_manifest(manifest, manifest.parents[1]); validate_dataset(records); typer.echo(f"Validated {len(records)} pages")
@app.command()
def run(manifest: Path = Path("data/manifest.jsonl"), results: Path = Path("results/latest"), resume: bool = False):
    records=load_manifest(manifest,manifest.parents[1]); adapters=list(AdapterRegistry.default().adapters.values()); predictions=run_benchmark(records,adapters,results,resume); write_report(predictions,[score_prediction(r,p) for p in predictions for r in records if r.id==p.document_id],results); typer.echo(f"Saved {len(predictions)} predictions to {results}")
@app.command()
def report(results: Path = Path("results/latest")):
    typer.echo(f"Open {results / 'summary.json'} in the dashboard")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
