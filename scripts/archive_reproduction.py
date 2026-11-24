"""Archive private experiment inputs and reviews, excluding model weights."""

import hashlib
import json
import tarfile
from pathlib import Path

import click

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = (
    "data/training_data",
    "data/private/gemine_generate",
    "data/private/supervised_fine_tuning_galsenai.json",
    "results/local/assistant-eval-GI297S",
    "results/local/google-vs-qat-qDz5zc",
    "results/local/v3-v4-Ngmbrd",
    "scripts",
    "src",
    "tests",
    "README.md",
    "data/prepare_training_data.py",
    "data/prepare_assistant_data.py",
    "data/prepare_review_data.py",
    "config",
    "pyproject.toml",
    "uv.lock",
)


@click.command()
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    required=True,
    help="New private directory, preferably under ignored data/backups/.",
)
@click.option("--include", "inputs", multiple=True, help="Repository-relative input paths; overrides defaults.")
def main(output, inputs):
    if output.exists():
        raise click.ClickException(f"Output already exists: {output}")
    files = {}
    for name in inputs or DEFAULT_INPUTS:
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.exists():
            raise click.ClickException(f"Missing or external input: {name}")
        for file in sorted(path.rglob("*") if path.is_dir() else [path]):
            if file.is_symlink():
                raise click.ClickException(f"Symlink not archived: {file}")
            if file.is_file() and "__pycache__" not in file.parts:
                if file.suffix in {".safetensors", ".pt", ".pth", ".bin", ".gguf"}:
                    continue
                if file.resolve().is_relative_to(output.resolve()):
                    raise click.ClickException("Output must be outside archived inputs")
                files[file.relative_to(ROOT).as_posix()] = hashlib.sha256(file.read_bytes()).hexdigest()
    output.mkdir(parents=True)
    archive = output / "inputs.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for name in sorted(files):
            bundle.add(ROOT / name, arcname=name, recursive=False)
    manifest = dict(
        files_sha256=files,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        weights="Excluded; adapters will be published separately on Hugging Face.",
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    click.echo(f"Archived {len(files)} files in {archive}. Keep a durable private copy off this machine.")


if __name__ == "__main__":
    main()
