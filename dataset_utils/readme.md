## Folder structure overview

- `run.sh` — runs the full pipeline end-to-end.
- `canonicalization_run/` — sampling + canonicalization entrypoints and configs.
- `rotation_run/` — rotation augmentation entrypoints and configs.
- `utils/` — shared pipeline utilities (canonicalization + rotation helpers).
- `data/` — input databases (e.g., `database_toy_example.json`).

## Minimal run

### 1) Set the input database

Edit `./canonicalization_run/cfg_sampling.yaml` and set `code_db` to the database of parts you want to process.

A toy example database is provided at `./data/database_toy_example.json`.

### 2) Run the pipeline

From the project root:

    chmod +x ./run.sh
    ./run.sh

You can also run it from anywhere:

    bash /path/to/CADEvolve/run.sh

## Outputs

Running `./run.sh` creates a new folder `./results/` in the project root (this folder is generated locally and is not tracked in the repository).

Usable results are stored in:
- `results/rotated/`
- `results/rotated_stl/`

Important: the rotation stage expects a flat folder at `results/canonicalized_flat/`.

## Notes on config paths

- `run.sh` executes stages with the working directory set to `./results/`, so paths inside YAML configs are resolved relative to `./results/`.
- Inputs stored in `./data/...` are typically referenced as `../data/...` inside YAML.
- Outputs should be written as relative paths like `sampled/...`, `canonicalized/...` so they land inside `./results/`.

## Output folder structure overview

The following folders are created after running the pipeline (not included in the repository by default):

- `results/` — all outputs are written here.
  - `results/sampled/` — sampling outputs (generated scripts + logs, depending on config).
  - `results/canonicalized/` — canonicalization stage outputs.
    - `results/canonicalized/binarized/` — final canonicalized scripts (nested structure).
  - `results/canonicalized_flat/` — **flat** scripts required by rotation stage (no subfolders).
  - `results/rotated/` — rotated scripts (main usable output).
  - `results/rotated_stl/` — STL exports for rotated scripts (main usable output).
  - `results/logs/` — pipeline logs.
