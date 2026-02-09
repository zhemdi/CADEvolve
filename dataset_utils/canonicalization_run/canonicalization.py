import argparse
from dataclasses import dataclass
from pathlib import Path
import yaml
from utils.canonicalization.standardizing import run_standardizing
from utils.canonicalization.centering import run_centering
from utils.canonicalization.scaling import run_scaling
from utils.canonicalization.binarization import run_binarization


@dataclass
class Cfg:
    root_dir: Path
    out_dir: Path
    logs_dir: Path
    flat: bool = False
    n_workers: int = 1911
    task_timeout: int = 1911
    steps: dict = None


def load_cfg(path: Path) -> Cfg:
    d = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Cfg(
        root_dir=Path(d["root_dir"]),
        out_dir=Path(d["out_dir"]),
        logs_dir=Path(d["logs_dir"]),
        flat=bool(d.get("flat", False)),
        n_workers=int(d.get("n_workers", 16)),
        task_timeout=int(d.get("task_timeout", 60)),
        steps=d.get("steps", {
            "standardize": True, "center": True, "scale": True, "binarize": True
        }),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    a = p.parse_args()

    cfg = load_cfg(a.config)

    standardized_dir = cfg.out_dir / "standardized"
    centered_dir     = cfg.out_dir / "centered"
    scaled_dir       = cfg.out_dir / "scaled"
    binarized_dir    = cfg.out_dir / "binarized"

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)

    # --- step 1: standardize ---
    if cfg.steps.get("standardize", True):
        standardized_dir.mkdir(parents=True, exist_ok=True)
        log_path = cfg.logs_dir / "standardizing.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        print("Standardizing in progress...")
        run_standardizing(
            root_dir=cfg.root_dir,
            logs_path=log_path,
            standardized_root=standardized_dir,
            n_workers=cfg.n_workers,
            task_timeout=cfg.task_timeout,
            flat=cfg.flat,
        )

    # --- step 2: center ---
    if cfg.steps.get("center", True):
        centered_dir.mkdir(parents=True, exist_ok=True)
        log_path = cfg.logs_dir / "centering.log"

        print("Centering in progress...")
        run_centering(
            centered_dir=centered_dir,
            standardized_dir=standardized_dir,
            logs_path=log_path,
            n_workers=cfg.n_workers,
            task_timeout=cfg.task_timeout,
            flat=cfg.flat,
        )

    # --- step 3: scale ---
    if cfg.steps.get("scale", True):
        scaled_dir.mkdir(parents=True, exist_ok=True)
        log_path = cfg.logs_dir / "scaling.log"

        print("Scaling in progress...")
        run_scaling(
            scaled_dir=scaled_dir,
            centered_dir=centered_dir,
            logs_path=log_path,
            standardized_dir=standardized_dir,
            n_workers=cfg.n_workers,
            task_timeout=cfg.task_timeout,
            flat=cfg.flat,
        )

    # --- step 4: binarize ---
    if cfg.steps.get("binarize", True):
        binarized_dir.mkdir(parents=True, exist_ok=True)
        log_path = cfg.logs_dir / "binarization.log"

        print("Binarization in progress...")
        run_binarization(
            scaled_dir=scaled_dir,
            binarized_dir=binarized_dir,
            logs_path=log_path,
            n_workers=cfg.n_workers,
            task_timeout=cfg.task_timeout,
            flat=cfg.flat,
        )


if __name__ == "__main__":
    main()
