import argparse
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class Cfg:
    code_db: Path
    log_path: Path
    save_dir: Path
    num_samples: int
    n_workers: int
    task_timeout: int

    start_method: str = "forkserver"
    only_missing: bool = True


def load_cfg(path: Path) -> Cfg:
    d = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Cfg(
        code_db=Path(d["code_db"]),
        log_path=Path(d["log_path"]),
        save_dir=Path(d["save_dir"]),
        num_samples=int(d["num_samples"]),
        n_workers=int(d["n_workers"]),
        task_timeout=int(d["task_timeout"]),
        start_method=str(d.get("start_method", "forkserver")),
        only_missing=bool(d.get("only_missing", True)),
    )


def run_sampling(cfg: Cfg):
    from utils.canonicalization.samples import (
        load_code_db,
        NonDaemonPool,
        init_worker,
        timed_process_part,
        append_txt_line,
    )

    cfg.log_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.save_dir.mkdir(parents=True, exist_ok=True)

    db = load_code_db(str(cfg.code_db))
    print("All parts:", len(db))

    width = max(2, len(str(cfg.num_samples)))
    require_file = f"{cfg.num_samples:0{width}d}.py"

    if cfg.only_missing:
        already_generated_names = {
            p.name
            for p in cfg.save_dir.iterdir()
            if p.is_dir() and (p / require_file).is_file()
        }
        print("Generated parts:", len(already_generated_names))
        parts = [item for item in db if item["name"] not in already_generated_names]
    else:
        parts = list(db)

    import utils.canonicalization.samples as M
    M.LOG_PATH = cfg.log_path
    M.SAVE_DIR = cfg.save_dir
    M.N_WORKERS = cfg.n_workers
    M.TASK_TIMEOUT = cfg.task_timeout
    M.CODE_DB = str(cfg.code_db)

    from multiprocessing import get_context
    ctx = get_context(cfg.start_method)

    pool = NonDaemonPool(
        processes=cfg.n_workers,
        initializer=init_worker,
        context=ctx,
    )
    print("POOL Initialized", flush=True)

    import atexit
    atexit.register(lambda: (pool.close(), pool.join()))

    async_results = [
        (part["name"], pool.apply_async(timed_process_part, args=(part, cfg.num_samples)))
        for part in parts
    ]

    for name, ar in async_results:
        try:
            result = ar.get()
        except Exception as e:
            append_txt_line(M.LOG_PATH, f"{name}: TASK CRASHED (pool get): {e}")
            continue

        if result == "__TIMEOUT__":
            append_txt_line(M.LOG_PATH, f"{name}: TASK TIMEOUT ({M.TASK_TIMEOUT}s)")
            continue
        if result == "__CRASH__":
            append_txt_line(M.LOG_PATH, f"{name}: TASK CRASHED")
            continue
        if result is not None:
            append_txt_line(M.LOG_PATH, result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args()

    cfg = load_cfg(args.config)
    run_sampling(cfg)


if __name__ == "__main__":
    main()
