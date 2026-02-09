#!/usr/bin/env python3
import sys
import random
import warnings
from pathlib import Path
from functools import partial
import argparse
import yaml
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from transformers import (
    AutoProcessor, Qwen2VLForConditionalGeneration,
    Trainer, TrainingArguments,
)
from qwen_vl_utils import process_vision_info

from visualization import Plotter

# -----------------------------------------------------------------------------
# config utils
# -----------------------------------------------------------------------------
def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def as_abs_path(p: str, *, strict: bool = False) -> Path:
    """
    Convert to absolute Path and enforce it is absolute in the config.
    - strict=False: path may not exist yet (useful for output/log paths)
    - strict=True: path must exist (useful for inputs/checkpoints)
    """
    if not isinstance(p, str) or not p:
        raise ValueError(f"Path must be a non-empty string, got: {p!r}")

    raw = Path(p).expanduser()
    if not raw.is_absolute():
        raise ValueError(f"Path must be ABSOLUTE, got: {p!r}")

    return raw.resolve(strict=strict)

# -----------------------------------------------------------------------------
# logging
# -----------------------------------------------------------------------------
def setup_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = open(log_path, "w", encoding="utf-8", buffering=1)
    sys.stderr = sys.stdout

warnings.filterwarnings("ignore", category=UserWarning, module="trimesh")


class STLImagesDataset(Dataset):
    
    def __init__(self,
                 items_pkl: Path,
                 max_script_len=None,
                 apply_augs=False,
                 language='cadevolve'):

        super().__init__()
        self.base_seed = 123456
        self.plotter = None
        self.max_script_len = max_script_len
        self.apply_augs = apply_augs
        self.language = (language or "").lower()  # 'cadevolve' | 'dsl' | ''

        with open(items_pkl, "rb") as f:
            self.items = pickle.load(f)   # [(py_path_str, stl_path_str), ...]



        random.shuffle(self.items)

    def __len__(self):
        return len(self.items)

    def __clean_code__(self, code: str):
        return '\n'.join(code.split('\n')[2:-3])
        
    def __getitem__(self, idx):
        if self.plotter is None:
            self.plotter = Plotter()

        for _ in range(10):
            py_path_str, stl_path_str = self.items[idx]
            py_path = Path(py_path_str)
            stl_path = Path(stl_path_str)

            code = py_path.read_text(encoding="utf-8", errors="ignore")
            # print(len(code), py_path_str)
            # if len(code) > 3000:
            #     idx = random.randrange(len(self.items))
            #     continue
            if self.language == 'dsl':
                code = self.__clean_code__(code)

            try:
                image = self.plotter.get_img(stl_path, None, apply_augs=self.apply_augs)
                return {"image": image, "answer": code}
            except Exception as e:
                print(f"Error in visualization for {stl_path}: {e}")
                self.plotter.reload()
                idx = random.randrange(len(self.items))

        raise RuntimeError("Too many consecutive render failures")

# -----------------------------------------------------------------------------
# training logic – everything else is identical
# -----------------------------------------------------------------------------
def find_assistant_spans(tokenizer, ids):
    """
    Return (start_im_start, end_im_end_inclusive) spans for each assistant turn.
    Includes the leading newline after 'assistant' (if present) and the <|im_end|>.
    """
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end   = tokenizer.convert_tokens_to_ids("<|im_end|>")
    asst_id  = tokenizer.convert_tokens_to_ids("assistant")  # Qwen uses literal 'assistant'

    spans = []
    i = 0
    n = len(ids)
    while i < n - 2:
        if ids[i] == im_start and ids[i+1] == asst_id:
            j = i + 2
            while j < n and ids[j] != im_end:
                j += 1
            if j < n:
                spans.append((i, j))
                i = j + 1
                continue
        i += 1
    return spans

def collate_fn_for_sft(batch, processor):
    batch = [b for b in batch if b is not None]
    if not batch:
        raise ValueError("Empty batch after filtering; dataset must yield valid samples.")

    messages = [[
        {"role": "user", "content": [{"type": "image", "image": b["image"]}]},
        {"role": "assistant", "content": [{"type": "text", "text": b["answer"]}]}
    ] for b in batch]

    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
             for m in messages]
    imgs, vids = process_vision_info(messages)

    inputs = processor(text=texts, images=imgs, videos=vids,
                       padding=True, return_tensors="pt")

    tok = processor.tokenizer
    labels = []
    for ids in inputs["input_ids"].tolist():
        mask = [-100] * len(ids)
        for s, e in find_assistant_spans(tok, ids):
            mask[s+2:e+1] = ids[s+2:e+1]
        labels.append(mask)
    inputs["labels"] = torch.tensor(labels)
    return inputs

def run_training(cfg: dict):
    script_path = Path(__file__).resolve().parent
    # ----- logging -----
    # log_path = as_abs_path(cfg["logging"]["log_path"])
    log_path = script_path / cfg["logging"]["log_path"]
    setup_logging(log_path)

    # ----- paths & constants -----
    # items_pkl = as_abs_path(cfg["data"]["items_pkl"], strict=True)
    items_pkl = Path(cfg["data"]["items_pkl"])
    model_id = cfg["model"]["model_id"]
    # processor_id = cfg["model"]["processor_id"]

    # output_dir = as_abs_path(cfg["paths"]["output_dir"])
    output_dir = Path(cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----- seeds -----
    seed = int(cfg["run"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ----- processor/model -----
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=bool(cfg["processor"]["trust_remote_code"]),
        resized_width=int(cfg["processor"]["resized_width"]),
        resized_height=int(cfg["processor"]["resized_height"]),
        padding_side=str(cfg["processor"]["padding_side"]),
    )


    dtype_str = str(cfg["model"]["torch_dtype"]).lower()
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported torch_dtype: {dtype_str}. Use one of: {list(dtype_map)}")

    torch_dtype = dtype_map[dtype_str]

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        attn_implementation=str(cfg["model"]["attn_implementation"]),
        trust_remote_code=bool(cfg["model"]["trust_remote_code"]),
    )

    # ----- dataset -----
    ds_cfg = cfg["dataset"]

    train_full = STLImagesDataset(
        items_pkl=items_pkl,
        # py_root=cfg["data"]["py_root"],
        # stl_root=cfg["data"]["stl_root"],
        max_script_len=ds_cfg.get("max_script_len", None),
        apply_augs=bool(ds_cfg.get("apply_augs", False)),
        language=str(ds_cfg.get("language", "cadevolve")),
    )

    val_size = int(cfg["data"]["val_size"])

    idx = random.sample(range(len(train_full)), len(train_full))
    val_ds = Subset(train_full, idx[-val_size:])
    train_ds = Subset(train_full, idx[:-val_size])

    print("TRAIN_DS SIZE", len(train_ds))

    # ----- training args -----
    t = cfg["training"]
    targs = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(t["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(t["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(t["gradient_accumulation_steps"]),
        dataloader_num_workers=int(t["dataloader_num_workers"]),
        learning_rate=float(t["learning_rate"]),
        weight_decay=float(t["weight_decay"]),
        lr_scheduler_type=str(t["lr_scheduler_type"]),
        num_train_epochs=float(t["num_train_epochs"]),
        warmup_steps=int(t["warmup_steps"]),
        logging_strategy=str(t["logging_strategy"]),
        logging_steps=int(t["logging_steps"]),
        save_strategy=str(t["save_strategy"]),
        save_steps=int(t["save_steps"]),
        save_total_limit=int(t["save_total_limit"]),
        eval_strategy=str(t["eval_strategy"]),
        eval_steps=int(t["eval_steps"]),
        load_best_model_at_end=bool(t["load_best_model_at_end"]),
        bf16=bool(t["bf16"]),
        dataloader_drop_last=bool(t["dataloader_drop_last"]),
        remove_unused_columns=bool(t["remove_unused_columns"]),
        report_to=str(t["report_to"]),
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=partial(collate_fn_for_sft, processor=processor),
        tokenizer=processor,
    )

    trainer.train(resume_from_checkpoint=bool(cfg["run"]["resume_from_checkpoint"]))
    trainer.save_model(str(output_dir / "final_model"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run_training(cfg)


