#!/usr/bin/env python3
# coding: utf-8
"""
Inference for Qwen2-VL fine-tuned on CadQuery STL renders.

- Takes a folder with .stl files.
- Renders each STL with the same Plotter used in training (visualization_iso).
- Feeds the render via Qwen chat template and extracts the assistant's code.
- Writes one output per STL (either raw predicted code, or wrapped as runnable DSL).

Usage:
  python inference_cadevolve.py \
      --stl_dir ./path/to/stls \
      --out_dir ./preds \
      --model_path ./work_dirs/cadevolve/final_model \
"""

import os, sys, json, argparse, warnings
from pathlib import Path
from typing import List, Tuple
import numpy as np

warnings.filterwarnings("ignore")



from visualization_norm import Plotter # noqa: E402

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration  # noqa: E402
from qwen_vl_utils import process_vision_info  # noqa: E402


def save_img_any(img, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(img, Image.Image):
        img.convert("RGB").save(path)
    elif isinstance(img, np.ndarray):
        # HxWxC uint8 expected
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        Image.fromarray(img).save(path)
    elif torch.is_tensor(img):
        t = img.detach().cpu()
        if t.ndim == 3 and t.shape[0] in (1,3,4):  # CxHxW -> HxWxC
            t = t.permute(1,2,0)
        t = t.clamp(0,255).to(torch.uint8).numpy()
        Image.fromarray(t).save(path)
    else:
        Image.fromarray(np.asarray(img)).save(path)


def list_stls(stl_root: Path) -> List[Path]:
    return sorted([p for p in stl_root.rglob("*.stl") if p.is_file() and p.stat().st_size > 0])


def build_inputs_for_images(imgs: List[Image.Image], processor: AutoProcessor):
    """
    Mimics the training collate: a single 'user' turn with one image, generation prompt enabled.
    """
    messages = [[{"role": "user", "content": [{"type": "image", "image": img}]}] for img in imgs]
    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
    vis_imgs, vis_vids = process_vision_info(messages)
    inputs = processor(text=texts, images=vis_imgs, videos=vis_vids, padding=True, return_tensors="pt")
    return inputs


def extract_assistant_text(decoded: str) -> str:
    """
    Extract the assistant content between "<|im_start|>assistant" and "<|im_end|>".
    Robust to extra prefixes.
    """
    start_tag = "<|im_start|>assistant"
    end_tag = "<|im_end|>"
    if start_tag in decoded:
        part = decoded.split(start_tag, maxsplit=1)[1]
        if part.startswith("\n"):
            part = part[1:]
        if end_tag in part:
            part = part.split(end_tag, maxsplit=1)[0]
        return part.strip()
    return decoded.strip()


def wrap_dsl(pred_body: str) -> str:
    """
    Rebuild a runnable .py from the 'cleaned' DSL body used during training.
    Adjust to your project needs.
    """
    header = "from api.dsl_api import *\n\n"
    footer = "\n\nresult = n0.build()\n"
    return header + pred_body.rstrip() + footer


def main():
    parser = argparse.ArgumentParser(description="Inference for CADEvolve on STL renders")
    parser.add_argument("--stl_dir", type=Path, required=True, help="Folder with .stl files (recursively searched)")
    parser.add_argument("--out_dir", type=Path, required=True, help="Where to write predictions")
    parser.add_argument("--model_path", type=str, default="kulibinai/cadevolve-rl1")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=4000)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--apply_augs", action="store_true", help="Use Plotter augmentations (default: off for eval)")
    parser.add_argument("--wrap_runnable", action="store_true", help="Wrap predicted body into runnable DSL .py")
    parser.add_argument("--save_jsonl", action="store_true", help="Also save a preds.jsonl manifest")
    parser.add_argument("--device", type=str, default=None, help="cuda|cpu (auto if not set)")
    parser.add_argument("--resized_width", type=int, default=14 * 17 * 2)
    parser.add_argument("--resized_height", type=int, default=14 * 17 * 4)

    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    stls = list_stls(args.stl_dir)
    if not stls:
        print(f"No non-empty .stl files found under: {args.stl_dir}")
        return

    # Device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        resized_width=args.resized_width,
        resized_height=args.resized_height,
        padding_side="left",
    )
    dtype = torch.bfloat16 if (device.startswith("cuda") and torch.cuda.is_bf16_supported()) else torch.float16 if device.startswith("cuda") else torch.float32
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2" if device.startswith("cuda") else "eager",
        trust_remote_code=True,
    )
    model.eval().to(device)

    eos_token_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    pad_token_id = processor.tokenizer.eos_token_id

    plotter = Plotter()

    manifest = []
    batch_imgs, batch_paths = [], []

    def flush_batch():
        nonlocal batch_imgs, batch_paths, manifest
        if not batch_imgs:
            return

        with torch.no_grad():
            inputs = build_inputs_for_images(batch_imgs, processor)
            for k, v in list(inputs.items()):
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)

            gen_kwargs = dict(
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                num_beams=args.num_beams,
                repetition_penalty=args.repetition_penalty,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
            )
            generated = model.generate(**inputs, **gen_kwargs)

            for i, out_ids in enumerate(generated):
                decoded = processor.tokenizer.decode(out_ids, skip_special_tokens=False)
                pred_body = extract_assistant_text(decoded)

                stl_path = batch_paths[i]
                stem = stl_path.stem
                if args.wrap_runnable:
                    text_to_save = wrap_dsl(pred_body)
                    out_ext = ".dsl.py"
                else:
                    text_to_save = pred_body
                    out_ext = ".txt"

                rel_dir = stl_path.parent.relative_to(args.stl_dir) if stl_path.parent != args.stl_dir else Path(".")
                save_dir = args.out_dir / rel_dir
                save_dir.mkdir(parents=True, exist_ok=True)
                out_path = save_dir / f"{stem}{out_ext}"
                out_path.write_text(text_to_save, encoding="utf-8")

                manifest.append({
                    "stl": str(stl_path),
                    "pred_path": str(out_path),
                    "wrapped": bool(args.wrap_runnable),
                    "tokens": int(out_ids.shape[-1]),
                })

        batch_imgs, batch_paths = [], []

    # print(len(stls))

    for stl in stls:
        try:
            img = plotter.get_img(stl, None, apply_augs=args.apply_augs)
            save_img_any(img, (args.out_dir / "renders" / stl.parent.relative_to(args.stl_dir) / f"{stl.stem}.png"))
            batch_imgs.append(img)
            batch_paths.append(stl)
        except Exception as e:
            print(f"[WARN] render failed for {stl}: {e}")
            try:
                plotter.reload()
            except Exception:
                pass
            continue

        if len(batch_imgs) >= args.batch_size:
            flush_batch()

    flush_batch()

    if args.save_jsonl:
        jsonl_path = args.out_dir / "preds.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for row in manifest:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Saved manifest: {jsonl_path}")

    print(f"Done. Processed {len(manifest)} / {len(stls)} STLs. Outputs -> {args.out_dir}")


if __name__ == "__main__":
    main()