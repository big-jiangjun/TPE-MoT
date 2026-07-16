#!/usr/bin/env python3
"""Extract the non-EMA model state from a TPE-MoT DeepSpeed checkpoint."""

import argparse
from pathlib import Path

import torch


def resolve_model_state(path: Path) -> Path:
    if path.is_file():
        return path
    tags = sorted(path.glob("global_step*"))
    if not tags:
        raise FileNotFoundError("No global_step* directory under {}".format(path))
    candidates = sorted(tags[-1].glob("mp_rank_*_model_states.pt"))
    if not candidates:
        raise FileNotFoundError("No model state file under {}".format(tags[-1]))
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="DeepSpeed iter directory or checkpoint file")
    parser.add_argument("output", type=Path, help="Output .pth state dict")
    args = parser.parse_args()

    source = resolve_model_state(args.input)
    checkpoint = torch.load(str(source), map_location="cpu")
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("module"), dict):
        state_dict = checkpoint["module"]
    elif isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise TypeError("Unsupported checkpoint payload: {}".format(type(checkpoint).__name__))

    state_dict = {key: value for key, value in state_dict.items() if not key.startswith("ema_")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state_dict, "meta": {"source": str(source)}}, str(args.output))
    print("Saved {} tensors to {}".format(len(state_dict), args.output))


if __name__ == "__main__":
    main()
