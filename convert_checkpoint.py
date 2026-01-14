#!/usr/bin/env python3
"""Convert old checkpoints (full state_dict) to new format (projection only)."""

import argparse
import torch


def convert_checkpoint(input_path, output_path=None):
    """Convert old checkpoint to new format."""
    if output_path is None:
        output_path = input_path.replace(".pt", "_converted.pt")

    print(f"Loading {input_path}...")
    state = torch.load(input_path, map_location="cpu", weights_only=True)

    # Check if already in new format (keys don't have "proj." prefix)
    if "weight" in state:
        print("Checkpoint is already in new format (projection only)")
        return

    # Extract projection layer weights
    proj_state = {}
    for k, v in state.items():
        if k.startswith("proj."):
            new_key = k.replace("proj.", "")
            proj_state[new_key] = v

    if not proj_state:
        print("Error: No projection weights found in checkpoint")
        print(f"Keys found: {list(state.keys())[:10]}...")
        return

    old_size = sum(v.numel() * v.element_size() for v in state.values())
    new_size = sum(v.numel() * v.element_size() for v in proj_state.values())

    print(f"Extracted projection weights: {list(proj_state.keys())}")
    print(f"Size reduction: {old_size / 1e9:.2f}GB -> {new_size / 1e6:.2f}MB")

    torch.save(proj_state, output_path)
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert old checkpoints to new format")
    parser.add_argument("input", type=str, help="Input checkpoint path")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output path (default: input_converted.pt)")
    args = parser.parse_args()

    convert_checkpoint(args.input, args.output)


if __name__ == "__main__":
    main()
