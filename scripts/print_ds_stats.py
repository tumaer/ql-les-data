"""Show dataset info."""

import argparse
from pathlib import Path

import h5py

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("dataset_kolm/datasets/2D_KOLM_4096_140kevery1"),
    )
    args = parser.parse_args()

    for split in ["train", "valid", "test"]:
        with h5py.File(args.src / f"{split}.h5", "r") as f:
            keys = list(f.keys())
            print(f"{split} has {len(f)} trajs: [{keys[0]},...{keys[-1]}], each with:")
            traj_0 = f[keys[0]]
            for k, v in traj_0.items():
                print(f"{' ' * 5} {k}.shape = {v.shape}")
