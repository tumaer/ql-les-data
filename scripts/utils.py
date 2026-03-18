from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def _traj_nr_from_dirname(dir):
    """e.g.: "traj_5"""
    return int(dir.stem.split("_")[1])


def _step_from_filename(file):
    """e.g.: "step_00000.h5" or "u_64_00000.npy"""
    return int(file.stem.split("_")[-1])


def _ls_sorted_trajs(trajs_root):
    traj_dirs = [d for d in trajs_root.glob("traj_*") if d.is_dir()]
    traj_dirs = sorted(traj_dirs, key=_traj_nr_from_dirname)
    if not traj_dirs:
        raise FileNotFoundError(f"No traj_* directories found in {trajs_root}.")
    return traj_dirs


def _ls_sorted_frames(traj_dir, mode="com", match="step_*.h5"):
    frame_files = (traj_dir / mode).glob(match)
    frame_files = sorted(frame_files, key=_step_from_filename)
    if not frame_files:
        raise FileNotFoundError(f"No {match} frames found in {traj_dir / mode}.")
    return frame_files


def load_trajectories_com(ds_root: Path, every_nth_frame=20, max_trajs=8) -> dict:
    """Load 3D rollout trajectories from h5 checkpoints.

    Returns dict with keys:
        - u: (S, T, Np, dim)
        - r: (S, T, Np, dim)
        - t: (T,)
        - steps: (T,)
    """
    traj_dirs = _ls_sorted_trajs(ds_root)[:max_trajs]

    frame_files = _ls_sorted_frames(traj_dirs[0])[::every_nth_frame]
    with h5py.File(frame_files[0], "r") as f0:
        n_points, dim = f0["r"].shape
    trajectories = {
        "u": np.zeros(
            (len(traj_dirs), len(frame_files), n_points, dim), dtype=np.float64
        ),
        "r": np.zeros(
            (len(traj_dirs), len(frame_files), n_points, dim), dtype=np.float64
        ),
    }

    for i, traj_dir in enumerate(traj_dirs):
        files = _ls_sorted_frames(traj_dir)[::every_nth_frame]

        for j, frame in enumerate(files):
            with h5py.File(frame, "r") as f:
                trajectories["u"][i, j] = f["u"][:]
                trajectories["r"][i, j] = f["r"][:]

    with open(traj_dirs[0] / "config.yaml", "r") as f:
        meta = yaml.load(f, Loader=yaml.FullLoader)

    steps = np.array([_step_from_filename(f) for f in frame_files])
    dt = float(meta["sim"]["dt"])
    trajectories["steps"] = steps
    trajectories["t"] = steps * dt
    return trajectories


def plt_diagnostics_sim(ds_root: Path) -> None:
    """Plot diagnostics evolution from a single mode=simulation csv."""
    for key in ["e_inj", "ekin", "umax"]:
        fig, ax = plt.subplots(figsize=(8, 4))
        df = pd.read_csv(ds_root / "diagnostics.csv")
        ax.plot(df["time"], df[key])
        ax.set_xlabel("Time [-]")
        ax.set_ylabel(key)
        ax.grid()
        fig.tight_layout()
        fig.savefig(ds_root / f"evolution_{key}.png")
        plt.close()
    print("Finished plt_diagnostics_sim !")


def plt_diagnostics_com(ds_root, max_trajs=10, fig_dir=Path("figs")):
    """Plot diagnostics evolution from a set of trajs with mode=combined."""
    traj_dirs = _ls_sorted_trajs(ds_root)
    (ds_root / fig_dir).mkdir(parents=True, exist_ok=True)
    for key in ["e_inj", "ekin", "umax", "rho_max"]:
        fig, ax = plt.subplots(figsize=(8, 4))
        for traj_dir in traj_dirs[:max_trajs]:
            df = pd.read_csv(traj_dir / "diagnostics.csv")
            ax.plot(df["time"], df[key])
        ax.set_xlabel("Time [-]")
        ax.set_ylabel(f"{key}")
        ax.grid()
        fig.tight_layout()
        fig.savefig(ds_root / fig_dir / f"evolution_{key}.png")
        plt.close()
    print("Finished plt_diagnostics_com !")
