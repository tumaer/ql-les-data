from pathlib import Path

import h5py
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import Normalize


def plt_burnin(path: Path, fig_dir: Path):
    """Plot in an imshow plot the u and v fields of the (2, 512, 512) tensor."""
    data = np.load(path)
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    kwargs = {"origin": "lower", "cmap": "turbo", "vmax": 6}
    fields = [data[0].T, data[1].T, np.linalg.norm(data, axis=0).T]
    axs[0].imshow(fields[0], vmin=-6, **kwargs)
    axs[1].imshow(fields[1], vmin=-6, **kwargs)
    axs[2].imshow(fields[2], vmin=0, **kwargs)

    for ax, lbl, f in zip(axs, ["ux", "uy", "|u|"], fields):
        ax.set_title(f"{lbl} (min={f.min():.2f}, max={f.max():.2f})")
    fig.tight_layout()
    fig.savefig(fig_dir / "burnin.png")
    plt.close()


def load_trajectories(ds_root: Path, burnin=4500, every_nth_frame=70) -> dict:
    """Loads trajectories from the specified dataset root directory."""

    traj_dirs = sorted([d for d in ds_root.glob("traj_*") if d.is_dir()])
    frame_files = sorted((traj_dirs[0] / "com").glob("step_*.h5"))
    frame_files = frame_files[burnin::every_nth_frame]
    trajectories = {
        "u": np.zeros((len(traj_dirs), len(frame_files), 4096, 2)),  # (S, T, N, D)
        "r": np.zeros((len(traj_dirs), len(frame_files), 4096, 2)),  # (S, T, N, D)
    }

    for i, traj_dir in enumerate(traj_dirs):
        frame_files = sorted((traj_dir / "com").glob("step_*.h5"))
        for j, frame in enumerate(frame_files[burnin::every_nth_frame]):
            with h5py.File(frame, "r") as f:
                # print(f["r"].shape, f["u"].shape) # (4096, 2)
                trajectories["u"][i, j] = f["u"][:]
                trajectories["r"][i, j] = f["r"][:]

    with open(traj_dirs[0] / "config.yaml", "r") as f:
        meta = yaml.load(f, Loader=yaml.FullLoader)

    times = np.arange(len(frame_files), dtype=float)[burnin::every_nth_frame] - burnin
    trajectories["t"] = times * (meta["sim"]["dt"] * meta["sim"]["ckp_freq"])
    return trajectories


def plt_u_max_per_timestep(trajs, fig_dir: Path = Path("figs")) -> None:
    """Prints max velocity u at each time step."""
    u_max_by_time = trajs["u"][..., 0].max(2)  # only velocity along x. shape (S, T)

    plt.figure(figsize=(8, 4))
    for idx in range(min(10, u_max_by_time.shape[0])):
        plt.plot(trajs["t"], u_max_by_time[idx])
    plt.title("Max u vs Time")
    plt.xlabel("Time")
    plt.ylabel("Max u")
    plt.grid()
    plt.tight_layout()
    plt.savefig(fig_dir / "max_u_over_time.png")
    plt.close()


def plt_ekin_evolution(trajs, fig_dir: Path = Path("figs")) -> None:
    """Plots kinetic energy evolution over time for a given trajectory."""
    ekin = 0.5 * (trajs["u"] ** 2).sum(-1).mean(-1)  # (S,T,N,D) -> (S,T)

    plt.figure(figsize=(8, 4))
    for idx in range(min(10, ekin.shape[0])):
        plt.plot(trajs["t"], ekin[idx])
    plt.title("Kinetic Energy Evolution Over Time")
    plt.xlabel("Time")
    plt.ylabel("Mean Kinetic Energy per Unit Mass")
    plt.grid()
    plt.tight_layout()
    plt.savefig(fig_dir / "kinetic_energy_evolution.png")
    plt.close()


def animate_rlt(trajs, idx, fig_dir: Path = Path("figs")) -> None:
    u_norm = np.linalg.norm(trajs["u"][idx], axis=-1)  # (S,T,N,D) -> (T,N)
    fig, ax = plt.subplots(figsize=(5, 5))
    norm = Normalize(vmin=0, vmax=6)
    length = trajs["t"].shape[0]

    scatter = ax.scatter(
        trajs["r"][idx, 0, :, 0],
        trajs["r"][idx, 0, :, 1],
        c=u_norm[0],
        cmap="turbo",
        norm=norm,
        s=10,
    )
    fig.colorbar(scatter, ax=ax, pad=0.02, fraction=0.048)

    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(0, 2 * np.pi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    title = ax.set_title(f"|u| at t={trajs['t'][0]:.2f}")
    fig.tight_layout()

    def update(frame):
        scatter.set_offsets(trajs["r"][idx, frame])
        scatter.set_array(u_norm[frame])
        title.set_text(f"|u| at t={trajs['t'][frame]:.2f}")
        return (scatter, title)

    anim = animation.FuncAnimation(fig, update, frames=length, interval=100, blit=False)
    anim.save(fig_dir / f"anim_{idx}.gif", writer="pillow", fps=10)
    plt.close()


if __name__ == "__main__":
    ds_root = Path("dataset_kolm/raw/2D_KOLM_4096_140kevery1")
    fig_dir = ds_root / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt_burnin(ds_root / "traj_5/u_512_04500_burnin.npy", ds_root / "traj_5")
    trajs = load_trajectories(ds_root, burnin=4500, every_nth_frame=100)
    plt_u_max_per_timestep(trajs, fig_dir)
    plt_ekin_evolution(trajs, fig_dir)
    animate_rlt(trajs, idx=0, fig_dir=fig_dir)

    # read all lines from "gen_dataset/jobs/kolm2d_10450_0.out" and extract the five
    # characters right after any match of the string "rho_max=" anywhere on a line

    idxs = np.arange(0, 4)
    res_all = []
    for idx in idxs:
        res = []
        file_path = Path(f"gen_dataset/jobs/kolm2d_10450_{idx}.out")
        target = "rho_max="
        if file_path.exists():
            with open(file_path, "r") as f:
                for line in f:
                    if target in line:
                        # Find the index where the value starts
                        start_idx = line.find(target) + len(target)
                        # Slice the next 5 characters
                        value = line[start_idx : start_idx + 5]
                        res.append(float(value))
        res_all.append(np.array(res[:]))
    print("Time steps: ", res_all[0].shape)

    fig, ax = plt.subplots(figsize=(10, 5))
    for res in res_all:
        ax.plot(res)
    plt.tight_layout()
    plt.grid()
    fig.savefig(fig_dir / "rho_max.png")
