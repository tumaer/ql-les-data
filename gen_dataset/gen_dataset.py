"""Script for generating ML datasets from h5 simulation frames"""

import argparse
import json
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import h5py
import numpy as np
from jax import vmap
from jax_sph.io_state import read_h5, write_h5
from jax_sph.jax_md import space
from omegaconf import OmegaConf


def write_h5_frame_for_visualization(state_dict, file_path_h5):
    path_file_vis = os.path.join(file_path_h5[:-3] + "_vis.h5")
    print("writing to", path_file_vis)
    write_h5(state_dict, path_file_vis)
    print("done")


def single_h5_files_to_h5_dataset(args):
    """Transform a set of .h5 files to a single .h5 dataset file

    Args:
        src_dir: source directory containing other directories, each with .h5 files
            corresponding to a trajectory
        dst_dir: destination directory where three files will be written: train.h5,
            valid.h5, and test.h5
        split: string of three integers separated by underscores, e.g. "80_10_10"
    """

    os.makedirs(args.dst_dir, exist_ok=True)

    # list only directories in a root with files and directories
    dirs = os.listdir(args.src_dir)
    dirs = [d for d in dirs if os.path.isdir(os.path.join(args.src_dir, d))]
    # order by seed value
    dirs = sorted(dirs, key=lambda x: int(x.split("_")[-1]))

    splits_array = np.array([int(s) for s in args.split.split("_")])
    splits_sum = splits_array.sum()

    # multiple trajectories
    num_eval = np.ceil(splits_array[1] / splits_sum * len(dirs)).astype(int)
    # at least one validation and one testing trajectory
    splits_trajs = np.cumsum([0, len(dirs) - 2 * num_eval, num_eval, num_eval])
    num_trajs_train, num_trajs_test = len(dirs) - 2 * num_eval, num_eval

    # seqience_length should be after subsampling every nth trajectory
    # and "-1" because of the last target position (see GNS dataset format)
    files_per_traj = len(os.listdir(os.path.join(args.src_dir, dirs[0], "com")))
    files_per_traj = np.ceil(
        (files_per_traj - args.skip_first_n_frames) / args.slice_every_nth_frame,
    ).astype(int)
    sequence_length_train = sequence_length_test = files_per_traj

    for i, split in enumerate(["train", "valid", "test"]):
        hf = h5py.File(os.path.join(args.dst_dir, f"{split}.h5"), "w")

        # multiple trajectories
        for j, dir in enumerate(dirs[splits_trajs[i] : splits_trajs[i + 1]]):
            traj_path = os.path.join(args.src_dir, dir, "com")
            files = os.listdir(traj_path)
            files = [f for f in files if (".h5" in f)]
            files = sorted(files, key=lambda x: int(x.split("_")[1][:-3]))
            files = files[args.skip_first_n_frames :: args.slice_every_nth_frame]

            position, velocity = [], []
            for k, filename in enumerate(files):
                file_path_h5 = os.path.join(traj_path, filename)
                state = read_h5(file_path_h5, array_type="numpy")
                r = state["r"]
                u = state["u"]

                if args.is_visualize:
                    write_h5_frame_for_visualization({"r": r, "u": u}, file_path_h5)
                position.append(r)
                velocity.append(u)

            position = np.stack(position)  # (time steps, particles, dim)
            velocity = np.stack(velocity)  # (time steps, particles, dim)
            particle_type = np.zeros(position.shape[1])  # (particles,)

            traj_str = str(j).zfill(5)
            pos_shape = position.shape
            hf.create_dataset(f"{traj_str}/particle_type", data=particle_type)
            hf.create_dataset(
                f"{traj_str}/u", 
                data=velocity, 
                dtype=np.float32, 
                compression="gzip",
                chunks=(1, pos_shape[1], pos_shape[2])
            )
            hf.create_dataset(
                f"{traj_str}/position",
                data=position,
                dtype=np.float32,
                compression="gzip",
                chunks=(1, pos_shape[1], pos_shape[2])
            )

        hf.close()
        print(f"Finished {args.src_dir} {split} with {j+1} entries!")
        print(f"Sample positions shape {position.shape}")

    # metadata
    # Compatible with the lagrangebench metadata.json files
    cfg = OmegaConf.load(os.path.join(args.src_dir, dir, "config.yaml"))

    metadata = {
        "case": cfg.sim.case.upper(),
        "dim": cfg.sim.dim,
        "dx": 2 * np.pi / cfg.sim.ckp_N,
        "dt": cfg.sim.dt,
        "t_end": cfg.sim.t_final,
        "viscosity": cfg.sim.nu,
        "u_ref": cfg.sim.u_ref,
        "write_every": cfg.com.ckp_freq * args.slice_every_nth_frame,
        "sequence_length_train": int(sequence_length_train),
        "num_trajs_train": int(num_trajs_train),
        "sequence_length_test": int(sequence_length_test),
        "num_trajs_test": int(num_trajs_test),
        "num_particles_max": cfg.sim.ckp_N**cfg.sim.dim,
        "periodic_boundary_conditions": [True] * 3,
        "bounds": [[0, 2 * np.pi]] * cfg.sim.dim,
    }
    x = 1.5 * metadata["dx"]
    x = np.format_float_positional(
        x, precision=2, unique=False, fractional=False, trim="k"
    )
    metadata["default_connectivity_radius"] = float(x)

    with open(os.path.join(args.dst_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)
    
    print("Finished writing metadata!")


def compute_statistics_h5(args):
    """Compute the mean and std of a h5 dataset files"""

    # metadata
    with open(os.path.join(args.dst_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)

    # apply PBC in all directions or not at all
    if np.array(metadata["periodic_boundary_conditions"]).any():
        box = np.array(metadata["bounds"])
        box = box[:, 1] - box[:, 0]
        displacement_fn, _ = space.periodic(side=box)
    else:
        displacement_fn, _ = space.free()

    displacement_fn_sets = vmap(vmap(displacement_fn, in_axes=(0, 0)))

    # shifting velocity
    vs, avs = [], []
    v_sq, av_sq = [], []
    v_mean = av_mean = 0.0
    # physical velocity
    us, aus = [], []
    u_sq, au_sq = [], []
    u_mean = au_mean = 0.0
    for loop in ["mean", "std"]:
        for split in ["train"]:  # ["train", "valid", "test"]
            hf = h5py.File(os.path.join(args.dst_dir, f"{split}.h5"), "r")

            for _, v in hf.items():
                tag = v.get("particle_type")[:]
                fluid_tags = tag == 0  # only fluid ("0") particles
                r = v["position"][::args.stats_every_nth][:, fluid_tags]
                u = v["u"][::args.stats_every_nth][:, fluid_tags]

                # The velocity and acceleration computation is based on an
                # inversion of Semi-Implicit Euler
                vel = displacement_fn_sets(r[1:], r[:-1])
                if loop == "mean":
                    vs.append(vel.mean((0, 1)))
                    avs.append((vel[1:] - vel[:-1]).mean((0, 1)))

                    us.append(u.mean((0, 1)))
                    aus.append((u[1:] - u[:-1]).mean((0, 1)))
                elif loop == "std":
                    centered_vel = vel - v_mean
                    v_sq.append(np.square(centered_vel).mean((0, 1)))
                    centered_av = vel[1:] - vel[:-1] - av_mean
                    av_sq.append(np.square(centered_av).mean((0, 1)))

                    centered_u = u - u_mean
                    u_sq.append(np.square(centered_u).mean((0, 1)))
                    centered_au = u[1:] - u[:-1] - au_mean
                    au_sq.append(np.square(centered_au).mean((0, 1)))
            hf.close()

        if loop == "mean":
            vel_mean = np.stack(vs).mean(0)
            acc_mean = np.stack(avs).mean(0)
            print(f"vel_mean={vel_mean}, acc_mean={acc_mean}")
            u_mean = np.stack(us).mean(0)
            au_mean = np.stack(aus).mean(0)
            print(f"u_mean={u_mean}, au_mean={au_mean}")
        elif loop == "std":
            vel_std = np.stack(v_sq).mean(0) ** 0.5
            acc_std = np.stack(av_sq).mean(0) ** 0.5
            print(f"vel_std={vel_std}, acc_std={acc_std}")
            u_std = np.stack(u_sq).mean(0) ** 0.5
            au_std = np.stack(au_sq).mean(0) ** 0.5
            print(f"u_std={u_std}, au_std={au_std}")

    # stds should not be 0. If they are, set them to 1.
    vel_std = np.where(vel_std < 1e-7, 1, vel_std)
    acc_std = np.where(acc_std < 1e-7, 1, acc_std)
    metadata["vel_mean"] = vel_mean.tolist()
    metadata["vel_std"] = vel_std.tolist()
    metadata["acc_mean"] = acc_mean.tolist()
    metadata["acc_std"] = acc_std.tolist()

    u_std = np.where(u_std < 1e-7, 1, u_std)
    au_std = np.where(au_std < 1e-7, 1, au_std)
    metadata["u_mean"] = u_mean.tolist()
    metadata["u_std"] = u_std.tolist()
    metadata["au_mean"] = au_mean.tolist()
    metadata["au_std"] = au_std.tolist()

    assert metadata["write_every"] == 1 or args.stats_every_nth == 1, (
        "Either keep the dataset without slicing and have sliced stats files, "
        "or slice the dataset and have unsliced stats."
    )
    metadata["write_every"] = metadata["write_every"] * args.stats_every_nth

    file_path = f"metadata_every{args.stats_every_nth}.json"
    with open(os.path.join(args.dst_dir, file_path), "w") as f:
        json.dump(metadata, f)
        
    print("Finished updating metadata!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str)
    parser.add_argument("--dst_dir", type=str)
    parser.add_argument("--split", default="2_1_1", type=str, help="Relative ratio.")
    parser.add_argument("--skip_first_n_frames", type=int, default=0)
    parser.add_argument("--slice_every_nth_frame", type=int, default=1)
    parser.add_argument("--is_visualize", action="store_true")
    parser.add_argument("--stats_every_nth", type=int, default=1)
    parser.add_argument("--only_stats", action="store_true",)
    args = parser.parse_args()

    assert args.slice_every_nth_frame == 1 or args.stats_every_nth == 1, (
        "Either keep the dataset without slicing and have sliced stats files, "
        "or slice the dataset and have unsliced stats."
    )

    if not args.only_stats:
        single_h5_files_to_h5_dataset(args)
    compute_statistics_h5(args)
