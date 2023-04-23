import os
import sys
import time
from argparse import Namespace
import json

import numpy as np

import h5py
import pyvista


def write_vtk(data_dict, path):
    """Store a .vtk file for ParaView"""
    data_pv = dict2pyvista(data_dict)
    data_pv.save(path)


def dict2pyvista(data_dict):
    # PyVista works only with 3D objects, thus we check whether the inputs
    # are 2D and then increase the degrees of freedom of the second dimension.
    # N is the number of points and dim the dimension
    r = np.asarray(data_dict["r"]) # TODO:
    N, dim = r.shape

    # PyVista treats the position information differently than the rest
    if dim == 2:
        r = np.hstack([r, np.zeros((N, 1))])
    data_pv = pyvista.PolyData(r)

    # copy all the other information also to pyvista, using plain numpy arrays
    for k, v in data_dict.items():
        # skip r because we already considered it above
        if k == "r":
            continue

        # working in 3D or scalar features do not require special care
        if dim == 2 and v.ndim == 2:
            v = np.hstack([v, np.zeros((N, 1))])

        data_pv[k] = np.asarray(v)

    return data_pv


def pos_init_cartesian_3d(box_size, dx):
    n = np.array((box_size / dx).round(), dtype=int)
    grid = np.meshgrid(range(n[0]), range(n[1]), range(n[2]), indexing='xy')
    r = (np.vstack(list(map(np.ravel, grid))).T + 0.5) * dx
    return r

def h5_to_vtk(file_name):
    hf = h5py.File(file_name, "r")
    run_name = file_name.split("/")[-1].split(".")[0]
    os.makedirs(run_name, exist_ok=True)
    
    # positions
    data_dict = {"r": pos_init_cartesian_3d(np.array([1.0, 1.0, 1.0]), 1/64)}
    
    for k, v in hf.items():
        velocity = np.array(v["velocity"])
        data_dict["v"] = np.reshape(np.transpose(velocity, (1, 2, 3, 0)), (-1, 3))

        vtk_file_path = "./data/" + run_name + "/frame_" + k + ".vtk"
        write_vtk(data_dict, vtk_file_path)
        print(f"Wrote {vtk_file_path} to disk")
    hf.close()


if __name__ == "__main__":
    h5_to_vtk("./data/tgv_fields/traj_800.h5")
