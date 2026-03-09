import os

import numpy as np
import xarray
import yaml

with open("results/kolm64_1/config.yaml", "r") as f:
    meta = yaml.load(f, Loader=yaml.FullLoader)

burnin = 65  # 65
trajs = 4  # 3

n_times = len(os.listdir("results/kolm64_1/ckp")) - burnin
u = np.zeros((2, trajs, n_times, 64, 64))  # (Dim, Trajs, Time, X, Y)

sample = np.arange(trajs)
# sample = np.array([0, 2])
for i, s in enumerate(sample):
    files = os.listdir(f"results/kolm64_{s+1}/ckp")
    files.sort()
    files = files[burnin:]  # burn-in period of t=4.5 and ckp_dt=0.07
    for j, f in enumerate(files):
        u[:, i, j] = np.load(f"results/kolm64_{s+1}/ckp/" + f)  # (2,64,64)

time = np.arange(len(files)) * meta["sim"]["dt"] * meta["sim"]["ckp_freq"]
x = y = (np.arange(64) + 0.5) * 2 * np.pi / 64
# print(time.shape, x.shape, y.shape, u.shape)
# print(time[:5], x[:5], y[:5], u[0, :5, :5])

ds = xarray.Dataset(
    data_vars={
        "u": (("sample", "time", "x", "y"), u[0]),
        "v": (("sample", "time", "x", "y"), u[1]),
    },
    coords={"sample": np.arange(len(sample)), "time": time, "x": x, "y": y},
    attrs={"description": "Synthetic dataset for testing"},
)
print(f"Dataset: \n{ds}")
ds.to_netcdf(f"results/kolm64_1/trajs{trajs}.nc")
