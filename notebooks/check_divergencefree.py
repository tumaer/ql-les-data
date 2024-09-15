"""Check for different initializations if they are divergence-free."""

import numpy as np
import matplotlib.pyplot as plt
from phi.field import CenteredGrid, Noise
from phi.geom import Box
from phi import field
from phi.flow import fluid, plot, tensor, channel, spatial
from omegaconf import OmegaConf

from l3es.init_fields import init_u_hit, init_u_kolm, init_u_tgv2d
from l3es.utils import comp_divergence

def plt_div(u, div, cfg):
    fig, axs = plt.subplots(3, 1, figsize=(6, 15))
    for i, title in enumerate(["u_x", "u_y"]):
        axs[i].imshow(u[i], cmap="turbo")
        axs[i].set_title(f"{title} (min={u[i].min():.2f}, max={u[i].max():.2f})")
    axs[2].imshow(div, cmap="turbo")
    axs[2].set_title(f"Div (min={div.min():.7f}, max={div.max():.7f})")
    plt.tight_layout()
    plt.savefig(f"fig_{cfg.case}_{cfg.is_incompressible}_{cfg.div_method}.png")
    plt.close()
    
    # plot(v.at_centers().downsample(2).as_points(), size=(4, 3))
    # plt.savefig("divergence_phi_0.png")
    # plt.close()

def setup_domain(Nx, case):
    L = {"kolm": 2*np.pi, "random": 2*np.pi, "tgv2d": 1}[case]
    domain = Box(x=L, y=L)
    dx = L / Nx
    return domain, dx, Nx

cfg = OmegaConf.create({
    "Nx": 64,  # number of grid points per dimension
    "case": "random",  # ["kolm", "random", "tgv2d"]
    "is_incompressible": 0,  # whether to enforce incompressibility. 1: yes, 0: no
    "div_method": "simple",  # divergence calculation implementation. ["simple", "phi"]
})

domain, dx, Nx = setup_domain(cfg.Nx, cfg.case)
if cfg.case == "kolm": # divergence-free by construction
    u_ = np.asarray(init_u_kolm(Nx, target_dim=3)[:2, :, :, 0])  # shape (2, Nx, Nx)
    u_ = u_[:, ::-1, ::-1]
    # reverse the order along the y-axis
    # u_ = u_[:, :, :]
    v0 = CenteredGrid(tensor(u_, channel(vector="x,y"), spatial("x,y")), 0, domain, x=Nx, y=Nx, extrapolation=fluid.extrapolation.PERIODIC)
elif cfg.case == "random": # not divergence-free
    v0 = CenteredGrid(Noise(vector='x,y', smoothness=1.0), 0, domain, x=Nx, y=Nx, extrapolation=fluid.extrapolation.PERIODIC)
elif cfg.case == "tgv2d": # divergence-free by construction
    u_ = init_u_tgv2d(Nx)
    v0 = CenteredGrid(tensor(u_, channel(vector="x,y"), spatial("x,y")), 0, domain, x=Nx, y=Nx, extrapolation=fluid.extrapolation.PERIODIC)

v = v0
if cfg.is_incompressible == 1: # enforce incompressiblity
    v, _ = fluid.make_incompressible(v0)
u = v.at_centers().downsample(1).numpy().transpose(2,0,1)

if cfg.div_method == "simple":   
    div = comp_divergence(u, dx, version=2)
    # for tmp in v.numpy():
    #     print(tmp.shape)
elif cfg.div_method == "phi":
    div = field.divergence(v).numpy()
    # zero out the boundaries
    div[0, :] = 0; div[-1, :] = 0; div[:, 0] = 0; div[:, -1] = 0
plt_div(u, div, cfg)
