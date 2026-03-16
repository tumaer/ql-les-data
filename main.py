import os

from jax import config
from omegaconf import OmegaConf


def check_cfg(cfg):
    default_cfg = OmegaConf.load("configs/defaults.yaml")
    # ensure cfg is a subset of default_cfg, ignore "config" key at top level
    for key in cfg:
        if key == "config":
            continue
        if key not in default_cfg:
            raise ValueError(f"Unknown config key: {key}")
        if isinstance(cfg[key], dict):
            for subkey in cfg[key]:
                if subkey not in default_cfg[key]:
                    raise ValueError(f"Unknown config key: {key}.{subkey}")


if __name__ == "__main__":
    cli_args = OmegaConf.from_cli()
    cfg = OmegaConf.merge(OmegaConf.load(cli_args.config), cli_args)
    check_cfg(cfg)

    if str(cfg.gpu) == "-1":
        print("Running on CPU")
        os.environ["JAX_PLATFORMS"] = "cpu"
        config.update("jax_platform_name", "cpu")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(cfg.get("xla_mem_fraction", 0.8))

    if cfg.float64:
        config.update("jax_enable_x64", True)

    print("#" * 79, "\nStarting a run with the following configs:")
    print(OmegaConf.to_yaml(cfg))
    print("#" * 79)

    from l3es.combined import combined
    from l3es.spectral_solver import simulate

    if cfg.mode == "simulate":
        os.makedirs(cfg.sim.dst_path, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(cfg.sim.dst_path, "config.yaml"))
        simulate(
            # Simulation
            case=cfg.sim.case,
            N=cfg.sim.N,
            dim=cfg.sim.dim,
            nu=cfg.sim.nu,
            t_final=cfg.sim.t_final,
            t_burnin=cfg.sim.t_burnin,
            dt=cfg.sim.dt,
            u_ref=cfg.sim.u_ref,
            ckp_N=cfg.sim.ckp_N,
            # Global
            seed=cfg.seed,
            rejit=cfg.get("rejit", False),
            # HIT forcing
            forcing_type=cfg.sim.get("forcing_type", "none"),
            e_kin_target=cfg.sim.get("e_kin_target", None),
            kf=cfg.sim.get("kf", None),
            # IO
            log_freq=cfg.sim.log_freq,
            vis_freq=cfg.sim.vis_freq,
            ckp_freq=cfg.sim.ckp_freq,
            dst_path=cfg.sim.dst_path,
        )
    elif cfg.mode == "combined":
        os.makedirs(cfg.com.dst_path, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(cfg.com.dst_path, "config.yaml"))
        combined(
            # Simulation
            case=cfg.sim.case,
            N=cfg.sim.N,
            dim=cfg.sim.dim,
            nu=cfg.sim.nu,
            t_final=cfg.sim.t_final,
            t_burnin=cfg.sim.t_burnin,
            dt=cfg.sim.dt,
            u_ref=cfg.sim.u_ref,
            ckp_N=cfg.sim.ckp_N,
            # Interpolation
            state_0_path=cfg.com.state_0_path,
            interp_backend=cfg.com.interp_backend,
            relax_dt_factor=cfg.com.relax_dt_factor,
            dft_splits=cfg.com.get("dft_splits", 8),
            # Global
            seed=cfg.seed,
            rejit=cfg.get("rejit", False),
            # HIT forcing
            forcing_type=cfg.sim.get("forcing_type", "none"),
            e_kin_target=cfg.sim.get("e_kin_target", None),
            kf=cfg.sim.get("kf", None),
            # IO
            log_freq=cfg.com.log_freq,
            vis_freq=cfg.com.vis_freq,
            ckp_freq=cfg.com.ckp_freq,
            dst_path=cfg.com.dst_path,
        )
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")
