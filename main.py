import os

from jax import config
from omegaconf import OmegaConf

if __name__ == "__main__":
    cli_args = OmegaConf.from_cli()
    # TODO: add a check whether the cli_args are a subset of the defaults
    cfg = OmegaConf.merge(OmegaConf.load(cli_args.config), cli_args)

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
    from l3es.integrator import integrate
    from l3es.spectral_solver import simulate

    if cfg.mode == "simulate":
        os.makedirs(cfg.sim.dst_path, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(cfg.sim.dst_path, "config.yaml"))
        simulate(
            case=cfg.sim.case,
            N=cfg.sim.N,
            dim=cfg.sim.dim,
            nu=cfg.sim.nu,
            t_final=cfg.sim.t_final,
            burnin=cfg.sim.get("burnin", 0),
            dt=cfg.sim.dt,
            u_ref=cfg.sim.u_ref,
            seed=cfg.seed,
            log_freq=cfg.sim.log_freq,
            vis_freq=cfg.sim.vis_freq,
            ckp_freq=cfg.sim.ckp_freq,
            ckp_N=cfg.sim.ckp_N,
            dst_path=cfg.sim.dst_path,
            kf=cfg.sim.get("kf", None),
            forcing_type=cfg.sim.get("forcing_type", "none"),
            e_kin_target=cfg.sim.e_kin_target,
            rejit=cfg.get("rejit", True),
        )
    elif cfg.mode == "integrate":
        os.makedirs(cfg.int.dst_path, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(cfg.int.dst_path, "config.yaml"))
        # dt_SPH = cfl * h / (c_ref + u_ref)
        # dt_SPH = 1.0 * 2*3.1416/32 / (11 * 4) = 0.0045 !
        integrate(
            src_path=cfg.sim.dst_path,
            dst_path=cfg.int.dst_path,
            state_0_path=cfg.int.state_0_path,
            N=cfg.sim.ckp_N,
            dim=cfg.sim.dim,
            dt=cfg.sim.dt,
            splits=cfg.int.splits,
            u_ref=cfg.sim.u_ref,
            relax=cfg.int.relax,
            vis_freq=cfg.int.vis_freq,
        )
    elif cfg.mode == "combined":
        os.makedirs(cfg.com.dst_path, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(cfg.com.dst_path, "config.yaml"))
        combined(
            case=cfg.sim.case,
            dst_path=cfg.com.dst_path,
            state_0_path=cfg.int.state_0_path,
            N=cfg.sim.N,
            ckp_N=cfg.sim.ckp_N,
            dim=cfg.sim.dim,
            nu=cfg.sim.nu,
            t_final=cfg.sim.t_final,
            burnin=cfg.sim.get("burnin", 0),
            dt=cfg.sim.dt,
            splits=cfg.int.splits,
            u_ref=cfg.sim.u_ref,
            relax=cfg.int.relax,
            log_freq=cfg.com.log_freq,
            vis_freq=cfg.com.vis_freq,
            ckp_freq=cfg.com.ckp_freq,
            seed=cfg.seed,
            debug=cfg.com.get("debug", False),
            kf=cfg.sim.get("kf", None),
            forcing_type=cfg.sim.get("forcing_type", "none"),
            e_kin_target=cfg.sim.e_kin_target,
            rejit=cfg.get("rejit", True),
        )
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")
