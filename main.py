import os

from jax import config
from omegaconf import OmegaConf

if __name__ == "__main__":
    cli_args = OmegaConf.from_cli()
    # TODO: add a check whether the cli_args are a subset of the defaults
    cfg = OmegaConf.merge(OmegaConf.load(cli_args.config), cli_args)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)

    if cfg.float64:
        config.update("jax_enable_x64", True)

    print("#" * 79, "\nStarting a run with the following configs:")
    print(OmegaConf.to_yaml(cfg))
    print("#" * 79)

    from l3es.integrator import integrate
    from l3es.spectral_solver import simulate

    if cfg.mode == "simulate":
        simulate(
            case=cfg.sim.case,
            N=cfg.sim.N,
            dim=cfg.sim.dim,
            nu=cfg.sim.nu,
            t_final=cfg.sim.t_final,
            dt=cfg.sim.dt,
            u_ref=cfg.sim.u_ref,
            seed=cfg.sim.seed,
            log_freq=cfg.sim.log_freq,
            vis_freq=cfg.sim.vis_freq,
            ckp_freq=cfg.sim.ckp_freq,
            ckp_N=cfg.sim.ckp_N,
            dst_path=cfg.sim.dst_path,
        )
    elif cfg.mode == "integrate":
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
        )
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")
