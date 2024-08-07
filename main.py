from jax import config
from omegaconf import OmegaConf

from l3es.integrator import integrate
from l3es.spectral_solver import simulate

if __name__ == "__main__":
    cli_args = OmegaConf.from_cli()
    # TODO: add a check whether the cli_args are a subset of the defaults
    cfg = OmegaConf.merge(OmegaConf.load(cli_args.config), cli_args)

    if cfg.float64:
        config.update("jax_enable_x64", True)

    print("#" * 79, "\nStarting a run with the following configs:")
    print(OmegaConf.to_yaml(cfg))
    print("#" * 79)

    if cfg.mode == "simulate":
        simulate(
            case=cfg.sim.case,
            N=cfg.sim.N,
            nu=cfg.sim.nu,
            t_final=cfg.sim.t_final,
            dt=cfg.sim.dt,
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
            N=cfg.sim.ckp_N,
            dt=cfg.sim.dt,
            splits=cfg.int.splits,
        )
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")
