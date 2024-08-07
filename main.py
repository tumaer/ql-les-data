from jax import config
from omegaconf import OmegaConf

from l3es.spectral_solver import simulate_wrapper

if __name__ == "__main__":
    cli_args = OmegaConf.from_cli()
    # TODO: add a check whether the cli_args are a subset of the defaults
    cfg = OmegaConf.merge(OmegaConf.load(cli_args.config), cli_args)

    if cfg.float64:
        config.update("jax_enable_x64", True)

    print("#" * 79, "\nStarting a run with the following configs:")
    print(OmegaConf.to_yaml(cfg))
    print("#" * 79)

    simulator = simulate_wrapper(
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
