import argparse
import tqdm
import h5py
import os

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")

import shtns
import jaxbind

import models.time_solver as imex
from models.sph_mhd import (
    SphericalMHD,
    dynamical_solver,
    integral,
    kinetic_energy,
)


def main(args: argparse.Namespace) -> None:
    eq = SphericalMHD(
        Ra=args.Ra,
        E=args.E,
        radius_ratio=args.rr,
        l_max=args.l_max,
        m_max=args.m_max,
        m_res=args.m_res,
        n_r=args.n_r,
    )

    shtns_cfg = shtns.sht(
        eq.l_max, mres=eq.m_res
    )  # single configuration used throughout
    shtns_cfg.set_grid(nl_order=2)

    shtns_cfg.wts = shtns_cfg.gauss_wts() * (2 * np.pi) / shtns_cfg.nphi
    shtns_cfg.wts = np.hstack((shtns_cfg.wts, np.flip(shtns_cfg.wts))).reshape((-1, 1))
    shtns_cfg.wts_1 = 1.0 / shtns_cfg.wts
    shtns_cfg._l2 = shtns_cfg.l * (shtns_cfg.l + 1.0)
    shtns_cfg._l2[eq.l_max + 1 :] *= 2.0
    shtns_cfg._l_2 = np.zeros_like(shtns_cfg._l2)
    shtns_cfg._l_2[1:] = 1.0 / shtns_cfg._l2[1:]

    def sht_synth(out, args, kwargs_dump):
        nargs = len(args)
        if nargs == 1:
            out[0][()] = shtns_cfg.synth(args[0])
        else:
            q = shtns_cfg.synth(*args)
            for i in range(nargs):
                out[i][()] = q[i]

    def sht_analys(out, args, kwargs_dump):
        nargs = len(args)
        if nargs == 1:
            out[0][()] = shtns_cfg.analys(args[0])
        else:
            alm = shtns_cfg.analys(*args)
            for i in range(nargs):
                out[i][()] = alm[i]

    def sht_synth_transposed(out, args, kwargs_dump):
        nargs = len(args)
        if nargs == 1:
            xlm = shtns_cfg.analys(args[0] * shtns_cfg.wts_1)
            xlm[shtns_cfg.lmax + 1 :] *= 2.0
            out[0][()] = xlm
        elif nargs == 2:
            slm, tlm = shtns_cfg.analys(
                args[0] * shtns_cfg.wts_1, args[1] * shtns_cfg.wts_1
            )
            out[0][()] = slm * shtns_cfg._l2
            out[1][()] = tlm * shtns_cfg._l2
        elif nargs == 3:
            qlm, slm, tlm = shtns_cfg.analys(
                args[0] * shtns_cfg.wts_1,
                args[1] * shtns_cfg.wts_1,
                args[2] * shtns_cfg.wts_1,
            )
            qlm[shtns_cfg.lmax + 1 :] *= 2.0
            out[0][()] = qlm
            out[1][()] = slm * shtns_cfg._l2
            out[2][()] = tlm * shtns_cfg._l2

    def sht_analys_transposed(out, args, kwargs_dump):
        nargs = len(args)

        if nargs == 1:
            qlm = args[0].copy()
            qlm[shtns_cfg.lmax + 1 :] *= 0.5
            out[0][()] = shtns_cfg.synth(qlm) * shtns_cfg.wts
        elif nargs == 2:
            u, v = shtns_cfg.synth(args[0] * shtns_cfg._l_2, args[1] * shtns_cfg._l_2)
            out[0][()] = u * shtns_cfg.wts
            out[1][()] = v * shtns_cfg.wts
        elif nargs == 3:
            qlm = args[0].copy()
            qlm[shtns_cfg.lmax + 1 :] *= 0.5
            q, u, v = shtns_cfg.synth(
                qlm, args[1] * shtns_cfg._l_2, args[2] * shtns_cfg._l_2
            )
            out[0][()] = q * shtns_cfg.wts
            out[1][()] = u * shtns_cfg.wts
            out[2][()] = v * shtns_cfg.wts

    def sht_synth_abstract_eval(*args, **kwargs):
        nargs = len(args)
        assert nargs <= 3
        for i in range(nargs):
            assert args[i].shape == (shtns_cfg.nlm,)
            assert args[i].dtype == jnp.complex128

        out_shape = (shtns_cfg.nlat, shtns_cfg.nphi)
        return ((out_shape, jnp.float64),) * nargs

    def sht_analys_abstract_eval(*args, **kwargs):
        nargs = len(args)
        assert nargs <= 3
        for i in range(nargs):
            assert args[i].shape == (shtns_cfg.nlat, shtns_cfg.nphi)
            assert args[i].dtype == jnp.float64

        out_shape = (shtns_cfg.nlm,)
        return ((out_shape, jnp.complex128),) * nargs

    sht_synth_jax = jaxbind.get_linear_call(
        sht_synth,
        sht_synth_transposed,
        sht_synth_abstract_eval,
        sht_analys_abstract_eval,
        func_can_batch=False,
    )

    sht_analys_jax = jaxbind.get_linear_call(
        sht_analys,
        sht_analys_transposed,
        sht_analys_abstract_eval,
        sht_synth_abstract_eval,
        func_can_batch=False,
    )

    print(eq)

    theta_lm = np.zeros((len(eq.fd_grid), shtns_cfg.nlm), dtype=np.complex128)
    u_tor_lm = np.zeros((len(eq.fd_grid), shtns_cfg.nlm), dtype=np.complex128)
    u_pol_lm = np.zeros((len(eq.fd_grid), shtns_cfg.nlm), dtype=np.complex128)

    lm44 = shtns_cfg.idx(4, 4)
    z2 = (2 * eq.fd_grid - eq.r_grid[0] - eq.r_grid[-1]) ** 2
    theta_lm[:, lm44] = 0.1 * (1 - 3 * z2 + 3 * z2**2 - z2**3)

    def source(theta_lm, u_tor_lm, u_pol_lm):
        return 0

    dt = args.dt
    time = 0
    step_fn = jax.jit(
        dynamical_solver(
            eq,
            imex.PC2(dt),
            source,
            sht_synth_jax,  # temporary
            sht_analys_jax,  # temporary
        )
    )

    print("Running spinup...")

    tke_t = []
    buo_t = []

    iters_spinup = int(args.spinup_time / dt)
    pbar = tqdm.tqdm(range(iters_spinup), bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")
    for i in pbar:
        c, theta_lm, u_tor_lm, u_pol_lm = step_fn(
            jnp.array(theta_lm),
            jnp.array(u_tor_lm),
            jnp.array(u_pol_lm),
        )

        time += dt
        if i % 25 == 0:
            buo_t.append(0.5 * integral(eq, np.array(theta_lm), shtns_cfg.m))
            eke_lm = kinetic_energy(eq, np.array(u_tor_lm), np.array(u_pol_lm))
            tke_t.append(
                np.sum(1.0 * eke_lm[:, shtns_cfg.m != 0])
                + np.sum(0.5 * eke_lm[:, shtns_cfg.m == 0])
            )

            pbar.set_postfix(buo_t=buo_t[-1], eke_t=tke_t[-1])

    data_path = os.path.join(os.path.join(os.getcwd(), "data"), args.config)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    eq.save(
        os.path.join(data_path, "spinup.h5"),
        time,
        theta_lm,
        u_tor_lm,
        u_pol_lm,
    )

    print("Saving samples...")

    data_samples = []
    data_t0 = []

    total_iters = args.samples
    sub_trajs_freq = args.steps

    pbar = tqdm.tqdm(range(total_iters), bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")
    for i in pbar:
        c, theta_lm, u_tor_lm, u_pol_lm = step_fn(
            jnp.array(theta_lm),
            jnp.array(u_tor_lm),
            jnp.array(u_pol_lm),
        )

        time += args.dt
        sub_traj_iter = i % sub_trajs_freq
        if sub_traj_iter == 0:
            # Starting a sub-trajectory
            data_t0.append(time)
        if sub_traj_iter < args.steps:
            # Sample each iterations
            data_samples.append((theta_lm, u_tor_lm, u_pol_lm))

    print("Saving dataset...")
    with h5py.File(os.path.join(data_path, args.config + "_samples.h5"), "w") as f:
        f.attrs["dt"] = args.dt
        f.attrs["steps"] = args.steps

        f.create_dataset("t_0", data=np.array(data_t0))
        f.create_dataset("f_m", data=np.array(data_samples))

    eq.save(
        os.path.join(data_path, args.config + "_snapshot.h5"),
        time,
        theta_lm,
        u_tor_lm,
        u_pol_lm,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python sample.py",
        description="Run a simulation and generate state samples (velocities and temperature).",
    )

    parser.add_argument(
        "-c", "--config", type=str, help="Name of the configuration", required=True
    )

    parser.add_argument("-Ra", type=float, help="Rayleigh number", required=True)
    parser.add_argument("-E", type=float, help="Ekman number", required=True)

    parser.add_argument(
        "-rr",
        type=float,
        default=0.35,
        help="Ratio between inner and outer spherical radius",
    )

    parser.add_argument(
        "-n_r", type=int, help="Number of radial grid points", required=True
    )
    parser.add_argument(
        "-l_max", type=int, help="Maximum degree of spherical harmonics", required=True
    )
    parser.add_argument(
        "-m_max", type=int, help="Maximum Fourier mode (phi)", required=True
    )
    parser.add_argument(
        "-m_res",
        type=int,
        default=1,
        help="Phi-periodicity (1 for full-sphere, 2 for half-sphere, etc)",
    )

    parser.add_argument(
        "-dt", type=float, help="Discrete (fixed) time step", required=True
    )

    parser.add_argument(
        "-steps",
        type=int,
        help="Number of discrete time steps for a sub-trajectory",
        required=True,
    )
    parser.add_argument(
        "-samples",
        type=int,
        help="Total number of samples for the dataset",
        required=True,
    )
    parser.add_argument(
        "-spinup_time", type=float, help="Time required for spinup", required=True
    )

    args = parser.parse_args()
    main(args)
