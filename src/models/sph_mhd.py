from typing import Callable, Dict, Optional, Tuple

import h5py
import numpy as np
import jax
import jax.numpy as jnp

import shtns

from models.time_solver import Scheme
from models._algebra import (
    tridiag_dot,
    pentadiag_dot,
    tridiag_solve,
    pentadiag_factor_lu,
    pentadiag_solve_lu,
)
from models._adj_sh import (
    shtns_synth,
    shtns_analys,
)


class SphericalMHD:
    def __init__(
        self,
        Ra: float,
        E: float,
        Pr: float,
        radius_ratio: float,
        l_max: int,
        m_max: int,
        m_res: int,
        n_r: int = 0,
        r: Optional[np.ndarray] = None
    ):
        assert n_r > 3 or len(r) > 3

        self.Ra = (
            Ra  # rotating Rayleigh number, that is the canonical Rayleigh times Ekman
        )
        self.E = E  # Ekman number
        self.Pr = Pr  # Prandtl number
        self.ri = radius_ratio / (1 - radius_ratio)
        self.ro = 1 / (1 - radius_ratio)
        self.vol = (4.0 / 3.0) * np.pi * (self.ro**3 - self.ri**3)
        self.l_max = l_max
        self.m_max = m_max
        self.m_res = m_res
        self.n_r = n_r if r is None else len(r)

        self.sh = shtns.sht(self.l_max, mres=self.m_res)
        self.sh.set_grid(
            nl_order=2, flags=shtns.SHT_ALLOW_GPU + shtns.SHT_PHI_CONTIGUOUS
        )  # GPU by default

        self.l_map = np.array(
            [
                l
                for m in range(0, self.m_max + 1)
                for l in range(m * self.m_res, self.l_max + 1)
            ]
        )
        self.n_lm = len(self.l_map)

        self.r_grid = r if r is not None else boundary_layer_grid(
            n_r=self.n_r,
            ro=self.ro,
            ri=self.ri,
            n_bli=10,  # ?
            n_blo=10,  # ?
            dx_bli=np.sqrt(self.E),
            dx_blo=np.sqrt(self.E),
        )

        dr, d2r = self.__fd_matrices__()
        self.d0dr = dr["dl"][0].item()
        self.dndr = dr["du"][-1].item()
        self.d0d2r = d2r["dl"][0].item()
        self.dnd2r = d2r["du"][-1].item()

        # Finite differences stencil
        dr["dl"][0] = d2r["dl"][0] = 0
        dr["du"][-1] = d2r["du"][-1] = 0
        self.fd_grid = self.r_grid[1:-1]
        self.fd_norm = np.expand_dims(
            self.fd_grid * self.fd_grid * (self.r_grid[2:] - self.r_grid[:-2]), -1
        )
        self.fd_grid_v = np.expand_dims(self.fd_grid, -1)

        # Gradient
        self.grad = {"dl": dr["dl"], "d": dr["d"], "du": dr["du"]}
        # Curl
        self.curl = {
            "dl": dr["dl"],
            "d": dr["d"] + (1 / self.fd_grid_v),
            "du": dr["du"],
        }
        # Laplacian
        self.lapl = {
            "dl": d2r["dl"] + (2 / self.fd_grid_v) * dr["dl"],
            "d": d2r["d"] + (2 / self.fd_grid_v) * dr["d"],
            "du": d2r["du"] + (2 / self.fd_grid_v) * dr["du"],
        }

        self.g_T0 = (
            np.sqrt(4 * np.pi)
            * tridiag_dot(
                self.grad["dl"],
                self.grad["d"],
                self.grad["du"],
                (self.ri * self.ro) / self.fd_grid_v - self.ri,
            )[:, -1]
        )

    def __fd_matrices__(self):
        dl = self.r_grid - np.roll(self.r_grid, 1)
        dr = np.roll(self.r_grid, -1) - self.r_grid
        dl = dl[1:-1, np.newaxis]
        dr = dr[1:-1, np.newaxis]

        dr_diags = {
            "dl": -dr / (dl * (dl + dr)),
            "d": (-dl + dr) / (dl * dr),
            "du": dl / (dr * (dl + dr)),
        }

        d2r_diags = {
            "dl": 2 / (dl * (dl + dr)),
            "d": -2 / (dl * dr),
            "du": 2 / (dr * (dl + dr)),
        }

        return (dr_diags, d2r_diags)

    def save(
        self,
        filename: str,
        time: float,
        theta_lm: np.ndarray,
        u_tor_lm: np.ndarray,
        u_pol_lm: np.ndarray,
    ):
        with h5py.File(filename, "w") as f:
            f.attrs["Ra"] = self.Ra
            f.attrs["E"] = self.E
            f.attrs["radius_ratio"] = self.ri / self.ro
            f.attrs["l_max"] = self.l_max
            f.attrs["m_max"] = self.m_max
            f.attrs["m_res"] = self.m_res
            f.attrs["n_r"] = self.n_r

            f.attrs["time"] = time

            f.create_dataset("theta_lm", data=np.array(theta_lm))
            f.create_dataset("u_tor_lm", data=np.array(u_tor_lm))
            f.create_dataset("u_pol_lm", data=np.array(u_pol_lm))

    def load(filename: str):
        with h5py.File(filename, "r") as f:
            eq = SphericalDynamo(
                Ra=f.attrs["Ra"].item(),
                E=f.attrs["E"].item(),
                radius_ratio=f.attrs["radius_ratio"].item(),
                l_max=f.attrs["l_max"].item(),
                m_max=f.attrs["m_max"].item(),
                m_res=f.attrs["m_res"].item(),
                n_r=f.attrs["n_r"].item(),
            )

            return (
                eq,
                f.attrs["time"].item(),
                np.array(f["theta_lm"]),
                np.array(f["u_tor_lm"]),
                np.array(f["u_pol_lm"]),
            )

    def __repr__(self):
        return (
            "SphericalMHD(\n"
            f"  Ra={self.Ra:.3e}, E={self.E:.3e}, Pr={self.Pr:.3e}"
            f"r=[{self.ri:.4f}, {self.ro:.4f}],\n"
            f"  spherical harmonics: n_lm={self.n_lm} "
            f"(l_max={self.l_max}, m_max={self.m_max}, m_res={self.m_res}),\n"
            f"  radial: n_r={self.n_r}\n"
            ")"
        )


def eq_system(
    eq: SphericalMHD,
    coef_dt: float,
    coef_lhs: float,
    coef_rhs: float,
):
    def __l_matrices__(
        l: int,
    ) -> Tuple:
        l2 = l * (l + 1)

        lapl_l = {
            "dl": eq.lapl["dl"],
            "d": eq.lapl["d"] - (l2 / eq.fd_grid_v**2),
            "du": eq.lapl["du"],
        }

        theta_lhs = {
            "dl": coef_lhs * lapl_l["dl"] / eq.Pr,
            "d": coef_dt + coef_lhs * lapl_l["d"] / eq.Pr,
            "du": coef_lhs * lapl_l["du"] / eq.Pr,
        }
        theta_rhs = {
            "dl": coef_rhs * lapl_l["dl"] / eq.Pr,
            "d": coef_dt + coef_rhs * lapl_l["d"] / eq.Pr,
            "du": coef_rhs * lapl_l["du"] / eq.Pr,
        }

        u_tor_lhs = {
            "dl": coef_lhs * lapl_l["dl"],
            "d": coef_dt + coef_lhs * lapl_l["d"],
            "du": coef_lhs * lapl_l["du"],
        }
        u_tor_rhs = {
            "dl": coef_rhs * lapl_l["dl"],
            "d": coef_dt + coef_rhs * lapl_l["d"],
            "du": coef_rhs * lapl_l["du"],
        }

        bilap_l = {
            "dll": lapl_l["dl"] * np.roll(lapl_l["dl"], 1),
            "dl": lapl_l["dl"] * np.roll(lapl_l["d"], 1) + lapl_l["dl"] * lapl_l["d"],
            "d": lapl_l["d"] ** 2
            + lapl_l["dl"] * np.roll(lapl_l["du"], 1)
            + lapl_l["du"] * np.roll(lapl_l["dl"], -1),
            "du": lapl_l["d"] * lapl_l["du"] + lapl_l["du"] * np.roll(lapl_l["d"], -1),
            "duu": lapl_l["du"] * np.roll(lapl_l["du"], -1),
        }

        lap_0 = (
            2
            / (eq.r_grid[1] - eq.r_grid[0]) ** 2
            * (eq.d0d2r + (2 / eq.r_grid[1]) * eq.d0dr)
        )
        lap_n = (
            2
            / (eq.r_grid[-2] - eq.r_grid[-1]) ** 2
            * (eq.dnd2r + (2 / eq.r_grid[-2]) * eq.dndr)
        )
        bilap_l["d"][0] += lap_0
        bilap_l["d"][-1] += lap_n

        u_pol_lhs = {
            "dll": -coef_lhs * bilap_l["dll"],
            "dl": -coef_dt * lapl_l["dl"] - coef_lhs * bilap_l["dl"],
            "d": -coef_dt * lapl_l["d"] - coef_lhs * bilap_l["d"],
            "du": -coef_dt * lapl_l["du"] - coef_lhs * bilap_l["du"],
            "duu": -coef_lhs * bilap_l["duu"],
        }

        u_pol_rhs = {
            "dll": -coef_rhs * bilap_l["dll"],
            "dl": -coef_dt * lapl_l["dl"] - coef_rhs * bilap_l["dl"],
            "d": -coef_dt * lapl_l["d"] - coef_rhs * bilap_l["d"],
            "du": -coef_dt * lapl_l["du"] - coef_rhs * bilap_l["du"],
            "duu": -coef_rhs * bilap_l["duu"],
        }

        return (
            theta_lhs,
            theta_rhs,
            u_tor_lhs,
            u_tor_rhs,
            u_pol_lhs,
            u_pol_rhs,
        )

    l_matrices = [__l_matrices__(l) for l in range(eq.l_max + 1)]
    return tuple(zip(*l_matrices))


def dynamical_solver(
    eq: SphericalMHD,
    solver: Scheme,
    source: Callable,
) -> Callable:
    # Courant–Friedrichs–Lewy condition (number).
    def __cfl__(u_tor_g: jnp.ndarray, u_pol_g: jnp.ndarray) -> float:
        u_tor2_max = jnp.max(u_tor_g**2, axis=0)
        u_pol2_max = jnp.max(u_tor_g**2, axis=0)

        return 0
        # dt_s = jnp.min(jnp.sqrt(eq.dx_ds2 / us2_max))
        # dt_p = jnp.min(jnp.sqrt(eq.dx_dp2 / up2_max))
        # return jnp.min(
        #    0.75 * jnp.array([jnp.finfo(jnp.float64).max, dt_s, dt_p])
        # )

    def stack_matrices(matrices_list):
        return jax.tree.map(
            lambda *xs: jnp.squeeze(jnp.stack(xs, axis=-1)), *matrices_list
        )

    (theta_lhs, theta_rhs, u_tor_lhs, u_tor_rhs, u_pol_lhs, u_pol_rhs) = (
        stack_matrices(sys)
        for sys in eq_system(eq, solver.coef_dt, solver.coef_lhs, solver.coef_rhs)
    )
    u_pol_lu = pentadiag_factor_lu(u_pol_lhs)

    # Implicit term
    def __implicit__(
        theta_lm: jnp.ndarray, u_tor_lm: jnp.ndarray, u_pol_lm: jnp.ndarray
    ) -> Tuple[jnp.ndarray]:
        return (
            tridiag_dot(
                theta_rhs["dl"][:, eq.l_map],
                theta_rhs["d"][:, eq.l_map],
                theta_rhs["du"][:, eq.l_map],
                theta_lm,
            ),
            tridiag_dot(
                u_tor_rhs["dl"][:, eq.l_map],
                u_tor_rhs["d"][:, eq.l_map],
                u_tor_rhs["du"][:, eq.l_map],
                u_tor_lm,
            ),
            pentadiag_dot(
                u_pol_rhs["dll"][:, eq.l_map],
                u_pol_rhs["dl"][:, eq.l_map],
                u_pol_rhs["d"][:, eq.l_map],
                u_pol_rhs["du"][:, eq.l_map],
                u_pol_rhs["duu"][:, eq.l_map],
                u_pol_lm,
            ),
        )

    # Explicit term
    def __explicit__(
        theta_lm: jnp.ndarray,
        u_tor_lm: jnp.ndarray,
        u_pol_lm: jnp.ndarray,
        source: Callable,
    ) -> Tuple[jnp.ndarray, float]:
        l2 = eq.l_map * (eq.l_map + 1)
        q_u = l2 * u_pol_lm * (1 / eq.fd_grid_v)
        s_u = tridiag_dot(eq.curl["dl"], eq.curl["d"], eq.curl["du"], u_pol_lm)
        t_w = -tridiag_dot(
            eq.lapl["dl"], eq.lapl["d"], eq.lapl["du"], u_pol_lm
        ) + l2 * (u_pol_lm / eq.fd_grid_v**2)
        q_w = l2 * u_tor_lm * (1 / eq.fd_grid_v)
        s_w = tridiag_dot(eq.curl["dl"], eq.curl["d"], eq.curl["du"], u_tor_lm)
        q_gt = tridiag_dot(eq.grad["dl"], eq.grad["d"], eq.grad["du"], theta_lm)
        s_gt = theta_lm / eq.fd_grid_v

        om_0 = 1 / eq.E
        om_z = 2 * om_0 * jnp.sqrt(4 * jnp.pi / 3)
        q_w = q_w.at[:, 1].add(om_z)
        s_w = s_w.at[:, 1].add(om_z)

        q_gt = q_gt.at[:, 0].add(eq.g_T0)

        def __cross__(ur, ut, up, wr, wt, wp):
            nlu_r = ut * wp - up * wt
            nlu_t = up * wr - ur * wp
            nlu_p = ur * wt - ut * wr
            return (nlu_r, nlu_t, nlu_p)

        ur, ut, up = shtns_synth(
            eq.sh,
            q_u.view(jnp.float64),
            s_u.view(jnp.float64),
            u_tor_lm.view(jnp.float64),
        )
        wr, wt, wp = shtns_synth(
            eq.sh, q_w.view(jnp.float64), s_w.view(jnp.float64), t_w.view(jnp.float64)
        )
        gtr, gtt, gtp = shtns_synth(
            eq.sh,
            q_gt.view(jnp.float64),
            s_gt.view(jnp.float64),
            jnp.zeros_like(s_gt.view(jnp.float64)),
        )

        nlu_r, nlu_t, nlu_p = __cross__(ur, ut, up, wr, wt, wp)
        nlt = -(ur * gtr + ut * gtt + up * gtp)
        q_gt = shtns_analys(eq.sh, nlt).view(jnp.complex128)
        q_w = shtns_analys(eq.sh, nlu_r).view(jnp.complex128)
        st_w = shtns_analys(eq.sh, nlu_t, nlu_p)
        s_w = st_w[0].view(jnp.complex128)
        t_w = st_w[1].view(jnp.complex128)

        p_w = (
            q_w / eq.fd_grid_v
            - tridiag_dot(eq.curl["dl"], eq.curl["d"], eq.curl["du"], s_w)
            + eq.Ra / (eq.E * eq.Pr * eq.ro) * theta_lm
        )

        c = None

        theta_source, u_tor_source, u_pol_source = source(theta_lm, u_tor_lm, u_pol_lm)
        return (
            q_gt + theta_source,
            t_w + u_tor_source,
            p_w + u_pol_source,
            c,
        )

    # Implicit solve
    def __solve__(
        rhs_theta: jnp.ndarray, rhs_u_tor: jnp.ndarray, rhs_u_pol: jnp.ndarray
    ) -> jnp.ndarray:
        return (
            tridiag_solve(
                theta_lhs["dl"][:, eq.l_map],
                theta_lhs["d"][:, eq.l_map],
                theta_lhs["du"][:, eq.l_map],
                rhs_theta,
            ),
            tridiag_solve(
                u_tor_lhs["dl"][:, eq.l_map],
                u_tor_lhs["d"][:, eq.l_map],
                u_tor_lhs["du"][:, eq.l_map],
                rhs_u_tor,
            ),
            pentadiag_solve_lu(
                u_pol_lu["dll"][:, eq.l_map],
                u_pol_lu["dl"][:, eq.l_map],
                u_pol_lu["d"][:, eq.l_map],
                u_pol_lu["du"][:, eq.l_map],
                u_pol_lu["duu"][:, eq.l_map],
                rhs_u_pol,
            ),
        )

    return solver(
        eq,
        source,
        __implicit__,
        __explicit__,
        __solve__,
    )


def boundary_layer_grid(
    n_r: int,
    ro: float,
    ri: float,
    n_blo: int,
    n_bli: int,
    dx_blo: float,
    dx_bli: float,
    slope: float = 1.1,
) -> Optional[np.ndarray]:
    r = np.zeros(n_r)
    r[0] = ri
    r[-1] = ro

    dx_ri = dx_bli * (slope - 1) / (slope**n_bli - 1)
    dx_ro = dx_blo * (slope - 1) / (slope**n_blo - 1)

    ii = 0
    io = n_r - 1
    while r[io] - r[ii] > min(dx_ri, dx_ro) * (io - ii):
        if ii + 1 >= io:
            return None

        if dx_ri <= dx_ro:
            r[ii + 1] = r[ii] + dx_ri
            dx_ri *= slope
            ii += 1
        else:
            r[io - 1] = r[io] - dx_ro
            dx_ro *= slope
            io -= 1

    dx_in = (r[io] - r[ii]) / (io - ii)
    r[ii:io] = np.arange(r[ii], r[io], step=dx_in)
    return r


def integral(eq: SphericalMHD, f_lm: np.ndarray, m: np.ndarray) -> float:
    f2 = f_lm.real**2 + f_lm.imag**2
    f2[:, m == 0] *= 0.5
    return np.sum(eq.fd_norm * f2)


def kinetic_energy(
    eq: SphericalMHD, u_tor_lm: np.ndarray, u_pol_lm: np.ndarray
) -> np.ndarray:
    l2 = eq.l_map * (eq.l_map + 1)

    u_tor2 = l2 * (u_tor_lm.real**2 + u_tor_lm.imag**2)
    u_rad_lm = l2 * u_pol_lm / eq.fd_grid_v
    u_hor_lm = tridiag_dot(eq.curl["dl"], eq.curl["d"], eq.curl["du"], u_pol_lm)
    u_pol2 = (
        l2 * (u_hor_lm.real**2 + u_hor_lm.imag**2) + u_rad_lm.real**2 + u_rad_lm.imag**2
    )

    return eq.fd_norm * (u_tor2 + u_pol2)
