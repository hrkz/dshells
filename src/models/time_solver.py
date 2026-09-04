from typing import Callable, List, Tuple

import jax
import jax.numpy as jnp


class Scheme:
    def __call__(
        self,
        eq,
        system: jnp.ndarray,
        source: Callable,
        implicit: Callable,
        explicit: Callable,
        solve: Callable,
    ) -> Callable:
        pass


class PC2(Scheme):
    def __init__(self, dt: float):
        self.dt = dt
        self.coef_dt = 1 / self.dt
        self.coef_lhs = -0.5
        self.coef_rhs = 0.5

    def __call__(
        self,
        eq,
        source: Callable,
        implicit: Callable,
        explicit: Callable,
        solve: Callable,
    ):
        def __imex_step__(
            theta_lm: jnp.ndarray,
            u_tor_lm: jnp.ndarray,
            u_pol_lm: jnp.ndarray,
        ) -> Tuple[float, jnp.ndarray]:

            theta_i_0, u_tor_i_0, u_pol_i_0 = implicit(theta_lm, u_tor_lm, u_pol_lm)
            theta_e_0, u_tor_e_0, u_pol_e_0, c = explicit(
                theta_lm, u_tor_lm, u_pol_lm, source
            )

            theta_lm = theta_e_0 + theta_i_0
            u_tor_lm = u_tor_e_0 + u_tor_i_0
            u_pol_lm = u_pol_e_0 + u_pol_i_0

            theta_lm, u_tor_lm, u_pol_lm = solve(theta_lm, u_tor_lm, u_pol_lm)

            theta_e_1, u_tor_e_1, u_pol_e_1, _ = explicit(
                theta_lm, u_tor_lm, u_pol_lm, source
            )

            theta_lm = (0.5 * theta_e_0 + 0.5 * theta_e_1) + theta_i_0
            u_tor_lm = (0.5 * u_tor_e_0 + 0.5 * u_tor_e_1) + u_tor_i_0
            u_pol_lm = (0.5 * u_pol_e_0 + 0.5 * u_pol_e_1) + u_pol_i_0

            theta_lm, u_tor_lm, u_pol_lm = solve(theta_lm, u_tor_lm, u_pol_lm)

            theta_e_2, u_tor_e_2, u_pol_e_2, _ = explicit(
                theta_lm, u_tor_lm, u_pol_lm, source
            )

            theta_lm = (0.5 * theta_e_0 + 0.5 * theta_e_2) + theta_i_0
            u_tor_lm = (0.5 * u_tor_e_0 + 0.5 * u_tor_e_2) + u_tor_i_0
            u_pol_lm = (0.5 * u_pol_e_0 + 0.5 * u_pol_e_2) + u_pol_i_0

            theta_lm, u_tor_lm, u_pol_lm = solve(theta_lm, u_tor_lm, u_pol_lm)
            return (c, theta_lm, u_tor_lm, u_pol_lm)

        return __imex_step__


class BPR353(Scheme):
    def __init__(self, dt: float):
        self.dt = dt

    def coef_dt(self) -> float:
        return 1 / self.dt

    def coef_lhs(self) -> float:
        return -0.5

    def coef_rhs(self) -> float:
        return 0.5

    def __call__(
        self,
        eq,
        source: Callable,
        implicit: Callable,
        explicit: Callable,
        solve: Callable,
    ):
        # TODO: does not work
        def __imex_step__(
            theta_lm: jnp.ndarray,
            u_tor_lm: jnp.ndarray,
            u_pol_lm: jnp.ndarray,
        ) -> Tuple[float, jnp.ndarray]:
            theta_e_0, u_tor_e_0, u_pol_e_0, c = explicit(
                theta_lm, u_tor_lm, u_pol_lm, None
            )
            theta_i_0, u_tor_i_0, u_pol_i_0 = implicit(theta_lm, u_tor_lm, u_pol_lm)

            theta_lm = theta_e_0 + theta_i_0
            u_tor_lm = u_tor_e_0 + u_tor_i_0
            u_pol_lm = u_pol_e_0 + u_pol_i_0

            theta_lm, u_tor_lm, u_pol_lm = solve(theta_lm, u_tor_lm, u_pol_lm)

            theta_e_1, u_tor_e_1, u_pol_e_1, _ = explicit(
                theta_lm, u_tor_lm, u_pol_lm, None
            )
            (
                theta_i_1,
                u_tor_i_1,
                u_pol_i_1,
            ) = implicit(theta_lm, u_tor_lm, u_pol_lm)

            theta_lm = (4.9 / 9.0 * theta_e_0 + 2.0 / 9.0 * theta_e_1) + (
                5.0 / 18.0 * theta_i_0 - 1.0 / 9.0 * theta_i_1
            )
            u_tor_lm = (4.9 / 9.0 * u_tor_e_0 + 2.0 / 9.0 * u_tor_e_1) + (
                5.0 / 18.0 * u_tor_i_0 - 1.0 / 9.0 * u_tor_i_1
            )
            u_pol_lm = (4.9 / 9.0 * u_pol_e_0 + 2.0 / 9.0 * u_pol_e_1) + (
                5.0 / 18.0 * u_pol_i_0 - 1.0 / 9.0 * u_pol_i_1
            )

            theta_lm, u_tor_lm, u_pol_lm = solve(theta_lm, u_tor_lm, u_pol_lm)

            theta_e_2, u_tor_e_2, u_pol_e_2, _ = explicit(
                theta_lm, u_tor_lm, u_pol_lm, None
            )
            (
                theta_i_2,
                u_tor_i_2,
                u_pol_i_2,
            ) = implicit(theta_lm, u_tor_lm, u_pol_lm)

            theta_lm = (0.25 * theta_e_0 + 0.75 * theta_e_2) + 0.5 * theta_i_0
            u_tor_lm = (0.25 * u_tor_e_0 + 0.75 * u_tor_e_2) + 0.5 * u_tor_i_0
            u_pol_lm = (0.25 * u_pol_e_0 + 0.75 * u_pol_e_2) + 0.5 * u_pol_i_0

            theta_lm, u_tor_lm, u_pol_lm = solve(theta_lm, u_tor_lm, u_pol_lm)

            (
                theta_i_3,
                u_tor_i_3,
                u_pol_i_3,
            ) = implicit(theta_lm, u_tor_lm, u_pol_lm)

            theta_lm = (0.25 * theta_e_0 + 0.75 * theta_e_2) + 2 * (
                0.25 * theta_i_0 + 0.75 * theta_i_2 - theta_i_3
            )
            u_tor_lm = (0.25 * u_tor_e_0 + 0.75 * u_tor_e_2) + 2 * (
                0.25 * u_tor_i_0 + 0.75 * u_tor_i_2 - u_tor_i_3
            )
            u_pol_lm = (0.25 * u_pol_e_0 + 0.75 * u_pol_e_2) + 2 * (
                0.25 * u_pol_i_0 + 0.75 * u_pol_i_2 - u_pol_i_3
            )

            theta_lm, u_tor_lm, u_pol_lm = solve(theta_lm, u_tor_lm, u_pol_lm)
            return (c, theta_lm, u_tor_lm, u_pol_lm)

        return __imex_step__
