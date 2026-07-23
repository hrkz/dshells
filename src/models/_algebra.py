import jax
import jax.numpy as jnp


def tridiag_dot(
    dl: jnp.ndarray, d: jnp.ndarray, du: jnp.ndarray, x: jnp.ndarray
) -> jnp.ndarray:
    return (  # diags
        d * x + dl * jnp.roll(x, 1, axis=0) + du * jnp.roll(x, -1, axis=0)
    )


def tridiag_solve(
    dl: jnp.ndarray, d: jnp.ndarray, du: jnp.ndarray, b: jnp.ndarray
) -> jnp.ndarray:
    x_real = jax.lax.linalg.tridiagonal_solve(
        jnp.moveaxis(dl, 0, -1),
        jnp.moveaxis(d, 0, -1),
        jnp.moveaxis(du, 0, -1),
        jnp.expand_dims(jnp.moveaxis(b.real, 0, -1), -1),
    )
    x_imag = jax.lax.linalg.tridiagonal_solve(
        jnp.moveaxis(dl, 0, -1),
        jnp.moveaxis(d, 0, -1),
        jnp.moveaxis(du, 0, -1),
        jnp.expand_dims(jnp.moveaxis(b.imag, 0, -1), -1),
    )
    x = x_real + 1j * x_imag
    return jnp.moveaxis(x[..., -1], -1, 0)


def pentadiag_dot(
    dll: jnp.ndarray,
    dl: jnp.ndarray,
    d: jnp.ndarray,
    du: jnp.ndarray,
    duu: jnp.ndarray,
    x: jnp.ndarray,
) -> jnp.ndarray:
    return (  # diags
        d * x
        + dl * jnp.roll(x, 1, axis=0)
        + du * jnp.roll(x, -1, axis=0)
        + dll * jnp.roll(x, 2, axis=0)
        + duu * jnp.roll(x, -2, axis=0)
    )


def pentadiag_factor_lu(A):
    dll, dl, d, du, duu = (A[k] for k in ("dll", "dl", "d", "du", "duu"))

    # row 0
    d = d.at[0].set(1.0 / d[0])

    # row 1
    t = dl[1]
    p = d[1] - t * d[0] * du[0]
    du = du.at[1].set(du[1] - t * d[0] * duu[0])
    dl = dl.at[1].set(t)
    d = d.at[1].set(1.0 / p)

    def row_step(carry, j):
        dll, dl, d, du, duu = carry

        alpha = dll[j] * d[j - 2]
        t = dl[j] - alpha * du[j - 2]
        p = d[j] - alpha * duu[j - 2]

        beta = t * d[j - 1]
        p = p - beta * du[j - 1]

        du = du.at[j].set(du[j] - beta * duu[j - 1])
        dl = dl.at[j].set(t)
        d = d.at[j].set(1.0 / p)

        return ((dll, dl, d, du, duu), None)

    # rows 2, ..., n
    n = d.shape[0]
    (dll, dl, d, du, duu), _ = jax.lax.scan(
        row_step, (dll, dl, d, du, duu), jnp.arange(2, n)
    )

    return {
        "dll": dll * d,
        "dl": dl * d,
        "d": d,
        "du": du * d,
        "duu": duu * d,
    }


def pentadiag_solve_lu(
    dll: jnp.ndarray,
    dl: jnp.ndarray,
    d: jnp.ndarray,
    du: jnp.ndarray,
    duu: jnp.ndarray,
    b: jnp.ndarray,
) -> jnp.ndarray:
    n = b.shape[0]
    x = b.copy()

    # forward
    def fwd_step(x, j):
        return (
            x.at[j].set(d[j] * x[j] - dl[j] * x[j - 1] - dll[j] * x[j - 2]),
            None,
        )

    x = x.at[0].mul(d[0])
    x = x.at[1].set(d[1] * x[1] - dl[1] * x[0])
    x, _ = jax.lax.scan(fwd_step, x, jnp.arange(2, n))

    # backward
    def bwd_step(x, j):
        return (
            x.at[j].add(-du[j] * x[j + 1] - duu[j] * x[j + 2]),
            None,
        )

    x = x.at[n - 2].add(-du[n - 2] * x[n - 1])
    x, _ = jax.lax.scan(bwd_step, x, jnp.arange(n - 3, -1, -1))
    return x
