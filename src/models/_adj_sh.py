import ctypes
import functools

import jax
import jax.numpy as jnp

import shtns

shtns_jax_lib = ctypes.cdll.LoadLibrary(
    shtns._shtns.__file__
)  # load aditional symbols from _shtns module extension

for shtns_fn in (
    "synth",
    "analys",
    "vector_synth",
    "vector_analys",
    "vector3_synth",
    "vector3_analys",
):
    cpu_fn = shtns_fn + "_cpu"
    gpu_fn = shtns_fn + "_gpu"
    jax.ffi.register_ffi_target(
        "shtns_" + cpu_fn,
        jax.ffi.pycapsule(getattr(shtns_jax_lib, cpu_fn)),
        platform="cpu",
    )
    jax.ffi.register_ffi_target(
        "shtns_" + gpu_fn,
        jax.ffi.pycapsule(getattr(shtns_jax_lib, gpu_fn)),
        platform="gpu",
    )


def _shtns_synth_jax(self, *args, **kwargs):
    """
    Unified synthesis interface that handles scalar, vector, and 3-component vector transforms.

    Args:
        *args: 1, 2, or 3 input arrays (qlm, [slm, tlm])
            - 1 arg: scalar transform (qlm -> q)
            - 2 args: vector transform (slm, tlm -> vt, vp)
            - 3 args: 3-component vector transform (qlm, slm, tlm -> vr, vt, vp)

    Returns:
        Tuple of output arrays matching the transform type
    """
    x = args[0]
    out_shape = (
        (self.nlat, self.nphi)
        if len(x.shape) == 1
        else (*x.shape[:-1], self.nlat, self.nphi)
    )
    adjoint = 1 if kwargs.get("adjoint", False) else 0

    for a in args:
        if a.dtype != jnp.float64:
            raise ValueError("Only the float64 dtype is implemented by shtns")
        if a.shape != x.shape:
            raise ValueError("shape mismatch")

    # Validate input arguments
    if len(args) == 1:
        # Scalar transform: qlm -> q
        def get_impl(target_name, adjoint=0):
            return lambda x: jax.ffi.ffi_call(
                target_name,
                jax.ShapeDtypeStruct(out_shape, jnp.float64),
                vmap_method="broadcast_all",
            )(x, cfg=int(self.this) + adjoint)

        return jax.lax.platform_dependent(
            x,
            cpu=get_impl("shtns_synth_cpu", adjoint),
            cuda=get_impl("shtns_synth_gpu", adjoint),
        )

    elif len(args) == 2:
        # Vector transform: slm, tlm -> vt, vp
        slm, tlm = args

        # For functions with multiple outputs, we need a tuple of ShapeDtypeStruct
        out_spec = (jax.ShapeDtypeStruct(out_shape, jnp.float64),) * 2

        def get_impl(target_name, adjoint=0):
            return lambda slm, tlm: jax.ffi.ffi_call(
                target_name, out_spec, vmap_method="broadcast_all"
            )(slm, tlm, cfg=int(self.this) + adjoint)

        return jax.lax.platform_dependent(
            slm,
            tlm,
            cpu=get_impl("shtns_vector_synth_cpu", adjoint),
            cuda=get_impl("shtns_vector_synth_gpu", adjoint),
        )

    elif len(args) == 3:
        # 3-component vector transform: qlm, slm, tlm -> vr, vt, vp
        qlm, slm, tlm = args

        # For functions with 3 outputs, we need a tuple of ShapeDtypeStruct
        out_spec = (jax.ShapeDtypeStruct(out_shape, jnp.float64),) * 3

        def get_impl(target_name, adjoint=0):
            return lambda qlm, slm, tlm: jax.ffi.ffi_call(
                target_name, out_spec, vmap_method="broadcast_all"
            )(qlm, slm, tlm, cfg=int(self.this) + adjoint)

        return jax.lax.platform_dependent(
            qlm,
            slm,
            tlm,
            cpu=get_impl("shtns_vector3_synth_cpu", adjoint),
            cuda=get_impl("shtns_vector3_synth_gpu", adjoint),
        )
    else:
        raise ValueError(f"synth_jax expects 1, 2, or 3 arguments, got {len(args)}")


def _shtns_analys_jax(self, *args, **kwargs):
    """
    Unified analysis interface that handles scalar, vector, and 3-component vector transforms.

    Args:
        *args: 1, 2, or 3 input arrays (q, [vt, vp], [vr, vt, vp])
            - 1 arg: scalar transform (q -> qlm)
            - 2 args: vector transform (vt, vp -> slm, tlm)
            - 3 args: 3-component vector transform (vr, vt, vp -> qlm, slm, tlm)

    Returns:
        Tuple of output arrays matching the transform type
    """
    x = args[0]
    # Input is spatial (vr, vt, vp), output is spectral (qlm, slm, tlm)
    # For batched inputs: (batch..., nlat, nphi) -> (batch..., nlm*2)
    out_shape = (self.nlm * 2,) if len(x.shape) == 2 else (*x.shape[:-2], self.nlm * 2)
    adjoint = 1 if kwargs.get("adjoint", False) else 0

    for a in args:
        if a.dtype != jnp.float64:
            raise ValueError("Only the float64 dtype is implemented by shtns")
        if a.shape != x.shape:
            raise ValueError("shape mismatch")

    # Validate input arguments
    if len(args) == 1:
        # Scalar transform: q -> qlm

        def get_impl(target_name, adjoint=0):
            return lambda x: jax.ffi.ffi_call(
                target_name,
                jax.ShapeDtypeStruct(out_shape, jnp.float64),
                vmap_method="broadcast_all",
            )(x, cfg=int(self.this) + adjoint)

        return jax.lax.platform_dependent(
            x,
            cpu=get_impl("shtns_analys_cpu", adjoint),
            cuda=get_impl("shtns_analys_gpu", adjoint),
        )

    elif len(args) == 2:
        # Vector transform: vt, vp -> slm, tlm
        vt, vp = args

        # For functions with multiple outputs, we need a tuple of ShapeDtypeStruct
        out_spec = (jax.ShapeDtypeStruct(out_shape, jnp.float64),) * 2

        def get_impl(target_name, adjoint=0):
            return lambda vt, vp: jax.ffi.ffi_call(
                target_name, out_spec, vmap_method="broadcast_all"
            )(vt, vp, cfg=int(self.this) + adjoint)

        return jax.lax.platform_dependent(
            vt,
            vp,
            cpu=get_impl("shtns_vector_analys_cpu", adjoint),
            cuda=get_impl("shtns_vector_analys_gpu", adjoint),
        )

    elif len(args) == 3:
        # 3-component vector transform: vr, vt, vp -> qlm, slm, tlm
        vr, vt, vp = args

        # For functions with 3 outputs, we need a tuple of ShapeDtypeStruct
        out_spec = (jax.ShapeDtypeStruct(out_shape, jnp.float64),) * 3

        def get_impl(target_name, adjoint=0):
            return lambda vr, vt, vp: jax.ffi.ffi_call(
                target_name, out_spec, vmap_method="broadcast_all"
            )(vr, vt, vp, cfg=int(self.this) + adjoint)

        return jax.lax.platform_dependent(
            vr,
            vt,
            vp,
            cpu=get_impl("shtns_vector3_analys_cpu", adjoint),
            cuda=get_impl("shtns_vector3_analys_gpu", adjoint),
        )
    else:
        raise ValueError(f"analys_jax expects 1, 2, or 3 arguments, got {len(args)}")


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def shtns_synth(sh, *args):
    """
    Unified synthesis transform with custom VJP for all transform types.

    Args:
        sh: shtns configuration object
        *args: 1, 2, or 3 input arrays (qlm, [slm, tlm])

    Returns:
        Tuple of output arrays matching the transform type
    """
    return sh.synth_jax(*args)


def shtns_synth_fwd(sh, *args):
    """Forward pass for all transform types."""
    return sh.synth_jax(*args), ()  # no residuals for linear functions


def shtns_synth_bwd(sh, residuals, g):
    """Backward pass - adjoint is the analysis transform, with normalization."""
    # Handle scalar case where g is a single array, not a tuple
    if isinstance(g, tuple):
        return sh.analys_jax(*g, adjoint=True)
    else:
        return (sh.analys_jax(g, adjoint=True),)  # return a tuple


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def shtns_analys(sh, *args):
    """
    Unified analysis transform with custom VJP for all transform types.

    Args:
        sh: shtns configuration object
        *args: 1, 2, or 3 input arrays: vr, (vr,vt), (vr,vt,vp)

    Returns:
        Tuple of output arrays matching the transform type
    """
    return sh.analys_jax(*args)


def shtns_analys_fwd(sh, *args):
    """Forward pass for all transform types."""
    return sh.analys_jax(*args), ()  # no residuals for linear functions


def shtns_analys_bwd(sh, residuals, g):
    """Backward pass - adjoint is the synthesis transform, with normalization."""
    # Handle scalar case where g is a single array, not a tuple
    if isinstance(g, tuple):
        return sh.synth_jax(*g, adjoint=True)
    else:
        return (sh.synth_jax(g, adjoint=True),)  # return a tuple


shtns_synth.defvjp(shtns_synth_fwd, shtns_synth_bwd)
shtns_analys.defvjp(shtns_analys_fwd, shtns_analys_bwd)


shtns.sht.synth_jax = _shtns_synth_jax
shtns.sht.analys_jax = _shtns_analys_jax
