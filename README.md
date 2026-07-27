<p align="center">
  <img src="https://github.com/hrkz/dshells/blob/main/graphical-abstract.png" alt="Repository Abstract" width="400"/>
</p>

> A differentiable simulation code for rotating incompressible flows and magnetic fields in spherical shells (currently limited to convection). Its goal is to reproduce the core equations solved by xshells (https://nschaeff.bitbucket.io/xshells/) with an AD backend.

---

## 📦 Getting started

To setup and run the Python scripts and notebooks, we use [uv](https://docs.astral.sh/uv/) to manage the package dependencies in a custom environment

1. **Setup the environment**

```bash
cd dshells
uv sync
```

Note: the code is currently limited to CPU.

---

## 🗂️ Notebooks & scripts

### Forward experiment — `notebooks/experiment.ipynb` · `scripts/experiment.py`

Integrates the MHD equations forward in time from a prescribed initial condition (a single spherical harmonic perturbation in temperature). Tracks kinetic energy and buoyancy flux over 1000 time steps and exports the final temperature field as a VTK file (`conv-experiment.vts`) for 3-D visualisation.

### Parameter calibration — `notebooks/calibration.ipynb` · `scripts/calibration.py`

Recovers the Rayleigh number $Ra$ and Ekman number $E$ from trajectory observations by differentiating end-to-end through the time integrator.

**Setup.** The simulation state at time step $n$ is $x_n = (\theta_{\ell m}, u^{\mathrm{tor}}_{\ell m}, u^{\mathrm{pol}}_{\ell m})$, the spectral coefficients of temperature and velocity. A single IMEX step defines the discrete dynamical system

$$x_{n+1} = F(x_n;\, Ra, E).$$

**Objective.** Given a reference trajectory $\{x_n\}_{n=1}^{T}$ and a simulated trajectory $\{\hat{x}_n\}_{n=1}^{T}$ unrolled from the same initial condition $x_0$, the loss measures the kinetic energy of the velocity mismatch at each step:

$$L(Ra, E) = \sum_{n=1}^{T} E_k(\hat{u}_n - u_n),$$

where $E_k(u) = \frac{1}{2}\int |u|^2\, dV$ is computed in spectral space.

**Initialisation.** Starting values are drawn from uniform distributions with a fixed seed:

$$Ra_0 \sim U(10,\, 50), \qquad E_0 \sim U(10^{-2},\, 10^{-1}).$$

**Differentiability.** The trajectory unroll is expressed as a `jax.lax.scan` over $F$, making $L$ a pure JAX expression. Gradients $\partial L / \partial Ra$ and $\partial L / \partial E$ are obtained by reverse-mode AD through the entire sequence of time steps — no adjoint solver is implemented manually. The parameters are stored in log-space ($p = \log Ra$, $q = \log E$) to enforce positivity, and updated with Adam via Optax.

The calibration requires a pre-generated sample dataset. Use `_sample.py` to produce it:

```bash
uv run python _sample.py \
    --config calibration \
    -Ra 100 -E 1e-3 -rr 0.35 \
    -n_r 84 -l_max 31 -m_max 7 -m_res 4 \
    -dt 4e-4 -spinup_time 1.0 -steps 10 -samples 500
```

This runs a spinup from a perturbed initial condition and saves state snapshots `(θ, u_tor, u_pol)` to `data/<config>/`.

---

## 🖥️ Running on IPGP's DANTE cluster (SLURM)

Ready-to-submit job scripts are provided under `scripts/`. They load the required FFTW module and invoke `uv run` so no manual environment activation is needed.

```bash
# Submit data generation, then calibration once it finishes
JID=$(sbatch scripts/sample.slurm)
sbatch --dependency=afterok:$JID scripts/calibration.slurm

# Or run the forward experiment independently
sbatch scripts/experiment.slurm
```
