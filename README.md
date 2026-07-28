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

**Setup.** The simulation state at time step $n$ is 

$$x_n = (\theta_{\ell m}, u^{t}_{\ell m}, u^{p}_{\ell m}),$$ 

the spectral coefficients of temperature and velocity. A single IMEX step defines the discrete dynamical system

$$x_{n+1} = F(x_n; Ra, E).$$

**Objective.** Given a reference trajectory $x_n^T$ and a simulated trajectory $\hat{x}_n$ unrolled from the same initial condition $x_0$, the loss measures the kinetic energy of the velocity mismatch at each step:

$$L(Ra, E) = \sum_{n=1}^{T} E_k(\hat{u}_n - u_n^T),$$

where $E_k(u) = \frac{1}{2}\int |u|^2 dV$ is computed in spectral space.

**Initialisation.** Starting values are drawn from uniform distributions with a fixed seed:

$$Ra_0 \sim U(10\, 50), \qquad E_0 \sim U(10^{-2}\, 10^{-1}).$$

**Differentiability.** The trajectory unroll is expressed as a `jax.lax.scan` over $F$, making $L$ a pure JAX expression. Gradients $\partial L / \partial Ra$ and $\partial L / \partial E$ are obtained by reverse-mode AD through the entire sequence of time steps — no adjoint solver is implemented manually. The parameters are stored in log-space ($p = \log Ra$, $q = \log E$) to enforce positivity, and updated with Adam via Optax.

The calibration requires a pre-generated sample dataset. Use `_sample.py` to produce it:

```bash
uv run python _sample.py \
    --config calibration \
    -Ra 100 -E 1e-3 -rr 0.35 \
    -n_r 84 -l_max 31 -m_max 7 -m_res 4 \
    -dt 4e-4 -spinup_time 0.4 -steps 25 -samples 250 
```

This runs a spinup from a perturbed initial condition and saves state snapshots `(θ, u_tor, u_pol)` to `data/<config>/`.

**Parameters**

| Flag | Type | Required | Default | Description |
|---|---|---|---|---|
| `-c`, `--config` | `str` | yes | — | Name of the configuration (used as the output sub-directory under `data/`) |
| `-Ra` | `float` | yes | — | Rayleigh number |
| `-E` | `float` | yes | — | Ekman number |
| `-rr` | `float` | no | `0.35` | Ratio between inner and outer spherical radius |
| `-n_r` | `int` | yes | — | Number of radial grid points |
| `-l_max` | `int` | yes | — | Maximum degree of spherical harmonics |
| `-m_max` | `int` | yes | — | Maximum Fourier mode ($\phi$) |
| `-m_res` | `int` | no | `1` | $\phi$-periodicity (1 for full sphere, 2 for half sphere, etc.) |
| `-dt` | `float` | yes | — | Discrete (fixed) time step |
| `-spinup_time` | `float` | yes | — | Physical time for the spinup phase |
| `-steps` | `int` | yes | — | Number of time steps per sub-trajectory sample |
| `-samples` | `int` | yes | — | Total number of samples to save in the dataset |

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
