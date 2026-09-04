<p align="center">
  <img src="https://github.com/hrkz/dshells/blob/main/graphical-abstract.png" alt="Repository Abstract" width="400"/>
</p>

> A differentiable simulation code for rotating incompressible flows and magnetic fields in spherical shells (currently limited to convection). Its goal is to reproduce the main equations solved by [xshells](https://nschaeff.bitbucket.io/xshells/) with an AD backend.

---

## 📦 Getting started

To setup and run the Python scripts and notebooks, we use [uv](https://docs.astral.sh/uv/) to manage the package dependencies in a custom environment

1. **Prepare dependencies**

Download the [shtns](https://gricad-gitlab.univ-grenoble-alpes.fr/schaeffn/shtns/-/tree/jax-gpu?ref_type=heads) library inside the `dshells/` folder and switch to the `jax-gpu` branch.

2. **Setup the environment**

```bash
cd dshells
uv sync
```

Note: the code can also run on CPU with minor changes.

---

## 🗂️ Notebooks & scripts

### Parameter calibration — `notebooks/calibration.ipynb` · `scripts/calibration.py`

Recovers the rotating Rayleigh number $Ra$, Ekman number $E$ and Prandtl number $Pr$ from trajectory observations by differentiating end-to-end through the time integrator.

**Setup.** The simulation state at time step $n$ is 

$$x_n = (\theta_{\ell m}, u^{t}_{\ell m}, u^{p}_{\ell m}),$$ 

the spectral coefficients of temperature and velocity. A single IMEX step defines the discrete dynamical system

$$x_{n+1} = F(x_n; Ra, E, Pr).$$

**Objective.** Given a reference trajectory $x_n^T$ and a simulated trajectory $\hat{x}_n$ unrolled from the same initial condition $x_0$, the loss measures the kinetic energy of the velocity mismatch at each step:

$$L(Ra, E, Pr) = \sum_{n=1}^{T} E_k(\hat{u}_n - u_n^T),$$

where $E_k(u) = \frac{1}{2}\int |u|^2 dV$ is computed in spectral space.

**Initialisation.** Starting values are drawn from uniform distributions with a fixed seed:

$$Ra_0 \sim U(10\, 50), \qquad E_0 \sim U(10^{-2}\, 10^{-1}), \qquad Pr_0 \sim U(10^{-1}, 3).$$

**Differentiability.** The trajectory unroll is expressed as a `jax.lax.scan` over $F$, making $L$ a pure JAX expression. Gradients $\partial L / \partial Ra$, $\partial L / \partial E$ and $\partial L / \partial Pr$ are obtained by reverse-mode AD through the entire sequence of time steps — no adjoint solver is implemented manually. The parameters are stored in log-space ($p_{Ra} = \log Ra$, $p_{E} = \log E$, $p_{Pr} = \log Pr$) to enforce positivity, and updated with Adam via Optax.

The calibration requires a pre-generated sample dataset. Use `_sample.py` to produce it:

```bash
uv run python _sample.py \
    --config calibration \
    -Ra 100 -E 1e-3 -Pr 1 -rr 0.35 \
    -n_r 84 -l_max 31 -m_max 7 -m_res 4 \
    -dt 4e-4 -spinup_time 0.4 -steps 25 -samples 250 
```

This runs a spinup from a perturbed initial condition and saves state snapshots `(θ, u_tor, u_pol)` to `data/<config>/`.

---

## 🖥️ Running on clusters

It is required to load modules related to local installations of `FFTW` and `CUDA`.
