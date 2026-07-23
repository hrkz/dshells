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
