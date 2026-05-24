# InformationDynamics-GeneralRelativity

This repository contains the numerical verification code and experimental data for the paper **"Deriving General Relativity from Information Dynamics: Direct Projection from the Generalized Ginzburg-Landau Equation to the Einstein Field Equations"** (Huang, 2026).
[preprint: https://doi.org/10.5281/zenodo.20367410]

## Repository Structure

| File | Description |
|------|-------------|
| `black_hole_steady_state.py` | Schwarzschild black hole steady-state solver via information free energy minimization |
| `gravitational_wave_simulation.py` | Gravitational wave propagation solver (wave equation, speed of light verification) |
| `dark_energy_simulation.py` | Cosmic accelerated expansion simulation with logarithmic barrier potential |
| `black_hole_steady_state.png` | Output figure: metric components and volume element |
| `gravitational_wave_propagation.png` | Output figure: wave packet evolution and spacetime diagram |
| `cosmic_acceleration.png` | Output figure: scale factor, deceleration parameter, field amplitude |
| `*.log.txt` | Console output logs for each experiment |

## Numerical Experiments

### 1. Black Hole Steady State

**File:** `black_hole_steady_state.py`

**Method:** Free energy minimization with L‑BFGS‑B  
**Variables:** u = ln(-g_tt), v = ln(g_rr)  
**Constraints:** Volume element -g_tt·g_rr = 1 (enforced via (u+v)² penalty)  
**Domain:** r ∈ [2.01, 20.0], N = 500  
**Output:** Metric components g_tt(r), g_rr(r) compared to Schwarzschild exact solution

**Key result:** Volume element MSE = 5.84 × 10⁻⁵, confirming that -g_tt·g_rr = 1 is satisfied with high precision.

### 2. Gravitational Wave Propagation

**File:** `gravitational_wave_simulation.py`

**Method:** Pseudospectral (FFT) + 4th‑order Runge‑Kutta  
**Initial condition:** Gaussian wave packet h(x,0) = exp(-(x - x₀)²/2σ²), ∂_t h(x,0) = 0  
**Domain:** L = 100, N = 1024, dt = 0.005, T_max = 30.0  
**Measurement:** Peak position shift → propagation speed

**Key result:** Measured speed c_meas = 1.0002 (natural units), deviation < 0.1% from c = 1.

### 3. Cosmic Accelerated Expansion

**File:** `dark_energy_simulation.py`

**Method:** Adaptive Euler integration of coupled Friedmann + Klein‑Gordon equations  
**Potential:** Logarithmic barrier V(φ) = κ|φ| ln(1 + |φ|), κ = 0.5  
**Initial conditions:** a(0) = 0.1, φ(0) = 0.5, φ̇(0) = 0, H(0) = 0.5  
**Integration:** T_max = 50.0, adaptive dt

**Key result:** Deceleration parameter q becomes negative (≈ -0.8), indicating late‑time accelerated expansion driven solely by the logarithmic barrier potential.

## Requirements

```bash
pip install numpy scipy matplotlib
```

## Citation

If you use this code or refer to the theoretical framework in your work, please cite:

```
@article{huang2026infodyn_gr,
  title={Deriving General Relativity from Information Dynamics: Direct Projection from the Generalized Ginzburg-Landau Equation to the Einstein Field Equations},
  author={Huang, Kai},
  year={2026},
  note={Numerical experiments and code available at \url{https://github.com/hkaiopen/InformationDynamics-GeneralRelativity}}
}
```
## Related Work

The same information dynamics framework has been applied to:

- **Kakeya conjecture** — Generalized Ginzburg‑Landau construction of Kakeya sets (Huang & Liu, 2026) [preprint: https://doi.org/10.5281/zenodo.19542718]
- **Riemann hypothesis** — Dynamic real‑imaginary coupling to prime density generation (Huang, 2026) [preprint: https://doi.org/10.5281/zenodo.20054082]
- **Computational biology** — Real‑imaginary duality principle for DNA sequencing and RNA inverse folding (Liu & Huang, 2026) [preprint: https://doi.org/10.5281/zenodo.20057468]
