"""
Experiment: Black Hole Steady State (High‑Accuracy Version)
============================================================
- Radial domain: r_min = 2.01, r_max = 20.0
- Variables: u = ln(-g_tt), v = ln(g_rr)
- Boundaries fixed to Schwarzschild values
- Enhanced volume constraint + quadratic penalty on v
- L-BFGS-B optimization
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ====================== Parameters ======================
N = 500                     # Number of grid points (high accuracy)
M = 1.0                     # Black hole mass
r_min = 2.01                # Slightly outside the horizon
r_max = 20.0
r = np.linspace(r_min, r_max, N)
dr = r[1] - r[0]

lambda_vol = 200.0          # Strength of volume constraint
lambda_v = 5.0              # Quadratic penalty on v (prevents excessively large g_rr)

# Schwarzschild solution (target)
g_tt_sch = -(1.0 - 2.0 * M / r)
g_rr_sch = 1.0 / (1.0 - 2.0 * M / r)
u_sch = np.log(-g_tt_sch)
v_sch = np.log(g_rr_sch)

# Initial guess: Schwarzschild + small noise
np.random.seed(42)
u0 = u_sch + 0.02 * np.random.randn(N)
v0 = v_sch + 0.02 * np.random.randn(N)

# Fix boundaries (both ends are not optimized)
free_indices = np.arange(1, N-1)
n_free = len(free_indices)
x0 = np.concatenate([u0[free_indices], v0[free_indices]])

# Variable bounds (loose, only to prevent overflow)
bounds = [(-10.0, 0.0)] * n_free + [(-5.0, 7.0)] * n_free

# ====================== Free Energy and Analytical Gradient ======================
def compute_energy_and_grad(x):
    u = u_sch.copy()
    v = v_sch.copy()
    u[free_indices] = x[:n_free]
    v[free_indices] = x[n_free:]

    # Curvature term: squared first differences
    du = np.diff(u) / dr
    dv = np.diff(v) / dr
    R = np.sum(du**2 + dv**2) / N

    # Volume constraint: (u+v)^2 on interior points
    interior = slice(1, N-1)
    penalty_vol = lambda_vol * np.sum((u[interior] + v[interior])**2) / N

    # Quadratic penalty on v (prevent g_rr from blowing up)
    penalty_v = lambda_v * np.sum(v[interior]**2) / N

    F = R + penalty_vol + penalty_v

    # ---------- Analytical gradients ----------
    grad_u = np.zeros(N)
    grad_v = np.zeros(N)

    # Curvature gradient (discrete Laplacian)
    for i in range(1, N-1):
        grad_u[i] = 2.0 * ((u[i] - u[i-1])/(dr**2) + (u[i] - u[i+1])/(dr**2)) / N
        grad_v[i] = 2.0 * ((v[i] - v[i-1])/(dr**2) + (v[i] - v[i+1])/(dr**2)) / N

    # Volume constraint gradient
    for i in range(1, N-1):
        grad_u[i] += 2.0 * lambda_vol * (u[i] + v[i]) / N
        grad_v[i] += 2.0 * lambda_vol * (u[i] + v[i]) / N

    # Quadratic penalty gradient
    for i in range(1, N-1):
        grad_v[i] += 2.0 * lambda_v * v[i] / N

    # Return gradient for free variables only
    grad = np.concatenate([grad_u[free_indices], grad_v[free_indices]])
    return F, grad

def objective(x):
    F, _ = compute_energy_and_grad(x)
    return F

def gradient(x):
    _, grad = compute_energy_and_grad(x)
    return grad

# ====================== Optimization ======================
print("Minimizing information free energy using L-BFGS-B (enhanced constraints)...")
result = minimize(objective, x0, method='L-BFGS-B', jac=gradient,
                  bounds=bounds, options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-9, 'maxcor': 20})

print(f"Optimization success: {result.success}")
print(f"Final free energy: {result.fun:.6e}")
print(f"Number of iterations: {result.nit}")

# Extract optimized fields
u_opt = u_sch.copy()
v_opt = v_sch.copy()
u_opt[free_indices] = result.x[:n_free]
v_opt[free_indices] = result.x[n_free:]

g_tt_evolved = -np.exp(u_opt)
g_rr_evolved = np.exp(v_opt)

# ====================== Error Analysis ======================
mse_tt = np.mean((g_tt_evolved - g_tt_sch)**2)
mse_rr = np.mean((g_rr_evolved - g_rr_sch)**2)
print(f"Mean squared error w.r.t. Schwarzschild: g_tt = {mse_tt:.2e}, g_rr = {mse_rr:.2e}")

# Volume element check
vol_evolved = -g_tt_evolved * g_rr_evolved
vol_target = 1.0   # For Schwarzschild, -g_tt * g_rr = 1
vol_mse = np.mean((vol_evolved - vol_target)**2)
print(f"Volume element (-g_tt*g_rr) MSE from 1: {vol_mse:.2e}")

# ====================== Plotting ======================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(r, -g_tt_evolved, 'r--', linewidth=2, label='Evolved')
axes[0].plot(r, -g_tt_sch, 'k-', linewidth=2, label='Schwarzschild')
axes[0].set_xlabel('r')
axes[0].set_ylabel('-g_tt')
axes[0].set_title('g_tt Component')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(r, g_rr_evolved, 'r--', linewidth=2, label='Evolved')
axes[1].plot(r, g_rr_sch, 'k-', linewidth=2, label='Schwarzschild')
axes[1].set_xlabel('r')
axes[1].set_ylabel('g_rr')
axes[1].set_title('g_rr Component')
axes[1].legend()
axes[1].grid(alpha=0.3)

axes[2].plot(r, vol_evolved, 'r--', linewidth=2, label='Evolved')
axes[2].axhline(y=1.0, color='k', linestyle='-', label='Target = 1')
axes[2].set_xlabel('r')
axes[2].set_ylabel('-g_tt * g_rr')
axes[2].set_title('Volume Element')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('black_hole_steady_state.png', dpi=150)
plt.show()

if mse_tt < 1e-4 and mse_rr < 1e-2:
    print("✅ Optimization successfully converged to the Schwarzschild black hole steady state.")
else:
    print("⚠️ Slight residual error – try increasing N or adjusting lambda_vol, lambda_v.")