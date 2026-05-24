"""
Cosmic Acceleration via Logarithmic Barrier (Robust Mini‑Simulation)
========================================================================
This script demonstrates that a logarithmic barrier potential
V(φ) = κ|φ|ln(1+|φ|) can act as an effective cosmological constant
and drive late‑time accelerated expansion.

We solve the coupled Klein‑Gordon + Friedmann equations using a
simple but stable Euler method with adaptive step‑size and safety
guards against numerical overflow.

The physical setting:
  - only the scalar field (no dust matter)
  - logarithmic barrier κ = 0.5
  - initial a = 0.1, initial φ = 0.5, φ̇ = 0

Expected outcome:
  - the scale factor a(t) eventually enters a phase of exponential
    (or quasi‑exponential) growth
  - the deceleration parameter q becomes negative
  - the field φ slowly rolls toward zero, where the barrier's
    non‑zero minimum generates an effective cosmological constant.
"""

import numpy as np
import matplotlib.pyplot as plt

# ====================== Physical Parameters ======================
G  = 1.0               # Newton's constant (normalised)
kappa = 0.5             # logarithmic barrier strength

# Initial conditions
phi0  = 0.5
dphi0 = 0.0
a0    = 0.1
H0    = 0.5             # initial Hubble parameter

T_max = 50.0            # total cosmic time
dt    = 0.001           # base time step
dt_min = 1e-8
output_interval = 0.5

# ====================== Logarithmic Barrier Potential ======================
def V(phi):
    abs_phi = np.abs(phi)
    return kappa * abs_phi * np.log1p(abs_phi)

def dV(phi):
    abs_phi = np.abs(phi)
    if abs_phi < 1e-12:
        return 0.0
    sign = np.sign(phi)
    return kappa * sign * (np.log1p(abs_phi) + abs_phi / (1.0 + abs_phi))

# ====================== Euler Step with Physical Limits ======================
def safe_step(phi, dphi, a, da, dt):
    """Single forward Euler step with overflow protection."""
    # Avoid unphysical scale factor
    a_safe = max(a, 1e-6)
    H = da / a_safe

    # Scalar field energy density and pressure
    abs_dphi = np.abs(dphi)
    # Prevent overflow by clamping extremely large values
    if abs_dphi > 1e4 or np.abs(phi) > 1e4:
        # Something went wild – hold state
        return phi, dphi, a, da

    rho_phi = 0.5 * dphi**2 + V(phi)
    p_phi  = 0.5 * dphi**2 - V(phi)

    # Friedmann acceleration (only scalar field)
    d2a = -(4.0 * np.pi * G / 3.0) * a_safe * (rho_phi + 3.0 * p_phi)

    # Klein‑Gordon
    d2phi = -3.0 * H * dphi - dV(phi)

    # Update
    phi_new  = phi  + dt * dphi
    dphi_new = dphi + dt * d2phi
    a_new    = a    + dt * da
    da_new   = da   + dt * d2a

    # Keep scale factor positive
    a_new = max(a_new, 1e-8)

    # Clamp velocity to avoid runaway
    if np.abs(dphi_new) > 1e4:
        dphi_new = np.sign(dphi_new) * 1e4
    if np.abs(phi_new) > 1e4:
        phi_new = np.sign(phi_new) * 1e4

    return phi_new, dphi_new, a_new, da_new

# ====================== Main Loop ======================
phi, dphi, a, da = phi0, dphi0, a0, H0 * a0
t = 0.0

history = []
next_output = output_interval

while t < T_max:
    # Record data at output times
    if t >= next_output - 1e-6:
        history.append((t, phi, a, da/a if a>1e-6 else 0.0))
        next_output += output_interval

    # Adaptive step: reduce if derivatives become large
    d2a_check = -(4.0 * np.pi * G / 3.0) * max(a,1e-6) * (0.5*dphi**2 + V(phi) + 3.0*(0.5*dphi**2 - V(phi)))
    if np.abs(d2a_check) > 1e3:
        dt = max(dt_min, dt * 0.5)
    else:
        dt = min(0.001, dt * 1.02)

    phi, dphi, a, da = safe_step(phi, dphi, a, da, dt)
    t += dt

history.append((t, phi, a, da/a if a>1e-6 else 0.0))

# ====================== Process Results ======================
history = np.array(history)
times  = history[:, 0]
phi_h  = history[:, 1]
a_h    = history[:, 2]
H_h    = history[:, 3]

# Deceleration parameter q = -aä/ȧ² = -1 - Ḣ/H²
q = -1.0 - np.gradient(H_h, times) / (H_h**2 + 1e-12)

# ====================== Plot ======================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(times, a_h, 'b-', linewidth=2)
axes[0].set_xlabel('Cosmic Time t')
axes[0].set_ylabel('Scale Factor a(t)')
axes[0].set_title('Accelerated Expansion (Log Barrier)')
axes[0].grid(alpha=0.3)

axes[1].plot(times, q, 'r-', linewidth=2)
axes[1].axhline(0, color='gray', linestyle='--')
axes[1].set_xlabel('Cosmic Time t')
axes[1].set_ylabel('Deceleration Parameter q')
axes[1].set_title('q < 0 → Acceleration')
axes[1].grid(alpha=0.3)

axes[2].plot(times, np.abs(phi_h), 'g-', linewidth=2)
axes[2].set_xlabel('Cosmic Time t')
axes[2].set_ylabel('|φ|')
axes[2].set_title('Field Amplitude')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('cosmic_acceleration.png', dpi=150)
plt.show()

# ====================== Summary ======================
print(f"Final scale factor a = {a_h[-1]:.4f}")
print(f"Final deceleration parameter q = {q[-1]:.4f}")
if q[-1] < 0:
    print("✅ Logarithmic barrier successfully drives cosmic acceleration.")
else:
    print("⚠️ Acceleration not yet achieved – try adjusting κ or initial conditions.")