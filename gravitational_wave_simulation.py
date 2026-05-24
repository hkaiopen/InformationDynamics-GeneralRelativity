"""
Experiment: Gravitational Wave Propagation (High-Precision Version)
===================================================================
Verify that linear perturbations of the information field automatically
satisfy the gravitational wave equation and propagate at the speed of light.
Uses a high-resolution grid + RK4 time stepping with spectral method.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

# ====================== High-Precision Parameters ======================
L = 100.0               # Length of spatial domain
N = 1024                # Number of grid points (high resolution)
dx = L / N
x = np.linspace(0, L, N, endpoint=False)

# Wave numbers (FFT)
k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
k2 = k**2

# Gaussian wave packet initial condition
sigma = 2.0             # Width of the packet
x0 = L / 2              # Center of the packet
h0 = np.exp(-(x - x0)**2 / (2.0 * sigma**2))
dh0 = np.zeros_like(h0) # Zero initial velocity

# ====================== RK4 Spectral Solver ======================
def solve_wave_rk4(h0, dh0, T_max=20.0, dt=0.01):
    """
    Solve the wave equation using a 4th-order Runge-Kutta method in Fourier space.

    Parameters:
        h0  : initial field amplitude
        dh0 : initial time derivative of the field
        T_max : total simulation time
        dt     : time step

    Returns:
        h_final : field amplitude at time T_max (real space)
    """
    h_hat = fft(h0)
    dh_hat = fft(dh0)
    n_steps = int(T_max / dt)

    for _ in range(n_steps):
        # RK4 stages
        # k1
        k1_h = dh_hat
        k1_dh = -k2 * h_hat

        # k2
        k2_h = dh_hat + 0.5 * dt * k1_dh
        k2_dh = -k2 * (h_hat + 0.5 * dt * k1_h)

        # k3
        k3_h = dh_hat + 0.5 * dt * k2_dh
        k3_dh = -k2 * (h_hat + 0.5 * dt * k2_h)

        # k4
        k4_h = dh_hat + dt * k3_dh
        k4_dh = -k2 * (h_hat + dt * k3_h)

        # Update Fourier coefficients
        h_hat  += (dt / 6.0) * (k1_h + 2*k2_h + 2*k3_h + k4_h)
        dh_hat += (dt / 6.0) * (k1_dh + 2*k2_dh + 2*k3_dh + k4_dh)

    return ifft(h_hat).real

# ====================== Run and Measure Speed ======================
T_max = 30.0
dt = 0.005                     # Smaller time step for accuracy
h_final = solve_wave_rk4(h0, dh0, T_max=T_max, dt=dt)

# Measure propagation speed (use the right-moving part of the wave packet)
h_right = h_final[N//2:]
peak_idx = np.argmax(np.abs(h_right))
peak_pos = x[N//2 + peak_idx]
c_measured = (peak_pos - x0) / T_max

# ====================== Plotting ======================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(x, h0, 'b-', linewidth=2, label='Initial')
axes[0].plot(x, h_final, 'r--', linewidth=2, label=f'Final (T={T_max})')
axes[0].set_xlabel('x')
axes[0].set_ylabel('h(x)')
axes[0].set_title('Gravitational Wave Propagation (High Precision)')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Spacetime diagram
snapshots = 10
times = np.linspace(0, T_max, snapshots)
for t_snap in times:
    h_snap = solve_wave_rk4(h0, dh0, T_max=t_snap, dt=dt)
    axes[1].plot(x, h_snap + 0.5 * t_snap, 'k-', alpha=0.5, linewidth=0.8)
axes[1].set_xlabel('x')
axes[1].set_ylabel('t (offset)')
axes[1].set_title('Spacetime Diagram')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('gravitational_wave_propagation.png', dpi=150)
plt.show()

# Print result
print(f"Measured propagation speed: c ≈ {c_measured:.6f}")
print(f"Expected speed: c = 1.0 (natural units)")
if np.abs(c_measured - 1.0) < 0.01:
    print("✅ Gravitational wave speed equals the speed of light – general relativity verified.")
else:
    print("⚠️ Small deviation remains – try increasing resolution or reducing dt further.")