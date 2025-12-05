# ==== main.py ====
import numpy as np
import matplotlib.pyplot as plt

# System parameters
m = 1.0  # mass (kg)
k = 1.0  # stiffness (N/m)
omega_n = np.sqrt(k / m)  # natural frequency (rad/s)
X0 = 1.0  # initial displacement (m)
V0 = 0.0  # initial velocity (m/s)

# Damping ratios for the three regimes
zeta_vals = {
    'underdamped': 0.5,   # ζ < 1
    'critical': 1.0,      # ζ = 1
    'overdamped': 1.5     # ζ > 1
}

# Time vector for simulation
t_max = 20.0
num_points = 2000
t = np.linspace(0, t_max, num_points)

# Containers for results
results = {}

def underdamped_solution(zeta, t):
    omega_d = omega_n * np.sqrt(1 - zeta ** 2)
    exp_term = np.exp(-zeta * omega_n * t)
    cos_term = np.cos(omega_d * t)
    sin_term = np.sin(omega_d * t)
    A = X0
    B = (zeta / np.sqrt(1 - zeta ** 2)) * X0
    x = exp_term * (A * cos_term + B * sin_term)
    # Velocity (first derivative)
    x_dot = exp_term * (
        -zeta * omega_n * (A * cos_term + B * sin_term)
        - omega_d * (A * sin_term) + omega_d * B * cos_term
    )
    return x, x_dot

def critical_solution(t):
    exp_term = np.exp(-omega_n * t)
    x = X0 * (1 + omega_n * t) * exp_term
    # Velocity
    x_dot = X0 * omega_n * exp_term * ( -omega_n * t )
    return x, x_dot

def overdamped_solution(zeta, t):
    s = np.sqrt(zeta ** 2 - 1)
    r1 = -omega_n * (zeta - s)
    r2 = -omega_n * (zeta + s)
    # Coefficients from initial conditions
    A = (r2 * X0 - V0) / (r2 - r1)
    B = (V0 - r1 * X0) / (r2 - r1)
    x = A * np.exp(r1 * t) + B * np.exp(r2 * t)
    # Velocity
    x_dot = A * r1 * np.exp(r1 * t) + B * r2 * np.exp(r2 * t)
    return x, x_dot

# Compute solutions for each damping regime
for regime, zeta in zeta_vals.items():
    if zeta < 1.0:
        x, x_dot = underdamped_solution(zeta, t)
    elif np.isclose(zeta, 1.0):
        x, x_dot = critical_solution(t)
    else:
        x, x_dot = overdamped_solution(zeta, t)
    results[regime] = {
        't': t,
        'x': x,
        'x_dot': x_dot,
        'zeta': zeta
    }

# Experiment 1: Displacement vs Time
plt.figure(figsize=(8, 5))
for regime, data in results.items():
    plt.plot(data['t'], data['x'], label=f"{regime} (ζ={data['zeta']:.2f})")
plt.title('Displacement vs Time for Different Damping Ratios')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('displacement_vs_time.png')
plt.close()

# Experiment 2: Phase Space (Velocity vs Displacement)
plt.figure(figsize=(8, 5))
for regime, data in results.items():
    plt.plot(data['x'], data['x_dot'], label=f"{regime} (ζ={data['zeta']:.2f})")
    # Add arrows to indicate direction
    skip = max(1, len(data['t']) // 30)
    plt.quiver(
        data['x'][::skip], data['x_dot'][::skip],
        np.gradient(data['x'][::skip]), np.gradient(data['x_dot'][::skip]),
        angles='xy', scale_units='xy', scale=0.5, width=0.003, alpha=0.6
    )
plt.title('Phase‑Space Trajectories')
plt.xlabel('Displacement (m)')
plt.ylabel('Velocity (m/s)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('phase_space.png')
plt.close()

# Primary numeric answer: natural frequency of the undamped system
answer = omega_n
print('Answer:', answer)

