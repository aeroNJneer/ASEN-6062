import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy

def jacobi_coord(masses, pos):
    M = np.sum(masses)
    R = np.zeros_like(pos)
    R[0] = pos[1] - pos[0]
    R[1] = pos[2] - (masses[0] * pos[0] + masses[1] * pos[1]) / (masses[0] + masses[1])
    R[2] = pos[3] - (masses[0] * pos[0] + masses[1] * pos[1] + masses[2] * pos[2]) / M
    return R

def reduced_mass(masses):
    M = np.sum(masses)
    mu1 = masses[0] * masses[1] / (masses[0] + masses[1])
    mu2 = (masses[0] + masses[1]) * masses[2] / M
    return mu1, mu2

def invert_jacobi_coord(masses, R):
    M = np.sum(masses)
    pos = np.zeros_like(R)
    pos[0] = -masses[1] / (masses[0] + masses[1]) * R[0] - masses[2] / M * R[1] - mu2 / M * R[2]
    pos[1] = masses[0] / (masses[0] + masses[1]) * R[0] - masses[2] / M * R[1] - masses[2] / M * R[2]
    pos[2] = (masses[0] + masses[1]) / M * R[1] - masses[2] / M * R[2]
    pos[3] = R[2]
    return pos

def calc_com(masses, pos):
    M = np.sum(masses)
    com = np.sum(pos * masses[:, None], axis=0) / M
    return com

def transfer_coordinates(masses, pos):
    com = calc_com(masses, pos)
    pos_centered = pos - com
    R = jacobi_coord(masses, pos_centered)
    return R

def dist_to_particle(pos, particle_index):
    return np.linalg.norm(pos - pos[particle_index], axis=1)

def effective_potential(masses, pos, particle_index):
    G = 1.0  # Gravitational constant in normalized units
    potential = 0.0
    for i in range(len(masses)):
        if i != particle_index:
            r = np.linalg.norm(pos[particle_index] - pos[i])
            potential -= G * masses[i] / r
    return potential

def eqn_of_motion(masses, pos, vel):
    acc = np.zeros_like(pos)
    for i in range(len(masses)):
        for j in range(len(masses)):
            if i != j:
                r_vec = pos[j] - pos[i]
                r = np.linalg.norm(r_vec)
                acc[i] += masses[j] * r_vec / r**3 
    dydt = np.concatenate((vel.flatten(), acc.flatten()))
    return dydt

def integrate_orbits(masses, pos, vel, t_span, dt):
    y0 = np.concatenate((pos.flatten(), vel.flatten()))
    t_eval = np.arange(t_span[0], t_span[1], dt)
    n = len(masses)
    sol = solve_ivp(lambda t, y: eqn_of_motion(masses, y[:n*3].reshape(-1, 3), y[n*3:].reshape(-1, 3)), t_span, y0, t_eval=t_eval, method='DOP853', rtol=1e-10, atol=1e-12)
    n_t = len(sol.t)
    return sol.t, sol.y[:n*3].reshape(n, 3, n_t), sol.y[n*3:].reshape(n, 3, n_t)


def compute_energy(positions, velocities, masses):
    G = 1.0  # Gravitational constant in AU^3 / (yr^2 * solar_mass)
    masses = np.array(masses, dtype=float)

    # Kinetic energy
    kinetic = 0.5 * np.sum(masses[:, None] * velocities**2)

    # Potential energy
    potential = 0.0
    N = len(masses)
    for i in range(N):
        for j in range(i+1, N):
            r_vec = positions[j] - positions[i]
            r_mag = np.linalg.norm(r_vec)
            potential -= G * masses[i] * masses[j] / r_mag
    return kinetic + potential

# compute angular momentum
def compute_angular_momentum(positions, velocities, masses):
    masses = np.array(masses, dtype=float)
    L = np.zeros(3)
    for i in range(len(masses)):
        L += masses[i] * np.cross(positions[i], velocities[i])
    return L  

def calculate_residuals(positions, velocities, masses):
    M = len(velocities)
    energies = np.zeros(M)
    angular_momenta = np.zeros((M, 3))
    # Store initial conserved values as well
    energies[0] = compute_energy(positions[0], velocities[0], masses)
    angular_momenta[0] = compute_angular_momentum(positions[0], velocities[0], masses)

    for k in range(1, M):
        energies[k] = compute_energy(positions[k], velocities[k], masses)
        angular_momenta[k] = compute_angular_momentum(positions[k], velocities[k], masses)

    return energies-energies[0], angular_momenta-angular_momenta[0]


masses = np.array([1.0, 0.03, 0.003, 10e-12], dtype=float)
pos = np.array([
        [0.0, 0.0, 0.0],
        [1, 0.0, 0.0],
        [0.5, np.sqrt(3) / 2, 0.0]
    ], dtype=float)

p4 = calc_com(masses[0:3], pos)  # particle at com which is R[2] = 0 in jacobi coordinates
# add particle to pos
pos = np.vstack((pos, p4))

Jacobi_pos = jacobi_coord(masses, pos)
mu1, mu2 = reduced_mass(masses)

transformed_pos = pos - calc_com(masses, pos)
omega = np.sqrt(np.sum(masses) / 1.0**3)  # omega^2 = G*M/a^3, G=1, a=1
velocity = omega * np.column_stack((-transformed_pos[:, 1], transformed_pos[:, 0], np.zeros(len(masses))))  # v = omega x r

print(transformed_pos)

dist_to_p4 = dist_to_particle(pos, 3)
print("Distances to particle 4:", dist_to_p4)

t, y, v = integrate_orbits(masses, pos, velocity, (0, 10), 0.01)
plt.figure(figsize=(8, 6))
for i in range(len(masses)):
    plt.plot(y[i, 0], y[i, 1], label=f'Particle {i+1}')
plt.xlabel('x')
plt.ylabel('y') 
plt.title('Orbits of Particles')
plt.legend()
plt.grid()
plt.axis('equal')
plt.show()

plt.figure(figsize=(8, 6))
energies, angular_momenta = calculate_residuals(y.transpose(2, 0, 1), v.transpose(2, 0, 1), masses)
plt.subplot(2, 1, 1)
plt.plot(t, energies, label='Energy Residual')
plt.xlabel('Time')  
plt.ylabel('Energy Residual')
plt.title('Energy Residual Over Time')
plt.legend()
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(t, angular_momenta, label='Angular Momentum Residual')
plt.xlabel('Time')
plt.ylabel('Angular Momentum Residual')
plt.title('Angular Momentum Residual Over Time')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

