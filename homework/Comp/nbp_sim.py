# generate jacobi coordinates for the NBP simulation
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy

def jacobi_coordinates(positions, masses):
    """
    Compute Jacobi coordinates for a system of bodies.

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        Cartesian positions of N bodies in an inertial frame.
    masses : ndarray, shape (N,)
        Masses of the N bodies.

    Returns
    -------
    jacobi_pos : ndarray, shape (N, 3)
        Jacobi coordinates for each body.
    """
    positions = np.array(positions, dtype=float)
    masses = np.array(masses, dtype=float)
    M = np.cumsum(masses)

    N = len(masses)
    jacobi_pos = np.zeros_like(positions+1)

    # First Jacobi coordinate: position of body 1 relative to body 0
    jacobi_pos[0] = positions[0]

    # Iteratively compute Jacobi coordinates
    for i in range(1, N):
        # Center of mass of all previous bodies
        total_mass_prev = np.sum(masses[:i])
        com_prev = np.sum(positions[:i] * masses[:i, None], axis=0) / total_mass_prev
        jacobi_pos[i] = positions[i] - com_prev
    return jacobi_pos


def reduced_masses(masses):
    """
    Compute the Jacobi reduced masses eta_i = M_i * m_{i+1} / M_{i+1}.

    Following Scheeres' notation (Jacobi_Notation.pdf):
      M_0 = 0, M_i = M_{i-1} + m_i  for i = 1, ..., N
      eta_i = M_i * m_{i+1} / M_{i+1}  for i = 1, ..., N-1

    Here the input masses array is 0-indexed: masses[0] = m_1, ..., masses[N-1] = m_N.

    Parameters
    ----------
    masses : ndarray, shape (N,)
        Masses of the N bodies (0-indexed).

    Returns
    -------
    eta : ndarray, shape (N-1,)
        Reduced masses for Jacobi coordinates i = 1, ..., N-1.
    """
    masses = np.array(masses, dtype=float)
    N = len(masses)
    # M[i] = sum of masses[0..i] 
    M = np.cumsum(masses)
    eta = np.zeros(N - 1)
    for i in range(N - 1):
        # eta_{i+1} in Scheeres' 1-indexed notation = M[i] * masses[i+1] / M[i+1]
        # Here i is 0-indexed Jacobi index (R_1 corresponds to i=0 in eta)
        eta[i] = M[i] * masses[i + 1] / M[i + 1]
    return eta


def jacobi_linear_transform(masses):
    """
    Return linear transform L (shape (N, N)) such that
    jacobi_pos = L @ cartesian_positions (applied per-coordinate).

    The Jacobi coordinates used in this module are defined recursively as
    J_0 = x_0
    J_i = x_i - (sum_{j=0..i-1} m_j * x_j) / (sum_{j=0..i-1} m_j),  for i >= 1

    Therefore each row i of L satisfies:
      L[i, i] = 1
      L[i, j] = -m_j / (sum_{k=0..i-1} m_k)   for j < i
      L[i, j] = 0                                 for j > i

    This linear operator maps Cartesian positions to Jacobi coordinates
    independently for each Cartesian axis (x, y, z). We use the same
    linear mapping to transform Cartesian accelerations into Jacobi
    accelerations (jac_acc = L @ cart_acc) because the mapping is
    mass-weighted and purely linear.
    """
    masses = np.array(masses, dtype=float)
    N = len(masses)
    L = np.zeros((N, N), dtype=float)
    for i in range(N):
        if i == 0:
            L[0, 0] = 1.0
        else:
            L[i, i] = 1.0
            total_prev = np.sum(masses[:i])
            for j in range(i):
                L[i, j] = -masses[j] / total_prev
    return L

def _potential_gradient_jacobi(jac_pos_rel, masses):
    """
    Compute -dU/dR_l for l = 1, ..., N-1 directly in Jacobi coordinates.

    Uses the partial derivative formula from Scheeres (Jacobi_Notation.pdf).
    For body pair (i, j) with i < j (1-indexed), the partial of the
    separation vector r_{ij} w.r.t. Jacobi coordinate R_l is:

        dr_{ij}/dR_l =  0                       l > j-1  or  l < i-1
                        +I                       l = j-1
                        (m_{l+1}/M_{l+1}) I      i-1 < l < j-1
                        -(M_{i-1}/M_i) I         l = i-1

    Mapping to 0-indexed bodies a < b  (a = i-1, b = j-1):
        l = b    =>  +1
        a < l < b => masses[l] / cum[l]
        l = a    =>  -(cum[a-1] / cum[a])   (zero when a = 0)

    The generalized force is then:
        -dU/dR_l = -sum_{a<b} F_{ab} * c_l

    where F_{ab} = G * m_a * m_b * r_{ab} / |r_{ab}|^3  and c_l is the
    coefficient from the partial derivative table above.

    Note the sign: U = -G sum m_i m_j / |r_{ij}|, so
      dU/dR_l = +G sum m_i m_j r_{ij}/|r_{ij}|^3 * c_l = +sum F_{ab} * c_l
      => -dU/dR_l = -sum F_{ab} * c_l

    Parameters
    ----------
    jac_pos_rel : ndarray, shape (N-1, 3)
        Jacobi relative positions R_1, ..., R_{N-1} (0-indexed as [0..N-2]).
    masses : ndarray, shape (N,)
        Masses of the N bodies (0-indexed).

    Returns
    -------
    neg_dUdR : ndarray, shape (N-1, 3)
        The generalized Jacobi force -dU/dR_l for l = 1, ..., N-1.
    """
    masses = np.array(masses, dtype=float)
    N = len(masses)
    d = jac_pos_rel.shape[1]
    G = 1.0

    cum = np.cumsum(masses)  # cum[k] = m_0 + ... + m_k

    # Convert Jacobi -> inertial to get separation vectors
    full_jac = np.zeros((N, d))
    full_jac[1:] = jac_pos_rel
    cart_pos = positions_from_jacobi(full_jac, masses)

    neg_dUdR = np.zeros((N - 1, d))

    for a in range(N):
        for b in range(a + 1, N):
            r_vec = cart_pos[b] - cart_pos[a]
            r_mag = np.linalg.norm(r_vec)
            if r_mag == 0:
                continue
            F_ab = G * masses[a] * masses[b] * r_vec / r_mag**3

            # -dU/dR_l = -sum F_ab * c_l  (note the overall minus sign)

            # l = b: coefficient +1
            neg_dUdR[b - 1] -= F_ab

            # l = a: coefficient -(cum[a-1]/cum[a]), only for a >= 1
            if a >= 1:
                neg_dUdR[a - 1] += (cum[a - 1] / cum[a]) * F_ab

            # a < l < b: coefficient masses[l]/cum[l]
            for l in range(max(a + 1, 1), b):
                neg_dUdR[l - 1] -= (masses[l] / cum[l]) * F_ab

    return neg_dUdR

# generate the EOM using Lagrangian equations in Jacobi coordinates
def equations_of_motion(t, y, masses):
    """
    Compute the derivatives for the N-body problem using the Lagrangian
    formulation in Jacobi coordinates

    The state vector contains only the N-1 relative Jacobi coordinates
    R_1, ..., R_{N-1} and their velocities (R_0 is the ignorable CoM
    coordinate, fixed at the origin).

    The Lagrangian is L = T - U where:
      T = (1/2) sum_{i=1}^{N-1} eta_i * V_i . V_i
      U = -G sum_{i<j} m_i m_j / |r_{ij}|

    The EOM are:  eta_i * R_i'' = -dU/dR_i
    or equivalently: R_i'' = (1/eta_i) * (-dU/dR_i)

    Parameters
    ----------
    t : float
        Time variable (required by ODE solvers).
    y : ndarray, shape (2*(N-1)*3,)
        State vector: [R_1, ..., R_{N-1}, V_1, ..., V_{N-1}] flattened.
    masses : ndarray, shape (N,)
        Masses of the N bodies.

    Returns
    -------
    dydt : ndarray, shape (2*(N-1)*3,)
        Derivative of the state vector.
    """
    masses = np.array(masses, dtype=float)
    N = len(masses)
    n_rel = N - 1
    dim = 3

    state_size = n_rel * dim
    jac_pos_rel = y[:state_size].reshape((n_rel, dim))   # R_1, ..., R_{N-1}
    jac_vel_rel = y[state_size:].reshape((n_rel, dim))   # V_1, ..., V_{N-1}

    # Compute the Jacobi reduced masses eta_i
    eta = reduced_masses(masses)

    # Compute the generalized Jacobi force: -dU/dR_l
    neg_dUdR = _potential_gradient_jacobi(jac_pos_rel, masses)

    # Jacobi accelerations: R_i'' = (1/eta_i) * (-dU/dR_i)
    jac_acc_rel = neg_dUdR / eta[:, None]

    dydt = np.zeros_like(y)
    dydt[:state_size] = jac_vel_rel.flatten()
    dydt[state_size:] = jac_acc_rel.flatten()

    return dydt

def positions_from_jacobi(jacobi_pos, masses):
    """
    Convert Jacobi coordinates back to Cartesian positions.

    Parameters
    ----------
    jacobi_pos : ndarray, shape (N, 3)
        Jacobi coordinates for each body.
    masses : ndarray, shape (N,)
        Masses of the N bodies.

    Returns
    -------
    positions : ndarray, shape (N, 3)
        Cartesian positions of N bodies in an inertial frame.
    """
    N = len(masses)
    positions = np.zeros_like(jacobi_pos)

    # First body is at the origin
    positions[0] = jacobi_pos[0]

    # Iteratively compute Cartesian positions
    for i in range(1, N):
        total_mass_prev = np.sum(masses[:i])
        com_prev = np.sum(positions[:i] * masses[:i, None], axis=0) / total_mass_prev
        positions[i] = jacobi_pos[i] + com_prev

    return positions

def velocities_from_jacobi(jacobi_vel, masses):
    """
    Convert Jacobi velocities back to Cartesian velocities.

    Parameters
    ----------
    jacobi_vel : ndarray, shape (N, 3)
        Jacobi velocities for each body.
    masses : ndarray, shape (N,)
        Masses of the N bodies.

    Returns
    -------
    velocities : ndarray, shape (N, 3)
        Cartesian velocities of N bodies in an inertial frame.
    """
    N = len(masses)
    velocities = np.zeros_like(jacobi_vel)

    # First body is at rest in the inertial frame
    velocities[0] = jacobi_vel[0]

    # Iteratively compute Cartesian velocities
    for i in range(1, N):
        total_mass_prev = np.sum(masses[:i])
        com_prev_vel = np.sum(velocities[:i] * masses[:i, None], axis=0) / total_mass_prev
        velocities[i] = jacobi_vel[i] + com_prev_vel

    return velocities

def simulate_nbp(initial_positions, initial_velocities, masses, t_span, dt):
    """
    Simulate the N-body problem using the Lagrangian formulation in Jacobi
    coordinates. 
    The EOM are:

        eta_i * R_i'' = -dU/dR_i

    where eta_i = M_i * m_{i+1} / M_{i+1} are the Jacobi reduced masses
    """
    # Ensure masses is a numpy array
    masses = np.array(masses, dtype=float)
    N = len(masses)

    # Convert initial conditions to Jacobi coordinates
    jacobi_pos = jacobi_coordinates(initial_positions, masses)
    jacobi_vel = jacobi_coordinates(initial_velocities, masses)

    # Extract only the N-1 relative Jacobi coordinates (drop R_0, the CoM)
    jac_pos_rel = jacobi_pos[1:]   # shape (N-1, 3)
    jac_vel_rel = jacobi_vel[1:]   # shape (N-1, 3)

    # Flatten initial conditions for ODE solver
    y0 = np.hstack((jac_pos_rel.flatten(), jac_vel_rel.flatten()))

    # Time points for integration (inclusive endpoints, avoid floating-point overshoot)
    t0, t1 = t_span
    num_steps = max(2, int(np.floor((t1 - t0) / dt)) + 1)
    t_eval = np.linspace(t0, t1, num_steps)

    # Integrate equations of motion with tighter tolerances and high-order method
    sol = solve_ivp(
        equations_of_motion,
        t_span,
        y0,
        args=(masses,),
        t_eval=t_eval,
        method='DOP853',
        rtol=1e-12,
        atol=1e-14,
    )

    times = sol.t
    M_steps = len(times)
    n_rel = N - 1
    state_size = n_rel * 3

    # Convert Jacobi solution back to Cartesian coordinates for each time step
    positions = np.zeros((M_steps, N, 3))
    velocities = np.zeros((M_steps, N, 3))
    for k in range(M_steps):
        jac_pos_rel_k = sol.y[:state_size, k].reshape((n_rel, 3))
        jac_vel_rel_k = sol.y[state_size:, k].reshape((n_rel, 3))

        # Reconstruct full Jacobi arrays (R_0 = 0 in CoM frame)
        jac_pos_full = np.zeros((N, 3))
        jac_pos_full[1:] = jac_pos_rel_k
        jac_vel_full = np.zeros((N, 3))
        jac_vel_full[1:] = jac_vel_rel_k

        positions[k] = positions_from_jacobi(jac_pos_full, masses)
        velocities[k] = velocities_from_jacobi(jac_vel_full, masses)

        # The reconstruction above places body 0 at the origin (since
        # jac_pos_full[0] = 0).  Shift to the centre-of-mass frame so
        # that conserved-quantity checks use an inertial frame.
        M_total = masses.sum()
        com_pos = np.sum(masses[:, None] * positions[k], axis=0) / M_total
        com_vel = np.sum(masses[:, None] * velocities[k], axis=0) / M_total
        positions[k] -= com_pos
        velocities[k] -= com_vel

    return times, positions, velocities


# Conserved quantities in the N-body problem include:
# 1. Total energy (kinetic + potential)
# 2. Total linear momentum
# 3. Total angular momentum

def compute_energy(positions, velocities, masses):
    """
    Compute the total energy of the system.

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        Cartesian positions of N bodies in an inertial frame.
    velocities : ndarray, shape (N, 3)
        Cartesian velocities of N bodies in an inertial frame.
    masses : ndarray, shape (N,)
        Masses of the N bodies.

    Returns
    -------
    energy : float
        Total energy of the system (kinetic + potential).
    """
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
    """
    Compute the total angular momentum of the system.

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        Cartesian positions of N bodies in an inertial frame.
    velocities : ndarray, shape (N, 3)
        Cartesian velocities of N bodies in an inertial frame.
    masses : ndarray, shape (N,)
        Masses of the N bodies.

    Returns
    -------
    L : ndarray, shape (3,)
        Total angular momentum vector of the system.
    """
    masses = np.array(masses, dtype=float)
    L = np.zeros(3)
    for i in range(len(masses)):
        L += masses[i] * np.cross(positions[i], velocities[i])
    return L    

# simulate dynamics and check conservation of energy and angular momentum
def simulate_and_check_conservation(positions, velocities, masses, t_span, dt):
    times, pos_ts, vel_ts = simulate_nbp(positions, velocities, masses, t_span, dt)

    # Pre-allocate arrays to store conserved quantities at each time step
    M = len(times)
    energies = np.zeros(M)
    angular_momenta = np.zeros((M, 3))

    # Store initial conserved values as well
    energies[0] = compute_energy(positions, velocities, masses)
    angular_momenta[0] = compute_angular_momentum(positions, velocities, masses)

    for k in range(1, M):
        energies[k] = compute_energy(pos_ts[k], vel_ts[k], masses)
        angular_momenta[k] = compute_angular_momentum(pos_ts[k], vel_ts[k], masses)

    return times, pos_ts, vel_ts, energies, angular_momenta

def plot_results(times, pos_ts, vel_ts, masses, energies=None, angular_momenta=None,
                 title=None, test_particle_idx=None):
    """
    Two-figure summary:
      Figure 1: x-y trajectory (standalone)
      Figure 2: x(t), y(t), conservation residuals (3 subplots)

    If test_particle_idx is not None, that body is drawn with a distinct
    dashed black line and labelled 'Test Particle (CoG)'.
    """
    masses = np.asarray(masses, dtype=float)
    N = len(masses)
    # Exclude test particle from mass-based label/color assignment
    N_massive = N if test_particle_idx is None else N - 1
    labels = [f'Body {i+1} (m={masses[i]:.4g})' for i in range(N)]
    colors = plt.cm.tab10(np.linspace(0, 1, max(N_massive, 10)))[:N]
    if test_particle_idx is not None:
        labels[test_particle_idx] = 'Test Particle (CoG)'

    if energies is None:
        energies = np.array([compute_energy(pos_ts[k], vel_ts[k], masses) for k in range(len(times))])
    if angular_momenta is None:
        angular_momenta = np.array([compute_angular_momentum(pos_ts[k], vel_ts[k], masses) for k in range(len(times))])

    # ===== Figure 1: Trajectory =====
    fig1 = plt.figure(figsize=(8, 8))
    if title:
        fig1.suptitle(title, fontsize=14, fontweight='bold')
    ax_traj = fig1.add_subplot(111)
    for i in range(N):
        if i == test_particle_idx:
            ax_traj.plot(pos_ts[:, i, 0], pos_ts[:, i, 1], 'k', lw=1.5, label=labels[i])
            ax_traj.plot(pos_ts[0, i, 0], pos_ts[0, i, 1], 'kD', ms=6)
            ax_traj.plot(pos_ts[-1, i, 0], pos_ts[-1, i, 1], 'k*', ms=8)
        else:
            ax_traj.plot(pos_ts[:, i, 0], pos_ts[:, i, 1], color=colors[i], lw=0.8, label=labels[i])
            ax_traj.plot(pos_ts[0, i, 0], pos_ts[0, i, 1], 'o', color=colors[i],
                         ms=2 + 2 * masses[i] / masses[:N_massive].max())
            ax_traj.plot(pos_ts[-1, i, 0], pos_ts[-1, i, 1], '*', color=colors[i],
                         ms=3 + 2 * masses[i] / masses[:N_massive].max())
    ax_traj.set_aspect('equal')
    ax_traj.legend(fontsize=7, loc='lower left')
    ax_traj.set_xlabel('x')
    ax_traj.set_ylabel('y')
    ax_traj.set_title('Trajectories (x-y plane)')
    ax_traj.grid(True, alpha=0.3)

    # ===== Figure 2: x(t), y(t), residuals =====
    fig2, (ax_x, ax_y, ax_res) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    if title:
        fig2.suptitle(title, fontsize=14, fontweight='bold')
    fig2.subplots_adjust(hspace=0.25, top=0.93)

    # x(t)
    for i in range(N):
        if i == test_particle_idx:
            ax_x.plot(times, pos_ts[:, i, 0], 'k', lw=1.5, label=labels[i])
        else:
            ax_x.plot(times, pos_ts[:, i, 0], color=colors[i], lw=0.8, label=labels[i])
    ax_x.set_ylabel('x')
    ax_x.set_title('x(t)')
    ax_x.legend(fontsize=5, loc='upper right')
    ax_x.grid(True, alpha=0.3)

    # y(t)
    for i in range(N):
        if i == test_particle_idx:
            ax_y.plot(times, pos_ts[:, i, 1], 'k', lw=1.5, label=labels[i])
        else:
            ax_y.plot(times, pos_ts[:, i, 1], color=colors[i], lw=0.8, label=labels[i])
    ax_y.set_ylabel('y')
    ax_y.set_title('y(t)')
    ax_y.legend(fontsize=5, loc='upper right')
    ax_y.grid(True, alpha=0.3)

    # Conservation residuals
    E_vals = energies
    L_vals = np.linalg.norm(angular_momenta, axis=1)
    E0, L0 = E_vals[0], L_vals[0]
    E_res, L_res = E_vals - E0, L_vals - L0

    ax_res.plot(times, E_res, label='E - E0')
    ax_res.plot(times, L_res, label='|L| - |L0|')
    ax_res.set_xlabel('Time')
    ax_res.set_ylabel('Residual')
    ax_res.set_title('Conservation Residuals')
    ax_res.legend(fontsize=7)
    ax_res.grid(True, alpha=0.3)

    # ===== Figure 3: Test particle only =====
    if test_particle_idx is not None:
        tp = test_particle_idx
        fig3, (ax_tp_traj, ax_tp_t) = plt.subplots(1, 2, figsize=(14, 6))
        if title:
            fig3.suptitle(f'{title} — Test Particle', fontsize=14, fontweight='bold')

        # x-y trajectory
        ax_tp_traj.plot(pos_ts[:, tp, 0], pos_ts[:, tp, 1], 'k-', lw=1.2)
        ax_tp_traj.plot(pos_ts[0, tp, 0], pos_ts[0, tp, 1], 'kD', ms=8, label='start')
        ax_tp_traj.plot(pos_ts[-1, tp, 0], pos_ts[-1, tp, 1], 'k*', ms=10, label='end')
        ax_tp_traj.set_aspect('equal')
        ax_tp_traj.set_xlabel('x')
        ax_tp_traj.set_ylabel('y')
        ax_tp_traj.set_title('Test Particle Trajectory (x-y)')
        ax_tp_traj.legend(fontsize=8)
        ax_tp_traj.grid(True, alpha=0.3)

        # x(t) and y(t)
        ax_tp_t.plot(times, pos_ts[:, tp, 0], label='x(t)')
        ax_tp_t.plot(times, pos_ts[:, tp, 1], label='y(t)')
        ax_tp_t.set_xlabel('Time')
        ax_tp_t.set_ylabel('Position')
        ax_tp_t.set_title('Test Particle x(t), y(t)')
        ax_tp_t.legend(fontsize=8)
        ax_tp_t.grid(True, alpha=0.3)

        fig3.tight_layout()

def run_problem(name, pos, vel, masses, T_orbit, n_orbits=1, steps_per_orbit=500, test_particle=False):
    """
    Common driver: simulate, print conservation diagnostics, plot, and show.

    Parameters
    ----------
    T_orbit : float
        One orbital period.
    n_orbits : int or float
        Number of orbital periods to simulate.
    steps_per_orbit : int
        Number of max-step intervals per orbit (controls integration resolution).
    test_particle : bool
        If True, add an infinitesimal-mass body at the system centre of mass
        and track its trajectory.
    """
    masses = np.asarray(masses, dtype=float)
    tp_idx = None

    if test_particle:
        N = len(masses)
        # Test-particle initial conditions: system CoM position & velocity
        total_mass = masses.sum()
        tp_pos0 = np.sum(pos * masses[:, None], axis=0) / total_mass
        tp_vel0 = np.sum(vel * masses[:, None], axis=0) / total_mass

        # Nudge slightly if it coincides with any body (avoid singularity)
        for i in range(N):
            if np.linalg.norm(tp_pos0 - pos[i]) < 1e-10:
                tp_pos0 = tp_pos0 + np.array([1e-6, 1e-6, 0.0])
                break

        # Append test particle as (N+1)th body with infinitesimal mass.
        # Use 1e-10 rather than a smaller epsilon: the Jacobi EOM divides
        # the force by the reduced mass eta ~ epsilon, so too-small values
        # amplify floating-point roundoff and stall the integrator.
        pos = np.vstack([pos, tp_pos0[np.newaxis, :]])
        vel = np.vstack([vel, tp_vel0[np.newaxis, :]])
        masses = np.append(masses, 1e-10)
        tp_idx = len(masses) - 1

    T = T_orbit * n_orbits
    t_span = (0.0, T)
    dt = T_orbit / steps_per_orbit

    times, pos_ts, vel_ts, energies, angular_momenta = simulate_and_check_conservation(
        pos, vel, masses, t_span, dt
    )

    # For display, show test particle mass as 0
    if tp_idx is not None:
        masses[tp_idx] = 0.0

    E0 = energies[0]
    L_mags = np.linalg.norm(angular_momenta, axis=1)
    L0 = L_mags[0]
    print(f"{name}: T={T:.3f} ({n_orbits} orbits)")
    print(f"  Energy:  min={energies.min():.3e}, max={energies.max():.3e}, "
          f"rel change={(energies.max()-energies.min())/np.abs(E0):.3e}")
    print(f"  Ang Mom: min={L_mags.min():.3e}, max={L_mags.max():.3e}, "
          f"rel change={(L_mags.max()-L_mags.min())/np.abs(L0):.3e}")
    plot_results(times, pos_ts, vel_ts, masses, energies, angular_momenta,
                 title=name, test_particle_idx=tp_idx)
    plt.show()

def Euler_collinear_acceleration(m=[1/3, 1/3, 1/3], n_orbits=1, pert=0.0, test_particle=False):

    # TEST - use the Euler collinear central configuration for equal masses to verify the implementation
    # masses sum to 1, G=1 in the equations
    masses = np.array(m, dtype=float)/sum(m) # normalize to sum to 1
    masses[0] += pert
    # Place masses on the x-axis at -a, 0, +a (symmetric Euler collinear)
    a = 1.0
    pos_euler = np.array([[-a, 0.0, 0.0], [0.0, 0.0, 0.0], [a, 0.0, 0.0]])

    # Angular velocity for the Euler collinear CC with G=1:
    # omega^2 = (1/a^3) * sum_{j!=i} m_j * |x_i - x_j|^{-2} * sign  (identical for all i by CC condition)
    # For equal masses at -a, 0, +a:  omega^2 = m*(1/a^2 + 1/(2a)^2) = 1.25*m/a^3
    # For unequal masses use the general formula from body 0 at -a:
    #   omega^2 * (-a) = -G * [ m1*(x0-x1)/|x0-x1|^3 + m2*(x0-x2)/|x0-x2|^3 ]
    # which gives omega^2 = G * ( m1/a^2 + m2/(2a)^2 ) / a  (forces on body 0)
    G = 1.0
    omega = np.sqrt(G * (masses[1] / a**2 + masses[2] / (2*a)**2))

    # Velocities for rigid rotation about the system CoM: v_i = omega × (r_i - CoM)
    com = np.sum(pos_euler * masses[:, None], axis=0) / masses.sum()
    vel_euler = np.zeros_like(pos_euler)
    for i in range(len(masses)):
        r_vec = pos_euler[i] - com
        vel_euler[i] = np.array([-omega * r_vec[1], omega * r_vec[0], 0.0])

    # Integration settings
    T_orbit = 2 * np.pi / omega
    run_problem('Euler collinear', pos_euler, vel_euler, masses, T_orbit, n_orbits=n_orbits, steps_per_orbit=1000, test_particle=test_particle)

def LagrangeCC(m, n_orbits=9, pert=0.001, steps_per_orbit=500, test_particle=False):
    """ 
    Normalized Lagrange equilateral triangle configuration for three bodies. 
    """
    masses = np.array(m, dtype=float)/sum(m)    # normalize masses to sum to 1, G=1 in the equations

    # Place three masses at the vertices of an equilateral triangle (side a=1)
    a = 1.0
    pos_lagrange = np.array([
        [0.0+pert, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.50, np.sqrt(3) / 2, 0.0]
    ], dtype=float)
    m = masses[0]
    omega = 1.0  # chosen for this normalized example
    com = np.sum(pos_lagrange * masses[:, None], axis=0) / np.sum(masses)
    vel_lagrange = np.zeros_like(pos_lagrange)
    for i in range(len(masses)):
        r_vec = pos_lagrange[i] - com
        vel_lagrange[i] = np.array([-omega * r_vec[1], omega * r_vec[0], 0.0])  # Perpendicular to r_vec for rigid rotation

    # Integration settings
    T_orbit = 2 * np.pi / omega
    run_problem('Lagrange Equilateral', pos_lagrange, vel_lagrange, masses, T_orbit, n_orbits=n_orbits, steps_per_orbit=500, test_particle=test_particle)

# now verify that this works for a 5 body model; use solar system as model
def problem_2a(n_orbits=1):
    # Example 5-body system: Sun, Jupiter, Saturn, Uranus, Neptune (normalized)
    masses = np.array([1.0, 9.5458e-4, 2.858e-4, 4.366e-5, 5.15e-5], dtype=float)
    masses /= masses.sum()

    # Approximate initial positions (AU) and velocities (AU/yr) for a snapshot in time
    pos = np.array([
        [0.0, 0.0, 0.0],  # Sun
        [5.2, 0.0, 0.0],   # Jupiter
        [9.5, 0.0, 0.0],   # Saturn
        [19.2, 0.0, 0.0],  # Uranus
        [30.1, 0.0, 0.0],  # Neptune
    ], dtype=float)

    # Circular velocities consistent with G=1:
    # v = sqrt(G * M_sun / r) = sqrt(masses[0] / r)
    G = 1.0
    vel = np.zeros_like(pos)
    for i in range(1, len(masses)):
        r = np.linalg.norm(pos[i] - pos[0])
        v_circ = np.sqrt(G * masses[0] / r)
        vel[i] = np.array([0.0, v_circ, 0.0])

    # Shift to center of mass frame
    com_pos = np.sum(pos * masses[:, None], axis=0) / masses.sum()
    com_vel = np.sum(vel * masses[:, None], axis=0) / masses.sum()
    pos -= com_pos
    vel -= com_vel

    # Integration settings: simulate for one Jupiter orbital period
    # With G=1: T = 2*pi * r^(3/2) / sqrt(G * M_sun)
    r_jupiter = 5.2
    T_orbit = 2 * np.pi * r_jupiter**1.5 / np.sqrt(G * masses[0])

    run_problem('5-body solar system', pos, vel, masses, T_orbit, n_orbits=n_orbits, steps_per_orbit=10000)


# Run the test for the 5BP using a pentagon configuration
def problem_2b(n_orbits=1):
    G = 1.0
    # Five equal masses at the vertices of a regular pentagon, with velocities for rigid rotation
    N = 5
    masses = np.ones(N) / N
    radius = 1.0
    pos_pentagon = np.array([[radius * np.cos(2 * np.pi * i / N), radius * np.sin(2 * np.pi * i / N), 0.0] for i in range(N)])
    omega = np.sqrt(G * masses.sum() / radius**3)
    vel_pentagon = np.zeros_like(pos_pentagon)
    for i in range(N):
        r_vec = pos_pentagon[i]
        vel_pentagon[i] = np.array([-omega * r_vec[1], omega * r_vec[0], 0.0])  # Perpendicular to r_vec for rigid rotation

    T_orbit = 2 * np.pi / omega

    run_problem('Pentagon configuration', pos_pentagon, vel_pentagon, masses, T_orbit, n_orbits=n_orbits, steps_per_orbit=10000)

# place 5 bodies on eccentric orbit = 0.1 around the center of mass 
# with equidistant mean anomalies and velocities for rigid rotation; simulate and check conservation
# start simulation at periapsis for 1st body and compute initial conditions for the others accordingly
def problem_2c(n_orbits=1, eccentricity=0.1):
    G = 1.0
    N = 5
    masses = np.ones(N) / N
    a = 1.0  # semi-major axis
    e = eccentricity  # eccentricity

    # Compute positions and velocities for each body on the eccentric orbit
    mu = G * masses.sum()
    h = np.sqrt(mu * a * (1 - e**2))  # specific angular momentum
    pos_eccentric = np.zeros((N, 3))
    vel_eccentric = np.zeros((N, 3))
    for i in range(N):
        M = 2 * np.pi * i / N  # mean anomaly
        E = M  # initial guess for Newton's method
        for _ in range(20):  # Newton's method to solve Kepler's equation: M = E - e*sin(E)
            f = E - e * np.sin(E) - M
            f_prime = 1 - e * np.cos(E)
            E -= f / f_prime

        # True anomaly and radius
        theta = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
        r = a * (1 - e**2) / (1 + e * np.cos(theta))

        pos_eccentric[i] = [r * np.cos(theta), r * np.sin(theta), 0.0]

        # Keplerian velocity components (perifocal frame)
        vx = -mu / h * np.sin(theta)
        vy =  mu / h * (e + np.cos(theta))
        vel_eccentric[i] = [vx, vy, 0.0]

    # Shift to center-of-mass frame
    com_pos = np.sum(pos_eccentric * masses[:, None], axis=0) / masses.sum()
    com_vel = np.sum(vel_eccentric * masses[:, None], axis=0) / masses.sum()
    pos_eccentric -= com_pos
    vel_eccentric -= com_vel

    T_orbit = 2 * np.pi * a**1.5 / np.sqrt(mu)

    run_problem('Eccentric configuration', pos_eccentric, vel_eccentric, masses, T_orbit, n_orbits=n_orbits, steps_per_orbit=10000)

def problem_2d_flyby(scenario='A', steps_per_orbit=50):
    """
    2-body + 3-body mutual orbit & flyby scenarios.

    A binary (bodies 0,1) and a triple (bodies 2,3,4) are each set up
    in internal circular/rigid-rotation orbits, then placed on a mutual
    Keplerian orbit about the overall system CoM.

    Parameters (per scenario)
    -------------------------
    D0    : semi-major axis of the mutual orbit (CoM-to-CoM separation
            at periapsis when e_mut=0, i.e. circular)
    b     : initial true anomaly (radians) on the mutual orbit
    ecc   : eccentricity of the mutual orbit (0 = circular, >0 = eccentric)

    Scenarios
    ---------
    A : Circular mutual orbit (e=0)  — stable initial hierarchy
    B : Mildly eccentric (e=0.3)     — periodic close approaches
    C : Moderately eccentric (e=0.5) — stronger tidal pumping
    D : Highly eccentric (e=0.8)     — deep plunge, likely disruption
    E : Retrograde triple, circular  — opposite spin directions
    """
    G = 1.0

    # ---------- scenario parameters ----------
    # D0 = mutual semi-major axis, b = initial true anomaly (rad),
    # ecc = mutual eccentricity
    configs = {
        'A': dict(m=np.array([0.20, 0.20, 0.20, 0.20, 0.20]),
                  a_bin=2.0, a_trip=1.0, D0=8.0, b=0.0, ecc=0.0,
                  triple_type='lagrange', retrograde=False,
                  title='Scenario A: Circular mutual orbit (e=0)'),
        'B': dict(m=np.array([0.20, 0.20, 0.20, 0.20, 0.20]),
                  a_bin=2.0, a_trip=1.0, D0=8.0, b=0.0, ecc=0.3,
                  triple_type='lagrange', retrograde=False,
                  title='Scenario B: Mildly eccentric (e=0.3)'),
        'C': dict(m=np.array([0.20, 0.20, 0.20, 0.20, 0.20]),
                  a_bin=2.0, a_trip=1.0, D0=8.0, b=0.0, ecc=0.5,
                  triple_type='lagrange', retrograde=False,
                  title='Scenario C: Moderately eccentric (e=0.5)'),
        'D': dict(m=np.array([0.20, 0.20, 0.20, 0.20, 0.20]),
                  a_bin=2.0, a_trip=1.0, D0=8.0, b=0.0, ecc=0.8,
                  triple_type='lagrange', retrograde=False,
                  title='Scenario D: Highly eccentric (e=0.8)'),
        'E': dict(m=np.array([0.20, 0.20, 0.20, 0.20, 0.20]),
                  a_bin=2.0, a_trip=1.0, D0=8.0, b=0.0, ecc=0.0,
                  triple_type='lagrange', retrograde=True,
                  title='Scenario E: Retrograde triple, circular'),
    }
    cfg = configs[scenario]
    m     = cfg['m']
    a_bin = cfg['a_bin']
    a_trip = cfg['a_trip']
    D0    = cfg['D0']
    b     = cfg['b']
    ecc   = cfg['ecc']

    m1, m2 = m[0], m[1]
    m3, m4, m5 = m[2], m[3], m[4]
    M_bin  = m1 + m2
    M_trip = m3 + m4 + m5
    M_tot  = M_bin + M_trip

    # ========== BINARY (bodies 0,1) centered at origin ==========
    # Circular orbit: relative separation a_bin, relative speed v_rel
    v_rel_bin = np.sqrt(G * M_bin / a_bin)
    pos_bin = np.array([
        [-m2 / M_bin * a_bin, 0.0, 0.0],
        [ m1 / M_bin * a_bin, 0.0, 0.0],
    ])
    vel_bin = np.array([
        [0.0, -m2 / M_bin * v_rel_bin, 0.0],
        [0.0,  m1 / M_bin * v_rel_bin, 0.0],
    ])
    T_bin = 2 * np.pi * a_bin**1.5 / np.sqrt(G * M_bin)

    # ========== TRIPLE (bodies 2,3,4) centered at origin ==========
    if cfg['triple_type'] == 'hierarchical':
        # Inner binary (m3, m4) with separation a_inner, outer body (m5)
        a_inner = 0.2
        a_outer = a_trip
        M_inner = m3 + m4

        v_rel_inner = np.sqrt(G * M_inner / a_inner)
        pos_trip = np.array([
            [-m4 / M_inner * a_inner, 0.0, 0.0],
            [ m3 / M_inner * a_inner, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        vel_trip = np.array([
            [0.0, -m4 / M_inner * v_rel_inner, 0.0],
            [0.0,  m3 / M_inner * v_rel_inner, 0.0],
            [0.0, 0.0, 0.0],
        ])

        # Outer body orbits the inner-pair CoM (at origin) at distance a_outer
        phase_out = np.pi / 3          # start outer body at 60 deg
        v_out_mag = np.sqrt(G * M_trip / a_outer)
        r_out = a_outer * np.array([np.cos(phase_out), np.sin(phase_out), 0.0])
        v_out = v_out_mag * np.array([-np.sin(phase_out), np.cos(phase_out), 0.0])

        # Shift inner pair so that the whole-triple CoM is at origin
        shift_pos = -m5 / M_trip * r_out
        shift_vel = -m5 / M_trip * v_out
        pos_trip[0] += shift_pos;  vel_trip[0] += shift_vel
        pos_trip[1] += shift_pos;  vel_trip[1] += shift_vel
        pos_trip[2] = M_inner / M_trip * r_out
        vel_trip[2] = M_inner / M_trip * v_out

        T_trip = 2 * np.pi * a_outer**1.5 / np.sqrt(G * M_trip)
    else:
        # Lagrange equilateral triangle, side = a_trip
        masses_trip = np.array([m3, m4, m5])
        verts = np.array([
            [0.0, 0.0, 0.0],
            [a_trip, 0.0, 0.0],
            [a_trip / 2, a_trip * np.sqrt(3) / 2, 0.0],
        ])
        com_trip = np.sum(verts * masses_trip[:, None], axis=0) / M_trip
        verts -= com_trip

        omega_trip = np.sqrt(G * M_trip / a_trip**3)
        if cfg['retrograde']:
            omega_trip = -omega_trip

        vel_trip = np.zeros_like(verts)
        for i in range(3):
            vel_trip[i] = np.array([-omega_trip * verts[i, 1],
                                     omega_trip * verts[i, 0], 0.0])
        pos_trip = verts
        T_trip = 2 * np.pi / abs(omega_trip)

    # ========== MUTUAL ORBIT ==========
    # Place binary CoM and triple CoM on a circular orbit about the
    # overall system CoM, at separation D0.  The parameter b is the
    # initial true anomaly (radians) on that mutual orbit; ecc is
    # the eccentricity of the mutual orbit.
    e_mut = ecc             # mutual orbit eccentricity
    theta0 = b             # reinterpret: initial true anomaly (rad)
    mu_mut = G * M_tot     # gravitational parameter for mutual orbit

    # Semi-latus rectum and radial distance at theta0
    p_mut = D0 * (1 - e_mut**2) if e_mut < 1.0 else D0 * (e_mut**2 - 1)
    r_mut = p_mut / (1 + e_mut * np.cos(theta0))

    # Mutual orbit velocity components (perifocal frame)
    h_mut = np.sqrt(mu_mut * p_mut)
    vr_mut = mu_mut / h_mut * e_mut * np.sin(theta0)
    vt_mut = mu_mut / h_mut * (1 + e_mut * np.cos(theta0))

    # Position and velocity of triple CoM relative to binary CoM
    R_sep = r_mut * np.array([np.cos(theta0), np.sin(theta0), 0.0])
    rhat = np.array([np.cos(theta0), np.sin(theta0), 0.0])
    that = np.array([-np.sin(theta0), np.cos(theta0), 0.0])
    V_sep = vr_mut * rhat + vt_mut * that

    # Distribute to each subsystem CoM (CoM frame)
    r_bin_com  = -M_trip / M_tot * R_sep
    r_trip_com =  M_bin  / M_tot * R_sep
    v_bin_com  = -M_trip / M_tot * V_sep
    v_trip_com =  M_bin  / M_tot * V_sep

    pos_bin  += r_bin_com;   vel_bin  += v_bin_com
    pos_trip += r_trip_com;  vel_trip += v_trip_com

    pos = np.vstack([pos_bin, pos_trip])
    vel = np.vstack([vel_bin, vel_trip])

    # ========== INTEGRATION ==========
    T_mut = 2 * np.pi * (D0 if e_mut < 1 else p_mut)**1.5 / np.sqrt(mu_mut)
    T_ref = min(T_bin, T_trip)
    T_total = T_mut  / 2     # simulate ~2 mutual orbits
    n_orbits = T_total / T_ref

    print(f"\n{'='*60}")
    print(f"{cfg['title']}")
    print(f"  masses       = {m}")
    print(f"  T_binary     = {T_bin:.3f},  T_triple = {T_trip:.3f}")
    print(f"  D0={D0}, e_mut={e_mut}, theta0={theta0:.2f} rad")
    print(f"  T_mutual     = {T_mut:.3f}")
    print(f"  T_total      = {T_total:.1f}  ({n_orbits:.1f} ref orbits)")
    print(f"{'='*60}")

    run_problem(cfg['title'], pos, vel, m, T_ref,
                n_orbits=n_orbits, steps_per_orbit=steps_per_orbit)


if __name__ == '__main__':
  #  Euler_collinear_acceleration(m=[5/8, 1/4, 1/8], n_orbits=0.25, test_particle=True)

    # LagrangeCC([1/2, 1/3, 1/6], n_orbits=3, test_particle=True, pert=0.001)
    # Routhy Stability
    #     m_sun = 1.0,  m_earth = 3.003489614e-6, m_moon = 3.694e-8
    m1 = 1.0
    m2 = 0.03   # Earth/Sun mass ratio
    m3 = 3.0e-4          # Moon/Sun mass ratio (approx)
    routhy = m1*m2 + m1*m3 + m2*m3
    print(f"Routh's criterion for stability of the Lagrange points: m1*m2 + m1*m3 + m2*m3 = {routhy:.3e} < 0.03852 => L4/L5 are stable")
    LagrangeCC([m1,m2,m3], pert=0.005, n_orbits=1, steps_per_orbit=500, test_particle=True)
 #   problem_2a(n_orbits=2)
#    problem_2b(n_orbits=0.1)
#    problem_2c(n_orbits=4, eccentricity=0.3)
  #  problem_2d_flyby(scenario='E', steps_per_orbit=1000)
