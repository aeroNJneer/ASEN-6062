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
    jacobi_pos = np.zeros_like(positions)

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
    # M[i] = sum of masses[0..i] (i.e., M_1, M_2, ..., M_N in Scheeres' notation)
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
    formulation in Jacobi coordinates (Scheeres, Jacobi_Notation.pdf).

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
    coordinates (Scheeres, Jacobi_Notation.pdf).

    The integration is performed on the N-1 relative Jacobi coordinates
    R_1, ..., R_{N-1} (the CoM coordinate R_0 is ignorable and fixed at the
    origin). The EOM are:

        eta_i * R_i'' = -dU/dR_i

    where eta_i = M_i * m_{i+1} / M_{i+1} are the Jacobi reduced masses.

    Parameters
    ----------
    initial_positions : ndarray, shape (N, 3)
        Initial Cartesian positions of N bodies in an inertial frame.
    initial_velocities : ndarray, shape (N, 3)
        Initial Cartesian velocities of N bodies in an inertial frame.
    masses : ndarray, shape (N,)
        Masses of the N bodies.
    t_span : tuple
        Time span for the simulation (t_start, t_end).
    dt : float
        Time step for the simulation.

    Returns
    -------
    times : ndarray
        Array of time points.
    positions : ndarray, shape (M, N, 3)
        Array of Cartesian positions at each time point.
    velocities : ndarray, shape (M, N, 3)
        Array of Cartesian velocities at each time point.
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
    if t1 <= t0:
        raise ValueError("t_span must have t_end > t_start")
    if dt <= 0:
        raise ValueError("dt must be positive")
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

def plot_results(times, pos_ts, vel_ts, masses, energies=None, angular_momenta=None, title=None):
    """
    Single-figure summary: trajectories, position components, and
    conservation residuals arranged as a 2x2 grid of subplots.

    Layout (2 rows x 2 columns):
        [0,0] x-y trajectory          [0,1] x(t)
        [1,0] residuals (E-E0, |L|-|L0|)  [1,1] y(t)
    """
    masses = np.asarray(masses, dtype=float)
    N = len(masses)
    labels = [f'Body {i+1} (m={masses[i]:.4g})' for i in range(N)]
    colors = plt.cm.tab10(np.linspace(0, 1, max(N, 10)))[:N]

    if energies is None:
        energies = np.array([compute_energy(pos_ts[k], vel_ts[k], masses) for k in range(len(times))])
    if angular_momenta is None:
        angular_momenta = np.array([compute_angular_momentum(pos_ts[k], vel_ts[k], masses) for k in range(len(times))])

    fig = plt.figure(figsize=(14, 8))
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.30, top=0.93)

    # --- (0,0) x-y trajectory ---
    ax_traj = fig.add_subplot(gs[0, 0])
    for i in range(N):
        ax_traj.plot(pos_ts[:, i, 0], pos_ts[:, i, 1], color=colors[i], lw=0.8, label=labels[i])
        ax_traj.plot(pos_ts[0, i, 0], pos_ts[0, i, 1], 'o', color=colors[i],
                     ms=4 + 6 * masses[i] / masses.max())
        ax_traj.plot(pos_ts[-1, i, 0], pos_ts[-1, i, 1], '*', color=colors[i],
                     ms=6 + 6 * masses[i] / masses.max())
    ax_traj.set_aspect('equal')
    ax_traj.legend(fontsize=5, loc='lower left')
    ax_traj.set_xlabel('x')
    ax_traj.set_ylabel('y')
    ax_traj.set_title('Trajectories (x-y plane)')
    ax_traj.grid(True, alpha=0.3)

    # --- (0,1) x(t) ---
    ax_x = fig.add_subplot(gs[0, 1])
    for i in range(N):
        ax_x.plot(times, pos_ts[:, i, 0], color=colors[i], lw=0.8, label=labels[i])
    ax_x.set_ylabel('x')
    ax_x.set_xlabel('Time')
    ax_x.set_title('x(t)')
    ax_x.legend(fontsize=5, loc='upper right')
    ax_x.grid(True, alpha=0.3)

    # --- Conservation quantities ---
    E_vals = energies
    L_vals = np.linalg.norm(angular_momenta, axis=1)
    E0, L0 = E_vals[0], L_vals[0]
    E_res, L_res = E_vals - E0, L_vals - L0

    # --- (1,0) residuals ---
    ax_res = fig.add_subplot(gs[1, 0])
    ax_res.plot(times, E_res, label='E - E0')
    ax_res.plot(times, L_res, label='|L| - |L0|')
    ax_res.set_xlabel('Time')
    ax_res.set_ylabel('Residual')
    ax_res.set_title('Conservation Residuals')
    ax_res.legend(fontsize=7)
    ax_res.grid(True, alpha=0.3)

    # --- (1,1) y(t) ---
    ax_y = fig.add_subplot(gs[1, 1])
    for i in range(N):
        ax_y.plot(times, pos_ts[:, i, 1], color=colors[i], lw=0.8, label=labels[i])
    ax_y.set_ylabel('y')
    ax_y.set_xlabel('Time')
    ax_y.set_title('y(t)')
    ax_y.legend(fontsize=5, loc='upper right')
    ax_y.grid(True, alpha=0.3)

def run_problem(name, pos, vel, masses, T_orbit, n_orbits=1, steps_per_orbit=500):
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
    """
    T = T_orbit * n_orbits
    t_span = (0.0, T)
    dt = T_orbit / steps_per_orbit
    times, pos_ts, vel_ts, energies, angular_momenta = simulate_and_check_conservation(
        pos, vel, masses, t_span, dt
    )
    E0 = energies[0]
    L_mags = np.linalg.norm(angular_momenta, axis=1)
    L0 = L_mags[0]
    print(f"{name}: T={T:.3f} ({n_orbits} orbits)")
    print(f"  Energy:  min={energies.min():.3e}, max={energies.max():.3e}, "
          f"rel change={(energies.max()-energies.min())/np.abs(E0):.3e}")
    print(f"  Ang Mom: min={L_mags.min():.3e}, max={L_mags.max():.3e}, "
          f"rel change={(L_mags.max()-L_mags.min())/np.abs(L0):.3e}")
    plot_results(times, pos_ts, vel_ts, masses, energies, angular_momenta, title=name)
    plt.show()

def Euler_collinear_acceleration(n_orbits=1):

    # TEST - use the Euler collinear central configuration for equal masses to verify the implementation
    # masses sum to 1, G=1 in the equations
    masses = [1/3, 1/3, 1/3]

    # Place three equal masses on the x-axis at -a, 0, +a (symmetric Euler collinear)
    a = 1.0
    pos_euler = np.array([[-a, 0.0, 0.0], [0.0, 0.0, 0.0], [a, 0.0, 0.0]])
    m = masses[0]
    omega = np.sqrt(1.25 * m / a**3)

    # Velocities for rigid rotation are perpendicular (y-direction): v = omega * x
    vel_euler = np.array([[0.0, -omega * a, 0.0], [0.0, 0.0, 0.0], [0.0, omega * a, 0.0]])

    # Integration settings
    T_orbit = 2 * np.pi / omega
    run_problem('Euler collinear', pos_euler, vel_euler, masses, T_orbit, n_orbits=n_orbits, steps_per_orbit=500)


def problem_1b(n_orbits=9):
    """ 
    Normalized Lagrange equilateral triangle configuration for three bodies. 
    """
    masses = np.array([1/2, 1/3, 1/6], dtype=float)

    # Place three masses at the vertices of an equilateral triangle (side a=1)
    a = 1.0
    pos_lagrange = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.50, np.sqrt(3) / 2, 0.0],
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
    run_problem('Lagrange Equilateral', pos_lagrange, vel_lagrange, masses, T_orbit, n_orbits=n_orbits, steps_per_orbit=500)


def problem_1c(n_orbits=1000, delta_pos_earth=None, delta_vel_earth=None):
    """
    Set up a normalized Sun/Earth/Moon system and run a simulation with 
    or without perturbations to Earth's orbit.

    Parameters
    ----------
    delta_pos_earth : ndarray, shape (3,), optional
        Position perturbation to add to Earth.
    delta_vel_earth : ndarray, shape (3,), optional
        Velocity perturbation to add to Earth.
    """

    # Raw mass ratios (relative to Solar mass)
    m_sun = 1.0
    m_earth = 3.003489614e-6   # Earth/Sun mass ratio
    m_moon = 3.694e-8          # Moon/Sun mass ratio (approx)

    masses = np.array([m_sun, m_earth, m_moon], dtype=float)
    masses /= masses.sum()

    # Positions: place Earth at x=1 AU, Moon offset from Earth by the
    # mean Earth-Moon distance (~384400 km = 0.00257 AU) along +y.
    a_earth = 1.0

    # provisional positions (Sun, Earth, Moon)
    r_sun = np.array([0.0, 0.0, 0.0])
    r_earth = np.array([a_earth, 0.0, 0.0])
    r_moon = np.array([a_earth, 0.00257, 0.0])  # Moon ~0.00257 AU from Earth in +y

    # shift positions so center of mass is at origin
    com = (masses[0] * r_sun + masses[1] * r_earth + masses[2] * r_moon) / masses.sum()
    pos = np.vstack([r_sun, r_earth, r_moon]) - com
    G = 1.0

    # Earth circular speed about Sun (approx, using Sun mass only)
    r_ES = np.linalg.norm(pos[1] - pos[0])
    omega_earth = np.sqrt(G * masses[0] / r_ES**3)
    v_earth = omega_earth * np.array([- (pos[1] - pos[0])[1], (pos[1] - pos[0])[0], 0.0])

    # Moon speed relative to Earth (circular orbit using Earth's mass)
    r_ME = np.linalg.norm(pos[2] - pos[1])
    omega_moon = np.sqrt(G * masses[1] / r_ME**3) if r_ME > 0 else 0.0
    delta_ME = pos[2] - pos[1]
    v_moon_rel = omega_moon * np.array([-delta_ME[1], delta_ME[0], 0.0])

    # Inertial velocities: Earth moves with v_earth, Moon moves with v_earth + v_moon_rel
    vel = np.zeros_like(pos)
    vel[1] = v_earth
    vel[2] = v_earth + v_moon_rel

    # Apply perturbations to Earth (and Moon, to keep relative offset) before enforcing zero total momentum
    if delta_pos_earth is not None:
        dp = np.asarray(delta_pos_earth, dtype=float) * pos[1]
        pos[1] += dp
        pos[2] += dp  # Moon moves with Earth
    if delta_vel_earth is not None:
        dv = np.asarray(delta_vel_earth, dtype=float)
        vel[1] += dv
        vel[2] += dv  # Moon's inertial velocity shifts with Earth's

    # Set Sun velocity so total linear momentum is zero
    vel[0] = - (masses[1] * vel[1] + masses[2] * vel[2]) / masses[0]

    # Integration settings
    T_orbit = 2 * np.pi / omega_earth
    run_problem('Sun-Earth-Moon', pos, vel, masses, T_orbit, n_orbits=n_orbits, steps_per_orbit=10)

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
def problem_2c(n_orbits=1):
    G = 1.0
    N = 5
    masses = np.ones(N) / N
    a = 1.0  # semi-major axis
    e = 0.1  # eccentricity

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



if __name__ == '__main__':
    Euler_collinear_acceleration()
    # Perturb Earth's position by 0.1 AU in y
    # problem_1c(delta_pos_earth=[0.20, 0.00, 0.0])
    # problem_2c()
    problem_2c()
