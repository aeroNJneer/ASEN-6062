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

# generate the EOM using jacobi coordinates
def equations_of_motion(t, y, masses):
    """
    Compute the derivatives for the N-body problem in Jacobi coordinates.

    Parameters
    ----------
    t : float
        Time variable (not used in this function but required for ODE solvers).
    y : ndarray, shape (2*N*3,)
        State vector containing positions and velocities in Jacobi coordinates.
    masses : ndarray, shape (N,)
        Masses of the N bodies.
    Returns

    -------
    dydt : ndarray, shape (2*N*3,)
        Derivative of the state vector.
    """
    N = len(masses)

    # State `y` is in Jacobi coordinates: first N*3 entries are Jacobi positions
    # and the next N*3 entries are Jacobi velocities.
    jac_pos = y[:N*3].reshape((N, 3))
    jac_vel = y[N*3:].reshape((N, 3))

    dydt = np.zeros_like(y)

    # Gravitational constant in AU^3 / (yr^2 * solar_mass)
    G = 1.0

    # Convert Jacobi positions -> Cartesian positions to evaluate pairwise forces
    cart_positions = positions_from_jacobi(jac_pos, masses)

    # Compute Cartesian accelerations from pairwise gravity
    cart_acc = np.zeros((N, 3))
    for i in range(N):
        a_i = np.zeros(3)
        for j in range(N):
            if i == j:
                continue
            r_vec = cart_positions[j] - cart_positions[i]
            r_mag = np.linalg.norm(r_vec)
            # guard against zero separation
            if r_mag == 0:
                continue
            a_i += G * masses[j] * r_vec / r_mag**3
        cart_acc[i] = a_i

    # Map Cartesian accelerations into Jacobi accelerations using the linear
    # transform L where jac_pos = L @ cart_positions (per-coordinate)
    L = jacobi_linear_transform(masses)
    # The mapping L is applied per Cartesian component (x, y, z). Because
    # the Jacobi transform is linear in positions, the same L maps the
    # Cartesian accelerations into Jacobi accelerations component-wise.
    jac_acc = L.dot(cart_acc)

    # derivatives: jacobi positions' derivative is jacobi velocities
    dydt[:N*3] = jac_vel.flatten()
    # jacobi velocities' derivative is jacobi accelerations
    dydt[N*3:] = jac_acc.flatten()

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
    Simulate the N-body problem using Jacobi coordinates.

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

    # Convert initial conditions to Jacobi coordinates for integration
    jacobi_pos = jacobi_coordinates(initial_positions, masses)
    jacobi_vel = jacobi_coordinates(initial_velocities, masses)

    # Flatten initial conditions for ODE solver (Jacobi state)
    y0 = np.hstack((jacobi_pos.flatten(), jacobi_vel.flatten()))

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

    N = len(masses)
    times = sol.t
    M = len(times)

    # Convert Jacobi solution back to Cartesian coordinates for each time
    positions = np.zeros((M, N, 3))
    velocities = np.zeros((M, N, 3))
    for k in range(M):
        jac_pos_k = sol.y[:N*3, k].reshape((N, 3))
        jac_vel_k = sol.y[N*3:, k].reshape((N, 3))
        positions[k] = positions_from_jacobi(jac_pos_k, masses)
        velocities[k] = velocities_from_jacobi(jac_vel_k, masses)

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

# plot trajectories of the bodies
def plot_trajectories(pos_ts, masses):
    plt.figure()
    labels = [f'Body {i+1}' for i in range(len(masses))]
    for i in range(len(masses)):
        plt.plot(pos_ts[:, i, 0], pos_ts[:, i, 1], label=labels[i])
    plt.axis('equal')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Trajectories of bodies in the N-body simulation')

#plot time series of energy and angular momentum
def plot_conservation(times, pos_ts, vel_ts, masses, energies=None, angular_momenta=None):
    """
    Plot residuals and fractional (relative) changes of total energy and
    angular momentum. If `energies` and `angular_momenta` are provided they
    will be used; otherwise they are recomputed from the time series.
    """
    if energies is None:
        energies = np.array([compute_energy(pos_ts[k], vel_ts[k], masses) for k in range(len(times))])
    if angular_momenta is None:
        angular_momenta = np.array([compute_angular_momentum(pos_ts[k], vel_ts[k], masses) for k in range(len(times))])

    # Magnitudes and residuals
    E_vals = energies
    L_vals = np.linalg.norm(angular_momenta, axis=1)
    E0 = E_vals[0]
    L0 = L_vals[0]
    E_res = E_vals - E0
    L_res = L_vals - L0
    E_frac = E_res / np.abs(E0) if np.abs(E0) > 0 else E_res
    L_frac = L_res / np.abs(L0) if np.abs(L0) > 0 else L_res

    plt.figure(figsize=(12, 8))

    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(times, E_res, label='E - E0')
    ax1.plot(times, L_res, label='|L| - |L0|')
    ax1.set_xlabel('Time (yr)')
    ax1.set_ylabel('Residual')
    ax1.set_title('Residuals of Energy and Angular Momentum')
    ax1.grid()
    ax1.legend()
    try:
        ax1.yaxis.get_major_formatter().set_useOffset(False)
    except Exception:
        pass

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(times, E_frac, label='ΔE / |E0|')
    ax2.plot(times, L_frac, label='Δ|L| / |L0|')
    ax2.set_xlabel('Time (yr)')
    ax2.set_ylabel('Fractional Change')
    ax2.set_title('Fractional Change of Energy and Angular Momentum')
    ax2.grid()
    ax2.legend()
    try:
        ax2.yaxis.get_major_formatter().set_useOffset(False)
    except Exception:
        pass

    plt.tight_layout()
    # Do not call plt.show() here; caller will display all figures.

def Euler_collinear_acceleration():

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

    # Integration settings: simulate for one rotation period
    T = 2 * np.pi / omega
    t_span = (0.0, T)
    dt = T / 500.0

    times, pos_ts, vel_ts, energies, angular_momenta = simulate_and_check_conservation(pos_euler, vel_euler, masses, t_span, dt)

    # print max and min energy and angular momentum to check conservation
    print(f"Energy: min={energies.min():.3e}, max={energies.max():.3e}, relative change={(energies.max() - energies.min()) / np.abs(energies[0]):.3e}")
    L_mags = np.linalg.norm(angular_momenta, axis=1)

    plot_trajectories(pos_ts, masses)
    plot_conservation(times, pos_ts, vel_ts, masses, energies, angular_momenta)

    # Show all figures together (single blocking call)
    plt.show()


def problem_1b():
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

    # Integration settings: simulate for one rotation period
    T = 2 * np.pi / omega * 10
    t_span = (0.0, T)
    dt = T / 500.0

    times, pos_ts, vel_ts, energies, angular_momenta = simulate_and_check_conservation(pos_lagrange, vel_lagrange, masses, t_span, dt)
    
    plot_trajectories(pos_ts, masses)
    plot_conservation(times, pos_ts, vel_ts, masses, energies, angular_momenta)
    plt.show()


def problem_1c():
    """
    Set up a normalized Sun/Earth/Moon system and run a short simulation.

    Normalization used:
      - Distances in AU, time in years, gravitational constant G=1
      - Masses are given as mass ratios relative to the Sun and then
        normalized so they sum to 1 (to match the code's G=1 convention).

    This function places Earth at ~1 AU from the Sun and the Moon at the
    usual Earth–Moon separation in AU, computes circular velocities for
    Earth about the Sun and Moon about the Earth (approximate), then
    adjusts the Sun velocity to ensure zero total linear momentum.
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
    r_moon = np.array([0.7, np.sqrt(3) / 2, 0.0])

    # shift positions so center of mass is at origin
    com = (masses[0] * r_sun + masses[1] * r_earth + masses[2] * r_moon) / masses.sum()
    pos = np.vstack([r_sun, r_earth, r_moon]) - com

    # Gravitational constant used in EOM is G=1 (code uses G=1), so use
    # that when computing approximate circular speeds.
    G = 1.0

    # Earth circular speed about Sun (approx, using Sun mass only)
    r_ES = np.linalg.norm(pos[1] - pos[0])
    omega_earth = np.sqrt(G * masses[0] / r_ES**3)
    v_earth = omega_earth * np.array([- (pos[1] - pos[0])[1], (pos[1] - pos[0])[0], 0.0])

    # Moon speed relative to Earth (approx, using Earth's mass)
    r_ME = np.linalg.norm(pos[2] - pos[1])
    omega_moon = np.sqrt(G * masses[0] / r_ES**3) if r_ME > 0 else 0.0
    v_moon_rel = omega_moon * np.array([-(pos[2] - pos[1])[1], (pos[2] - pos[1])[0], 0.0])

    # Inertial velocities: Earth moves with v_earth, Moon moves with v_earth + v_moon_rel
    vel = np.zeros_like(pos)
    vel[1] = v_earth
    vel[2] = v_moon_rel

    # Set Sun velocity so total linear momentum is zero
    vel[0] = - (masses[1] * vel[1] + masses[2] * vel[2]) / masses[0]

    # Integration settings: simulate one Earth orbital period
    T = 2 * np.pi / omega_earth if omega_earth > 0 else 3
    t_span = (0.0, T)
    dt = T / 1000.0

    times, pos_ts, vel_ts, energies, angular_momenta = simulate_and_check_conservation(pos, vel, masses, t_span, dt)

    # Print quick diagnostics
    L_mags = np.linalg.norm(angular_momenta, axis=1)
    print(f"Earth–Moon–Sun simulation: T={T:.3f} (normalized units)")
    print(f"Energy: min={energies.min():.3e}, max={energies.max():.3e}, rel change={(energies.max()-energies.min())/np.abs(energies[0]):.3e}")
    print(f"Angular Momentum: min={L_mags.min():.3e}, max={L_mags.max():.3e}, rel change={(L_mags.max()-L_mags.min())/np.abs(L_mags[0]):.3e}")

    # Plotting
    plot_trajectories(pos_ts, masses)
    plot_conservation(times, pos_ts, vel_ts, masses, energies, angular_momenta)
    plt.show()

# now verify that this works for a 5 body model; use solar system as model
def problem_2a():
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
    T = 2 * np.pi * r_jupiter**1.5 / np.sqrt(G * masses[0])
    t_span = (0.0, T)
    dt = T / 10000

    times, pos_ts, vel_ts, energies, angular_momenta = simulate_and_check_conservation(pos, vel, masses, t_span, dt)

    print(f"5-body simulation: T={T:.3f} (normalized units)")
    print(f"Energy: min={energies.min():.3e}, max={energies.max():.3e}, rel change={(energies.max()-energies.min())/np.abs(energies[0]):.3e}")
    L_mags = np.linalg.norm(angular_momenta, axis=1)   
    print(f"Angular Momentum: min={L_mags.min():.3e}, max={L_mags.max():.3e}, rel change={(L_mags.max()-L_mags.min())/np.abs(L_mags[0]):.3e}")
    plot_trajectories(pos_ts, masses)
    plot_conservation(times, pos_ts, vel_ts, masses, energies, angular_momenta)
    plt.show()


# Run the test for the 5BP using a pentagon configuration
def problem_2b():
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

    T = 2 * np.pi / omega
    t_span = (0.0, T)
    dt = T / 10000

    times, pos_ts, vel_ts, energies, angular_momenta = simulate_and_check_conservation(pos_pentagon, vel_pentagon, masses, t_span, dt)

    print(f"Pentagon configuration: T={T:.3f} (normalized units)")
    print(f"Energy: min={energies.min():.3e}, max={energies.max():.3e}, rel change={(energies.max()-energies.min())/np.abs(energies[0]):.3e}")
    L_mags = np.linalg.norm(angular_momenta, axis=1)   
    print(f"Angular Momentum: min={L_mags.min():.3e}, max={L_mags.max():.3e}, rel change={(L_mags.max()-L_mags.min())/np.abs(L_mags[0]):.3e}")
    plot_trajectories(pos_ts, masses)
    plot_conservation(times, pos_ts, vel_ts, masses, energies, angular_momenta)
    plt.show()

# place 5 bodies on eccentric orbit = 0.1 around the center of mass 
# with equidistant mean anomalies and velocities for rigid rotation; simulate and check conservation
# start simulation at periapsis for 1st body and compute initial conditions for the others accordingly
def problem_2c():
    G = 1.0
    N = 5
    masses = np.ones(N) / N
    a = 1.0  # semi-major axis
    e = 0.1  # eccentricity

    # Compute positions and velocities for each body on the eccentric orbit
    pos_eccentric = np.zeros((N, 3))
    vel_eccentric = np.zeros((N, 3))
    for i in range(N):
        M = 2 * np.pi * i / N  # mean anomaly
        E = 0  # start at periapsis for the first body, so E=0 for i=0; others will be solved for M = E - e*sin(E)
        for _ in range(10):  # Newton's method to solve Kepler's equation: M = E - e*sin(E)
            f = E - e * np.sin(E) - M
            f_prime = 1 - e * np.cos(E)
            E -= f / f_prime

        # True anomaly and radius
        theta = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
        r = a * (1 - e**2) / (1 + e * np.cos(theta))

        pos_eccentric[i] = [r * np.cos(theta), r * np.sin(theta), 0.0]

        # Velocity magnitude from vis-viva equation: v^2 = G*M*(2/r - 1/a)
        v_mag = np.sqrt(G * masses.sum() * (2 / r - 1 / a))
        vel_eccentric[i] = [-v_mag * np.sin(theta), v_mag * np.cos(theta), 0.0]

    T = 2 * np.pi * a**1.5 / np.sqrt(G * masses.sum())

    # use shorter time span
    T = np.pi
    t_span = (0.0, T)
    dt = T / 10000

    times, pos_ts, vel_ts, energies, angular_momenta = simulate_and_check_conservation(pos_eccentric, vel_eccentric, masses, t_span, dt)

    print(f"Eccentric configuration: T={T:.3f} (normalized units)")
    print(f"Energy: min={energies.min():.3e}, max={energies.max():.3e}, rel change={(energies.max()-energies.min())/np.abs(energies[0]):.3e}")
    L_mags = np.linalg.norm(angular_momenta, axis=1)
    print(f"Angular Momentum: min={L_mags.min():.3e}, max={L_mags.max():.3e}, rel change={(L_mags.max()-L_mags.min())/np.abs(L_mags[0]):.3e}")
    plot_trajectories(pos_ts, masses)
    plot_conservation(times, pos_ts, vel_ts, masses, energies, angular_momenta)
    plt.show()





problem_2c()