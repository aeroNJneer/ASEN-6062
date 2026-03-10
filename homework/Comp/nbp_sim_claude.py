"""
N-Body Problem Simulator in Jacobi Coordinates
===============================================
Dimensionless units: G = 1,  sum(masses) = 1.

Jacobi coordinate chain (N bodies):
  rho_0      = R  = overall center of mass  [inert in CoM frame]
  rho_k      = r_{k+1} - CoM(bodies 0..k)   for k = 1, ..., N-1

Reduced masses:
  eta_k = M_k * m_{k+1} / M_{k+1}
  where M_k = m_0 + ... + m_{k-1}  (cumulative, 1-indexed)

Equations of motion:
  eta_k * rho_k'' = dV/d(rho_k)  via chain rule from inertial accelerations

The EOM are derived by converting the full inertial gravitational
accelerations a_i = -dV/d(r_i) / m_i  back to Jacobi accelerations:

  rho_k'' = a_{k+1} - sum_{j=0}^{k} (m_j / M_k) * a_j

which is the standard Jacobi EOM result (see e.g. Murray & Dermott Ch.1).
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ============================================================================
# Jacobi <-> Inertial Conversions  (vectorised: works on shape (N,d) or (T,N,d))
# ============================================================================

def inertial_to_jacobi(masses, positions, velocities=None):
    """
    Convert inertial positions (and optional velocities) to Jacobi coords.

    Parameters
    ----------
    masses    : (N,)
    positions : (N, d)  or  (T, N, d)
    velocities: same shape as positions, optional

    Returns
    -------
    rho  : Jacobi positions, same shape as input.
           rho[0] = overall CoM
           rho[k] = r_k - CoM(0..k-1),  k >= 1
    drho : Jacobi velocities (only if velocities supplied)
    """
    masses = np.asarray(masses, dtype=float)
    pos = np.asarray(positions, dtype=float)
    N = len(masses)
    cum = np.cumsum(masses)          # cum[k] = m_0+...+m_k

    batch = pos.ndim == 3            # True if shape (T, N, d)
    axis = 1 if batch else 0         # body axis

    rho = np.zeros_like(pos)
    # rho[0] = CoM
    w = masses / cum[-1]             # shape (N,)
    if batch:
        rho[:, 0, :] = np.einsum('n,tnj->tj', w, pos)
    else:
        rho[0] = np.einsum('n,nj->j', w, pos)

    # Partial CoMs: pcom[k] = CoM(0..k)
    pcom = np.zeros_like(pos)
    if batch:
        pcom[:, 0, :] = pos[:, 0, :]
        for k in range(1, N):
            pcom[:, k, :] = (cum[k-1] * pcom[:, k-1, :] + masses[k] * pos[:, k, :]) / cum[k]
        for k in range(1, N):
            rho[:, k, :] = pos[:, k, :] - pcom[:, k-1, :]
    else:
        pcom[0] = pos[0]
        for k in range(1, N):
            pcom[k] = (cum[k-1] * pcom[k-1] + masses[k] * pos[k]) / cum[k]
        for k in range(1, N):
            rho[k] = pos[k] - pcom[k-1]

    if velocities is None:
        return rho
    vel = np.asarray(velocities, dtype=float)
    drho = np.zeros_like(vel)
    pcom_v = np.zeros_like(vel)
    if batch:
        drho[:, 0, :] = np.einsum('n,tnj->tj', w, vel)
        pcom_v[:, 0, :] = vel[:, 0, :]
        for k in range(1, N):
            pcom_v[:, k, :] = (cum[k-1] * pcom_v[:, k-1, :] + masses[k] * vel[:, k, :]) / cum[k]
        for k in range(1, N):
            drho[:, k, :] = vel[:, k, :] - pcom_v[:, k-1, :]
    else:
        drho[0] = np.einsum('n,nj->j', w, vel)
        pcom_v[0] = vel[0]
        for k in range(1, N):
            pcom_v[k] = (cum[k-1] * pcom_v[k-1] + masses[k] * vel[k]) / cum[k]
        for k in range(1, N):
            drho[k] = vel[k] - pcom_v[k-1]
    return rho, drho


def jacobi_to_inertial(masses, rho, drho=None):
    """
    Convert Jacobi coords to inertial positions (and optional velocities).
    CoM (rho[0]) is set to zero (CoM frame).

    Parameters
    ----------
    masses : (N,)
    rho    : (N, d) Jacobi positions  (rho[0] ignored; CoM fixed at 0)
    drho   : (N, d) Jacobi velocities, optional

    Returns
    -------
    pos  : (N, d) inertial positions
    vel  : (N, d) inertial velocities (only if drho supplied)
    """
    masses = np.asarray(masses, dtype=float)
    rho = np.asarray(rho, dtype=float)
    N = len(masses)
    cum = np.cumsum(masses)
    d = rho.shape[-1]

    pos = np.zeros_like(rho)
    # Partial CoM chain, with overall CoM = 0
    # rho[k] = pos[k] - pcom[k-1]   => pos[k] = rho[k] + pcom[k-1]
    # pcom[k] = (cum[k-1]*pcom[k-1] + masses[k]*pos[k]) / cum[k]
    # pcom[0] = pos[0]  (unknown), propagated linearly:
    #   pcom[k] = alpha[k] * pos[0] + beta[k]
    alpha = np.zeros((N, d))
    beta  = np.zeros((N, d))
    alpha[0] = 1.0
    beta[0]  = 0.0
    for k in range(1, N):
        pos_k_beta  = rho[k] + beta[k-1]
        pos_k_alpha = alpha[k-1]
        alpha[k] = (cum[k-1] * alpha[k-1] + masses[k] * pos_k_alpha) / cum[k]
        beta[k]  = (cum[k-1] * beta[k-1]  + masses[k] * pos_k_beta ) / cum[k]

    # CoM = 0 => alpha[N-1]*pos[0] + beta[N-1] = 0
    pos[0] = -beta[N-1] / alpha[N-1]   # (alpha[N-1] = 1 always)

    pcom = np.zeros_like(rho)
    pcom[0] = pos[0]
    for k in range(1, N):
        pos[k] = rho[k] + pcom[k-1]
        pcom[k] = (cum[k-1] * pcom[k-1] + masses[k] * pos[k]) / cum[k]

    if drho is None:
        return pos

    drho = np.asarray(drho, dtype=float)
    vel = np.zeros_like(drho)
    a_alpha = np.zeros((N, d))
    b_beta  = np.zeros((N, d))
    a_alpha[0] = 1.0
    for k in range(1, N):
        vel_k_b = drho[k] + b_beta[k-1]
        vel_k_a = a_alpha[k-1]
        a_alpha[k] = (cum[k-1] * a_alpha[k-1] + masses[k] * vel_k_a) / cum[k]
        b_beta[k]  = (cum[k-1] * b_beta[k-1]  + masses[k] * vel_k_b) / cum[k]

    vel[0] = -b_beta[N-1] / a_alpha[N-1]
    pcom_v = np.zeros_like(drho)
    pcom_v[0] = vel[0]
    for k in range(1, N):
        vel[k] = drho[k] + pcom_v[k-1]
        pcom_v[k] = (cum[k-1] * pcom_v[k-1] + masses[k] * vel[k]) / cum[k]

    return pos, vel


# ============================================================================
# Equations of Motion
# ============================================================================

def _inertial_accels(masses, pos):
    """
    Compute inertial gravitational accelerations for all N bodies. G=1.
    pos : (N, d)
    Returns accels : (N, d)
    """
    N = len(masses)
    accels = np.zeros_like(pos)
    for i in range(N):
        for j in range(i+1, N):
            dr = pos[j] - pos[i]
            dist3 = np.linalg.norm(dr) ** 3
            accels[i] += masses[j] * dr / dist3
            accels[j] -= masses[i] * dr / dist3
    return accels


def eom_jacobi(t, y, masses):
    """
    Equations of motion in Jacobi coordinates, CoM frame.

    State vector y = [rho_1x, rho_1y, rho_2x, rho_2y, ...,   (positions, skip rho_0=CoM=0)
                      drho_1x, drho_1y, ...]                   (velocities)

    We work with the N-1 non-trivial Jacobi coords (indices 1..N-1).
    """
    masses = np.asarray(masses, dtype=float)
    N = len(masses)
    d = 2                            # 2-D planar motion
    n_rel = N - 1                    # number of relative coords
    state_size = n_rel * d

    rho_rel  = y[:state_size].reshape(n_rel, d)   # rho[1..N-1]
    drho_rel = y[state_size:].reshape(n_rel, d)

    # Reconstruct full Jacobi array (rho[0]=0 in CoM frame)
    rho_full = np.zeros((N, d))
    rho_full[1:] = rho_rel

    # Convert to inertial
    pos = jacobi_to_inertial(masses, rho_full)

    # Inertial accelerations
    a = _inertial_accels(masses, pos)

    # Jacobi accelerations:
    # d^2 rho_k / dt^2 = a_{k} - sum_{j=0}^{k-1} (m_j / M_k) * a_j
    # where M_k = sum(masses[0..k-1])  (partial sum BEFORE body k)
    cum = np.cumsum(masses)          # cum[k] = m_0+...+m_k
    d2rho = np.zeros((N, d))
    for k in range(1, N):
        M_k = cum[k-1]               # total mass of bodies 0..k-1
        d2rho[k] = a[k] - np.einsum('j,jd->d', masses[:k], a[:k]) / M_k

    d2rho_rel = d2rho[1:]            # shape (N-1, d)

    return np.concatenate([drho_rel.ravel(), d2rho_rel.ravel()])


# ============================================================================
# Conserved Quantities
# ============================================================================

def compute_energy(masses, pos, vel):
    """Total energy E = T + V, G=1."""
    masses = np.asarray(masses, dtype=float)
    T = 0.5 * np.sum(masses[:, None] * vel**2)
    V = 0.0
    N = len(masses)
    for i in range(N):
        for j in range(i+1, N):
            V -= masses[i] * masses[j] / np.linalg.norm(pos[j] - pos[i])
    return T + V


def compute_Lz(masses, pos, vel):
    """Z-component of angular momentum."""
    masses = np.asarray(masses, dtype=float)
    Lz = np.sum(masses * (pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0]))
    return Lz


# ============================================================================
# Main Simulator
# ============================================================================

def simulate(masses, rho0, drho0, t_span, dt=0.01,
             method='DOP853', rtol=1e-10, atol=1e-12):
    """
    Integrate the N-body problem in Jacobi coordinates.

    Parameters
    ----------
    masses  : array-like (N,), must sum to 1
    rho0    : (N-1, 2) initial Jacobi RELATIVE positions (rho[1..N-1])
              rho[k] = r_{k+1} - CoM(bodies 0..k)
    drho0   : (N-1, 2) initial Jacobi relative velocities
    t_span  : (t_start, t_end)
    dt      : output cadence

    Returns
    -------
    dict with:
      t       : (T,) time array
      pos     : (T, N, 2) inertial positions
      vel     : (T, N, 2) inertial velocities
      rho     : (T, N-1, 2) Jacobi relative positions
      drho    : (T, N-1, 2) Jacobi relative velocities
      energy  : (T,) total energy
      Lz      : (T,) angular momentum z-component
    """
    masses = np.asarray(masses, dtype=float)
    N = len(masses)
    assert abs(masses.sum() - 1.0) < 1e-12, f"Masses must sum to 1, got {masses.sum()}"

    rho0  = np.asarray(rho0,  dtype=float)   # (N-1, 2)
    drho0 = np.asarray(drho0, dtype=float)
    assert rho0.shape == (N-1, 2) and drho0.shape == (N-1, 2), \
        f"rho0 and drho0 must be shape ({N-1}, 2)"

    y0 = np.concatenate([rho0.ravel(), drho0.ravel()])
    t_eval = np.arange(t_span[0], t_span[1], dt)

    sol = solve_ivp(eom_jacobi, t_span, y0, args=(masses,),
                    method=method, t_eval=t_eval,
                    rtol=rtol, atol=atol, dense_output=False)

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    T_steps = len(sol.t)
    d = 2
    n_rel = N - 1
    state_size = n_rel * d

    rho_rel  = sol.y[:state_size].T.reshape(T_steps, n_rel, d)
    drho_rel = sol.y[state_size:].T.reshape(T_steps, n_rel, d)

    # Reconstruct inertial positions and velocities
    pos_all = np.zeros((T_steps, N, d))
    vel_all = np.zeros((T_steps, N, d))
    for i in range(T_steps):
        rho_full  = np.zeros((N, d));  rho_full[1:]  = rho_rel[i]
        drho_full = np.zeros((N, d));  drho_full[1:] = drho_rel[i]
        pos_all[i], vel_all[i] = jacobi_to_inertial(masses, rho_full, drho_full)

    energy = np.array([compute_energy(masses, pos_all[i], vel_all[i])
                       for i in range(T_steps)])
    Lz = np.array([compute_Lz(masses, pos_all[i], vel_all[i])
                   for i in range(T_steps)])

    return dict(t=sol.t, pos=pos_all, vel=vel_all,
                rho=rho_rel, drho=drho_rel,
                energy=energy, Lz=Lz, masses=masses)


# ============================================================================
# Plotting
# ============================================================================

COLORS = ['#58a6ff', '#f78166', '#3fb950', '#e3b341', '#d2a8ff',
          '#79c0ff', '#ffa657', '#56d364', '#ff7b72', '#bc8cff']

def plot_results(result, title="N-Body Problem", save_path=None):
    N = result['pos'].shape[1]
    fig = plt.figure(figsize=(16, 10), facecolor='#0d1117')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    def style_ax(ax, ttl):
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e', labelsize=8)
        for sp in ax.spines.values(): sp.set_color('#30363d')
        ax.set_title(ttl, color='#e6edf3', fontsize=11, fontweight='bold')
        ax.grid(True, color='#21262d', linewidth=0.5)

    # Trajectories
    ax_t = fig.add_subplot(gs[0:2, 0:2])
    style_ax(ax_t, f"Orbital Trajectories — {N} Bodies (Inertial Frame)")
    for i in range(N):
        r = result['pos'][:, i, :]
        c = COLORS[i % len(COLORS)]
        m = result['masses'][i]
        ax_t.plot(r[:, 0], r[:, 1], color=c, lw=0.8, alpha=0.85, label=f'Body {i+1} (m={m:.3f})')
        ax_t.plot(*r[0],  'o', color=c, ms=5 + 6*m)
        ax_t.plot(*r[-1], '*', color=c, ms=8 + 6*m)
    ax_t.set_xlabel('x', color='#8b949e', fontsize=9)
    ax_t.set_ylabel('y', color='#8b949e', fontsize=9)
    ax_t.legend(facecolor='#161b22', edgecolor='#30363d',
                labelcolor='#e6edf3', fontsize=8)
    ax_t.set_aspect('equal', adjustable='datalim')

    # Energy
    ax_E = fig.add_subplot(gs[0, 2])
    style_ax(ax_E, "Energy Conservation")
    E0 = result['energy'][0]
    ax_E.plot(result['t'], (result['energy'] - E0) / abs(E0), color='#e3b341', lw=0.8)
    ax_E.set_xlabel('t', color='#8b949e', fontsize=9)
    ax_E.set_ylabel('(E − E₀)/|E₀|', color='#8b949e', fontsize=9)
    ax_E.ticklabel_format(axis='y', style='sci', scilimits=(-4, 4))

    # Angular momentum
    ax_L = fig.add_subplot(gs[1, 2])
    style_ax(ax_L, "Angular Momentum Conservation")
    L0 = result['Lz'][0]
    dL = result['Lz'] - L0
    rel_L = dL / abs(L0) if abs(L0) > 1e-14 else dL
    ax_L.plot(result['t'], rel_L, color='#d2a8ff', lw=0.8)
    ax_L.set_xlabel('t', color='#8b949e', fontsize=9)
    ax_L.set_ylabel('(L − L₀)/|L₀|' if abs(L0) > 1e-14 else 'L − L₀',
                    color='#8b949e', fontsize=9)
    ax_L.ticklabel_format(axis='y', style='sci', scilimits=(-4, 4))

    fig.suptitle(title, color='#e6edf3', fontsize=14, fontweight='bold', y=0.98)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Saved → {save_path}")
    plt.show()
    return fig


# ============================================================================
# Built-in Scenarios
# ============================================================================

def scenario_figure8():
    """Classic figure-8 choreography (N=3, equal masses)."""
    N = 3
    m = 1.0 / N
    masses = np.full(N, m)
    s = N ** (1/3)
    r = np.array([[-0.97000436,  0.24308753],
                  [ 0.0,         0.0        ],
                  [ 0.97000436, -0.24308753]]) / s
    v3 = np.array([-0.93240737/2, -0.86473146/2]) / s
    v = np.array([v3, -2*v3, v3])
    rho_full, drho_full = inertial_to_jacobi(masses, r, v)
    return dict(masses=masses,
                rho0=rho_full[1:], drho0=drho_full[1:],
                t_span=(0, 6.3259), dt=0.005,
                label="Figure-8 Choreography (N=3)")


def scenario_hierarchical_triple():
    """Tight inner binary + distant outer body (N=3)."""
    masses = np.array([0.45, 0.45, 0.10])
    m1, m2, m3 = masses
    a_in  = 1.0;  v_in  = np.sqrt((m1+m2) / a_in)
    a_out = 5.0;  v_out = np.sqrt(1.0 / a_out)
    rho0  = np.array([[a_in,  0.0],
                      [a_out, 0.0]])
    drho0 = np.array([[0.0, v_in],
                      [0.0, v_out]])
    return dict(masses=masses, rho0=rho0, drho0=drho0,
                t_span=(0, 60), dt=0.05,
                label="Hierarchical Triple (N=3)")


def scenario_5body_ring():
    """
    N=5 hierarchical system: inner binary + 3 outer bodies on nested orbits.
    Masses sum to 1.
    """
    masses = np.array([0.30, 0.30, 0.15, 0.15, 0.10])
    N = 5

    # Build inertial ICs then convert to Jacobi
    # Inner binary (bodies 0, 1): circular orbit, separation a01
    m01 = masses[0] + masses[1]
    a01 = 1.0
    v01 = np.sqrt(masses[0]*masses[1] / (m01 * a01))   # relative velocity
    # Bodies 2,3,4: placed on progressively wider orbits around growing CoM
    r = np.zeros((N, 2))
    v = np.zeros((N, 2))

    # Body 0 and 1
    r[0] = np.array([-masses[1]/m01 * a01, 0.0])
    r[1] = np.array([ masses[0]/m01 * a01, 0.0])
    v[0] = np.array([0.0, -masses[1]/m01 * np.sqrt(m01/a01)])
    v[1] = np.array([0.0,  masses[0]/m01 * np.sqrt(m01/a01)])

    # Body 2: orbit around CoM(0+1) at radius a2
    M2 = m01 + masses[2]
    a2 = 3.0
    v_circ2 = np.sqrt(m01 / a2)
    r[2] = np.array([a2, 0.0])
    v[2] = np.array([0.0, v_circ2])

    # Body 3: orbit around CoM(0+1+2) at radius a3
    M3 = M2 + masses[3]
    a3 = 6.0
    v_circ3 = np.sqrt((m01 + masses[2]) / a3)
    r[3] = np.array([-a3, 0.0])
    v[3] = np.array([0.0, -v_circ3])

    # Body 4: outermost, orbit around total CoM at radius a4
    a4 = 10.0
    v_circ4 = np.sqrt((m01 + masses[2] + masses[3]) / a4)
    r[4] = np.array([0.0, a4])
    v[4] = np.array([-v_circ4, 0.0])

    # Shift to CoM frame
    CoM = np.einsum('i,ij->j', masses, r)
    vCoM = np.einsum('i,ij->j', masses, v)
    r -= CoM;  v -= vCoM

    rho_full, drho_full = inertial_to_jacobi(masses, r, v)
    return dict(masses=masses,
                rho0=rho_full[1:], drho0=drho_full[1:],
                t_span=(0, 80), dt=0.05,
                label="5-Body Hierarchical System")


def scenario_5body_pentagon():
    """
    N=5 equal-mass bodies on a regular pentagon (near-periodic choreography attempt).
    """
    N = 5
    masses = np.full(N, 1.0/N)
    angles = 2 * np.pi * np.arange(N) / N
    R = 1.0
    r = R * np.column_stack([np.cos(angles), np.sin(angles)])

    # Circular velocity: each body orbits CoM
    # For equal masses on regular polygon, v_circ ≈ sqrt(sum of forces * R / m)
    F_radial = 0.0
    for j in range(1, N):
        dangle = 2 * np.pi * j / N
        dist = 2 * R * np.sin(np.pi * j / N)
        F_radial += np.cos(np.pi/2 - np.pi*j/N) * masses[0] / dist**2
    v_circ = np.sqrt(F_radial * R)

    v = v_circ * np.column_stack([-np.sin(angles), np.cos(angles)])

    CoM = np.einsum('i,ij->j', masses, r)
    vCoM = np.einsum('i,ij->j', masses, v)
    r -= CoM;  v -= vCoM

    rho_full, drho_full = inertial_to_jacobi(masses, r, v)
    return dict(masses=masses,
                rho0=rho_full[1:], drho0=drho_full[1:],
                t_span=(0, 30), dt=0.02,
                label="5-Body Pentagon Configuration")


# ============================================================================
# CLI
# ============================================================================

SCENARIOS = {
    'figure8':      scenario_figure8,
    'hierarchical': scenario_hierarchical_triple,
    '5body':        scenario_5body_ring,
    'pentagon':     scenario_5body_pentagon,
}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="N-Body Simulator in Jacobi Coordinates")
    parser.add_argument('--scenario', choices=list(SCENARIOS.keys()),
                        default='5body', help="Built-in scenario")
    parser.add_argument('--t_end',  type=float, default=None)
    parser.add_argument('--dt',     type=float, default=None)
    parser.add_argument('--save',   type=str,   default=None)
    args = parser.parse_args()

    cfg = SCENARIOS[args.scenario]()
    label = cfg.pop('label')
    if args.t_end: cfg['t_span'] = (cfg['t_span'][0], args.t_end)
    if args.dt:    cfg['dt'] = args.dt

    N = len(cfg['masses'])
    print(f"\n{'='*60}")
    print(f"  Scenario : {label}  (N={N})")
    print(f"  Masses   : {cfg['masses']}")
    print(f"  Sum      : {cfg['masses'].sum():.12f}")
    print(f"  t_span   : {cfg['t_span']}")
    print(f"{'='*60}\n")

    result = simulate(**cfg)

    E0, Ef = result['energy'][0], result['energy'][-1]
    L0, Lf = result['Lz'][0],    result['Lz'][-1]
    print(f"  Steps    : {len(result['t'])}")
    print(f"  ΔE/E₀    : {abs(Ef-E0)/abs(E0):.2e}")
    print(f"  ΔL/|L₀|  : {abs(Lf-L0)/(abs(L0)+1e-30):.2e}")

    plot_results(result, title=label, save_path=args.save)