# Copilot Instructions — ASEN 6062: Celestial Mechanics

## Project Overview

This is a graduate-level Celestial Mechanics coursework repository (CU Boulder, ASEN 6062). The code implements N-body gravitational simulations, orbital mechanics computations, and related analyses in Python. Reference material follows notation from Scheeres' lectures and course PDFs (especially `resources/Jacobi_Notation.pdf`).

## Environment

- **Python 3.14** via a local venv at `homework/.venv`
- Activate: `homework\.venv\Scripts\activate` (Windows)
- Core dependencies: `numpy`, `scipy` (especially `solve_ivp`), `matplotlib`
- No build system, package manager lock file, or test framework — scripts are run directly
- Jupyter notebooks (`.ipynb`) are used alongside standalone `.py` files

## Running Code

```bash
cd homework
.venv\Scripts\activate
python Comp\nbp_sim.py          # Run an N-body simulation script
python Comp\r4bp.py             # Restricted 4-body problem
jupyter notebook                # For .ipynb files (hwk2/, project/)
```

## Architecture

### Simulation Scripts (`homework/Comp/`)

- **`nbp_sim.py`** — Primary N-body simulation using Jacobi coordinates with Lagrangian formulation. Contains:
  - Jacobi coordinate transforms and their inverse (`jacobi_coordinates`, `positions_from_jacobi`)
  - Jacobi reduced masses and linear transform matrix
  - Potential gradient computation directly in Jacobi coordinates (`_potential_gradient_jacobi`)
  - Equations of motion via Lagrangian mechanics (`equations_of_motion`)
  - Energy and angular momentum conservation checks
- **`r4bp.py`** — Restricted 4-body problem simulation using direct Cartesian N-body integration
- **`nbp_sim_claude.py`** — Alternative implementation for comparison

### Project (`homework/project/`)

- Moon capture trajectory analysis with both Python (`MoonCapture.py`) and MATLAB (`Mooncapture.m`) implementations, plus Jupyter notebooks for exploration

## Key Conventions

- **Normalized units**: G = 1.0 throughout. Masses, distances, and times are in normalized/canonical units (solar masses, AU, years or similar), not SI
- **Jacobi coordinates**: The primary coordinate system for N-body work. `R[0]` is the center of mass (ignorable); `R[1..N-1]` are relative Jacobi vectors. Indexing follows Scheeres' convention but is 0-based in code
- **Integration**: `scipy.integrate.solve_ivp` with `DOP853` method, tight tolerances (`rtol=1e-10`, `atol=1e-12`)
- **Conservation validation**: Every simulation should check energy and angular momentum residuals to verify numerical accuracy
- **Softened gravity**: Close encounters use softening parameter `eps = 1e-6` to prevent divergence
- **Plotting**: `matplotlib` is used inline; scripts typically end with `plt.show()` calls for orbit plots and residual diagnostics

## Scheeres' Notation & Conventions

This course follows Daniel Scheeres' formulation. Key references live in `resources/` (especially `Jacobi_Notation.pdf`, `CentralConfigurations.pdf`, `Equilibrium Solution Properties.pdf`).

### N-Body Problem (Scheeres' formulation)

- Bodies are **1-indexed** in the theory (m₁, m₂, ..., mₙ) but **0-indexed** in Python code
- Cumulative mass: M_i = m₁ + m₂ + ... + mᵢ (partial sums)
- **Jacobi coordinates** (hierarchical relative vectors):
  - ρ₀ = R = center of mass (ignorable in CoM frame)
  - ρₖ = rₖ₊₁ − CoM(bodies 1..k), for k = 1, ..., N−1
  - In code: `jacobi_pos[0]` = CoM, `jacobi_pos[k]` = ρₖ
- **Reduced masses**: η_k = M_k · m_{k+1} / M_{k+1}
- **Lagrangian EOM**: η_k · ρ_k'' = −∂U/∂ρ_k
  - Equivalently: ρ_k'' = (1/η_k) · (−∂U/∂ρ_k)
- **Potential gradient in Jacobi coords** (`_potential_gradient_jacobi`): Uses the partial derivative chain rule from Scheeres' `Jacobi_Notation.pdf`. For body pair (a, b) with a < b, the coefficient of the force F_ab in −∂U/∂R_l is:
  - l = b: +1
  - a < l < b: m_l / M_l (cumulative mass ratio)
  - l = a: −M_{a−1} / M_a (zero when a = 0)
- **Jacobi linear transform**: L matrix (lower-triangular) maps Cartesian → Jacobi positions per axis. Same L transforms accelerations since the mapping is linear and mass-weighted.

### Central Configurations

- A configuration where all acceleration vectors point toward the common center of mass with the same proportionality constant λ: a_i = −λ · r_i (in CoM frame)
- For N = 3: only equilateral triangle (Lagrange) and collinear (Euler) solutions
- The code places 3 bodies at vertices of a unit equilateral triangle, then centers on CoM
- Circular orbits around CoM with ω² = G·M/a³ give rigid rotation

### Equilibrium & Relative Equilibria

- **Relative equilibrium**: configuration that rotates rigidly with constant angular velocity — stationary in the rotating frame
- In the rotating frame, equilibrium requires: ∇Ω = 0 (gradient of pseudo-potential vanishes)
- Stability analyzed via eigenvalues of the linearized equations (see State Transition Matrix below)

### Restricted Problems

- **CR3BP**: Two massive primaries in circular orbit, third body massless. Scheeres extends this to:
- **R4BP**: Restricted 4-body problem (`r4bp.py`) — three primaries + test particle
- Test particle placed at the system CoM with infinitesimal mass (1e-12 in code)
- The `run_problem()` driver adds a test particle automatically when `test_particle=True`

## CR3BP & Earth-Moon System Reference

This section captures domain knowledge relevant to the course project (temporary Moon capture) and CR3BP work.

### CR3BP Formulation

The Circular Restricted 3-Body Problem uses a rotating frame co-rotating with the two primaries. For the Earth-Moon system:

- **Mass parameter**: μ = m_Moon / (m_Earth + m_Moon) ≈ 0.01215
- **Nondimensionalization**: Length unit = Earth-Moon distance (~384,400 km), time unit = 1/(orbital angular rate), so one lunar orbit = 2π time units
- **Primary positions** (in the rotating frame): Earth at (-μ, 0, 0), Moon at (1-μ, 0, 0)
- **Equations of motion** (rotating frame):
  - x'' - 2y' = ∂Ω/∂x
  - y'' + 2x' = ∂Ω/∂y
  - z'' = ∂Ω/∂z
  - where Ω = (1/2)(x² + y²) + (1-μ)/r₁ + μ/r₂ is the pseudo-potential
  - r₁ = distance to Earth, r₂ = distance to Moon

### Jacobi Constant

- C_J = -2E_rotating = 2Ω - (x'² + y'² + z'²) — the only integral of motion in the CR3BP
- Defines zero-velocity curves/surfaces that bound accessible regions
- Temporary capture requires C_J near the L1 or L2 values to allow passage through the neck regions

### Lagrange Points (Earth-Moon)

| Point | Description | Approx. x (nondim) |
|-------|-------------|---------------------|
| L1 | Between Earth and Moon | ~0.8369 |
| L2 | Beyond Moon (far side) | ~1.1557 |
| L3 | Opposite Moon from Earth | ~-1.0051 |
| L4 | Leading equilateral | (0.5-μ, √3/2, 0) |
| L5 | Trailing equilateral | (0.5-μ, -√3/2, 0) |

### Temporary Capture

- A body is "temporarily captured" when it transitions from heliocentric/free orbit to orbiting the secondary (Moon) for a finite time before escaping
- Key mechanisms: passage through L1/L2 gateway, ballistic capture, weak stability boundary
- Monitor: Hill sphere radius of Moon ≈ 0.1 nondim (~61,500 km); Jacobi constant relative to L1/L2 values
- Capture classification: track number of periapsis passages around the secondary, duration inside Hill sphere

### Useful Physical Constants (SI, for converting to/from normalized units)

- Earth mass: 5.972e24 kg
- Moon mass: 7.342e22 kg
- Earth-Moon distance: 384,400 km
- Lunar orbital period: 27.322 days
- G = 6.674e-11 m³/(kg·s²)

## State Transition Matrix & Stability

The STM Φ(t, t₀) maps perturbations from initial to final states: δx(t) = Φ(t, t₀) · δx(t₀).

### Computing the STM

- The CR3BP state is x = [x, y, z, x', y', z']ᵀ (6D)
- Linearized dynamics: δx' = A(t) · δx, where A is the Jacobian of the EOM
- A for the CR3BP rotating frame (planar):

```
A = | 0    0    1   0  |
    | 0    0    0   1  |
    | Ωxx  Ωxy  0   2  |
    | Ωxy  Ωyy -2   0  |
```

where Ωxx, Ωxy, Ωyy are second partials of the pseudo-potential Ω

- Integrate Φ' = A · Φ alongside the trajectory, with Φ(t₀, t₀) = I (identity)
- In practice: augment the state vector to [x(6); Φ(36)] and integrate together
- Eigenvalues of Φ at period T determine **stability** of periodic orbits:
  - Stable if all eigenvalues on the unit circle
  - Unstable if any eigenvalue |λ| > 1; its reciprocal 1/λ is the stable eigenvalue

### Stability of Lagrange Points

| Point | Stability (Earth-Moon) |
|-------|----------------------|
| L1 | Unstable (saddle) — one unstable eigenvalue pair |
| L2 | Unstable (saddle) — one unstable eigenvalue pair |
| L3 | Unstable (saddle) |
| L4, L5 | Linearly stable for μ < μ_Routh ≈ 0.0385 (Earth-Moon: stable) |

## Invariant Manifold Computation

Invariant manifolds of unstable periodic orbits (e.g., Lyapunov, halo orbits around L1/L2) form the "tubes" that govern transport in the CR3BP. These are central to temporary capture dynamics.

### Periodic Orbit Computation (Differential Correction)

1. **Initial guess**: For Lyapunov orbits near L1/L2, use Lindstedt-Poincaré or Richardson third-order approximation
2. **Symmetry**: Planar Lyapunov orbits cross the x-axis perpendicularly — exploit the symmetry condition: at x-axis crossing, y = 0 and x' = 0
3. **Shooting**: Integrate half-period, correct initial conditions using the STM to drive y(T/2) → 0 and x'(T/2) → 0
4. **Continuation**: Once one orbit is found, vary the Jacobi constant (or amplitude) and use the previous solution as initial guess for the next

### Manifold Globalization Algorithm

Given a periodic orbit with period T and its monodromy matrix Φ(T):

1. **Compute monodromy matrix**: Integrate the STM over one full period T
2. **Extract eigenvectors**: Find the unstable eigenvector v_u (|λ_u| > 1) and stable eigenvector v_s (|λ_s| < 1) of Φ(T)
3. **Sample points on orbit**: Choose M points x_i equally spaced in time around the periodic orbit
4. **At each sample point**, map the eigenvector to that point using the STM:
   - v_u(t_i) = Φ(t_i, 0) · v_u(0), then normalize
   - v_s(t_i) = Φ(t_i, 0) · v_s(0), then normalize
5. **Perturb**: Create initial conditions x_i ± ε · v_u(t_i) for unstable manifold (ε ~ 1e-6 in nondim units)
   - (+) direction gives one branch (e.g., toward Earth), (−) gives the other (e.g., toward Moon)
   - Same with v_s for stable manifold, but integrate **backward** in time
6. **Integrate**: Propagate each perturbed IC forward (unstable) or backward (stable) in time using the full nonlinear CR3BP EOM
7. **Result**: The collection of trajectories traces out the manifold tube

### Implementation Notes

- The **unstable manifold** W^u is obtained by forward integration from perturbations along v_u
- The **stable manifold** W^s is obtained by **backward** integration from perturbations along v_s
- Perturbation size ε should be small enough to stay in the linear regime (~1e-5 to 1e-7 nondim)
- For temporary capture: the **stable manifold of an L2 Lyapunov orbit** brings trajectories toward the Moon; the **unstable manifold of L1** provides escape paths back to Earth
- Heteroclinic/homoclinic connections between L1 and L2 manifolds form the backbone of ballistic capture trajectories
- In the Earth-Moon system, the manifold tubes are relatively narrow — Poincaré sections at y = 0 (or x = 1−μ) help visualize intersections

### Poincaré Sections

- Define a surface of section (e.g., y = 0 with y' > 0 near the Moon)
- Record (x, x') each time a manifold trajectory crosses the section
- Overlapping stable (L2) and unstable (L1) manifold traces on the same section indicate possible transfer/capture orbits
- Intersection of W^s(L2) and W^u(L1) ↔ ballistic capture trajectory

## Available MCP Tools for Astrodynamics

When working in Copilot sessions, the following MCP tool sets are available and relevant:

### IO Aerospace (io-aerospace-*)

Real ephemeris and orbital mechanics backed by NASA SPICE kernels:

- **Ephemeris**: `get_ephemeris_as_state_vectors` — get position/velocity of any solar system body (EARTH, MOON, SUN, etc.) in J2000, ICRF, or body-fixed frames over a time range
- **Coordinate conversions**: Convert between state vectors, Keplerian elements, and equinoctial elements (`convert_state_vector_to_keplerian_elements`, etc.)
- **Frame transforms**: `convert_state_vector_to_the_given_frame` — transform between any supported frames (J2000, ICRF, ITRF93, IAU_MOON, etc.)
- **Constraint finding**: `find_distance_constraint`, `find_coordinate_constraint`, `find_occulting_constraint` — find time windows when geometric conditions are met (e.g., closest approach, eclipses)
- **Celestial body properties**: `get_celestial_body_properties` — mass, radii, GM for any body
- **Unit conversions**: Degrees/radians/arcseconds, km/m/AU/parsec/light-years
- **Time conversions**: `convert_date_time` between UTC, TDB, TAI, GPS

### Astronomy MCP (astronomy-mcp-*)

- **JPL Horizons**: `search_horizons_object`, `get_horizons_ephemeris` — current ephemeris for solar system objects
- **SIMBAD/NED**: Deep sky and extragalactic object lookup
- **Visibility checks**: `check_visibility` — determine if an object is observable from a location
