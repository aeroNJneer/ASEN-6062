"""
CR3BP Temporary Moon Capture Simulation
========================================
Earth-Moon system: pick initial conditions outside the Moon's Hill sphere
with Jacobi constant between CJ(L1) and CJ(L2) to explore temporary capture.

Nondimensional units:
  Length = Earth-Moon distance (384,400 km)
  Time   = 1/n where n = lunar mean motion => one orbit = 2*pi
  Mass   = M_Earth + M_Moon
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ── System Parameters ──────────────────────────────────────────────
MU = 0.012150585  # Earth-Moon mass parameter
X_EARTH = -MU
X_MOON = 1.0 - MU
R_HILL_MOON = (MU / 3.0) ** (1.0 / 3.0)  # ~0.1598


# ── CR3BP Functions ────────────────────────────────────────────────
def pseudo_potential(x, y, mu=MU):
    """Ω = (1/2)(x² + y²) + (1-μ)/r1 + μ/r2"""
    r1 = np.sqrt((x + mu) ** 2 + y ** 2)
    r2 = np.sqrt((x - 1 + mu) ** 2 + y ** 2)
    return 0.5 * (x**2 + y**2) + (1 - mu) / r1 + mu / r2


def jacobi_constant(state, mu=MU):
    """CJ = 2*Ω - v²"""
    x, y, vx, vy = state[0], state[1], state[3], state[4]
    return 2 * pseudo_potential(x, y, mu) - (vx**2 + vy**2)


def cr3bp_eom(t, state, mu=MU):
    """Planar CR3BP equations of motion in the rotating frame."""
    x, y, vx, vy = state
    r1 = np.sqrt((x + mu) ** 2 + y ** 2)
    r2 = np.sqrt((x - 1 + mu) ** 2 + y ** 2)

    ax = (
        2 * vy
        + x
        - (1 - mu) * (x + mu) / r1**3
        - mu * (x - 1 + mu) / r2**3
    )
    ay = (
        -2 * vx
        + y
        - (1 - mu) * y / r1**3
        - mu * y / r2**3
    )
    return [vx, vy, ax, ay]


def find_lagrange_L1(mu=MU):
    """Find L1 x-coordinate (between Earth and Moon on x-axis, y=0)."""
    def eq(x):
        r1 = abs(x + mu)
        r2 = abs(x - 1 + mu)
        return x - (1 - mu) * (x + mu) / r1**3 - mu * (x - 1 + mu) / r2**3
    return brentq(eq, 0.5, 1.0 - mu - 1e-6)


def find_lagrange_L2(mu=MU):
    """Find L2 x-coordinate (beyond Moon)."""
    def eq(x):
        r1 = abs(x + mu)
        r2 = abs(x - 1 + mu)
        return x - (1 - mu) * (x + mu) / r1**3 - mu * (x - 1 + mu) / r2**3
    return brentq(eq, 1.0 - mu + 1e-6, 1.5)


# ── Compute Reference Values ──────────────────────────────────────
x_L1 = find_lagrange_L1()
x_L2 = find_lagrange_L2()
CJ_L1 = 2 * pseudo_potential(x_L1, 0.0)  # v=0 at Lagrange point
CJ_L2 = 2 * pseudo_potential(x_L2, 0.0)

print("=" * 60)
print("CR3BP Earth-Moon System — Temporary Capture Setup")
print("=" * 60)
print(f"  μ           = {MU:.9f}")
print(f"  Moon at     = ({X_MOON:.6f}, 0)")
print(f"  Hill radius = {R_HILL_MOON:.4f}  ({R_HILL_MOON * 384400:.0f} km)")
print(f"  L1: x = {x_L1:.6f}   CJ(L1) = {CJ_L1:.6f}")
print(f"  L2: x = {x_L2:.6f}   CJ(L2) = {CJ_L2:.6f}")
print()

# ── Initial Conditions ─────────────────────────────────────────────
# Strategy: place particle on Earth side of L1, just outside the
# Moon's Hill sphere, with a velocity that gives CJ between L1 & L2.
#
# Position: on x-axis at x0, y0=0 — distance from Moon:
#   |x0 - (1-μ)| should be > R_HILL_MOON
#
# Velocity: vx0=0 (crossing x-axis going "up"), solve for vy0
# to hit a target CJ.
#
# CJ_target between CJ_L2 and CJ_L1 means the L1 neck is open
# (particle can pass through) but L2 may still be closed or narrow.

x0 = x_L1 - 0.02  # slightly Earth-ward of L1
y0 = 0.0
vx0 = 0.0

# Distance from Moon
dist_moon = abs(x0 - X_MOON)
print(f"  IC position: ({x0:.6f}, {y0})")
print(f"  Distance to Moon: {dist_moon:.4f}  (Hill sphere = {R_HILL_MOON:.4f})")
print(f"  Outside Hill sphere: {dist_moon > R_HILL_MOON}")

# Target CJ: midway between L1 and L2 values — L1 neck is open,
# L2 neck is narrow or just opening
CJ_target = 0.5 * (CJ_L1 + CJ_L2)
print(f"  Target CJ = {CJ_target:.6f}  (L1={CJ_L1:.6f}, L2={CJ_L2:.6f})")

# Solve for vy: CJ = 2*Omega - vx^2 - vy^2 => vy = sqrt(2*Omega - CJ - vx^2)
Omega_0 = pseudo_potential(x0, y0)
vy_sq = 2 * Omega_0 - CJ_target - vx0**2
if vy_sq < 0:
    raise ValueError("Target CJ not achievable at this position — increase x0 or lower CJ_target")
vy0 = np.sqrt(vy_sq)

state0 = [x0, y0, vx0, vy0]
CJ_actual = jacobi_constant(state0)
print(f"  vy0 = {vy0:.6f}")
print(f"  Actual CJ = {CJ_actual:.6f}")
print()

# ── Propagation ────────────────────────────────────────────────────
T_lunar = 2 * np.pi  # one lunar orbit period in nondim time
n_orbits = 8
t_span = (0, n_orbits * T_lunar)
t_eval = np.linspace(*t_span, 50000)

print(f"Integrating for {n_orbits} lunar orbit periods...")
sol = solve_ivp(
    cr3bp_eom,
    t_span,
    state0,
    method="DOP853",
    rtol=1e-12,
    atol=1e-14,
    t_eval=t_eval,
    dense_output=True,
)
print(f"  Integration status: {'success' if sol.success else 'FAILED'}")
print(f"  Steps taken: {len(sol.t)}")

x, y = sol.y[0], sol.y[1]
vx, vy = sol.y[2], sol.y[3]

# Verify Jacobi constant conservation
CJ_check = np.array([jacobi_constant(sol.y[:, k]) for k in range(len(sol.t))])
print(f"  CJ drift: {np.max(np.abs(CJ_check - CJ_actual)):.2e}")

# ── Capture Analysis ───────────────────────────────────────────────
dist_to_moon = np.sqrt((x - X_MOON) ** 2 + y**2)
inside_hill = dist_to_moon < R_HILL_MOON

# Find capture windows (contiguous segments inside Hill sphere)
transitions = np.diff(inside_hill.astype(int))
entries = np.where(transitions == 1)[0]
exits = np.where(transitions == -1)[0]

print()
print("─── Capture Analysis ───")
print(f"  Moon Hill sphere radius: {R_HILL_MOON:.4f}")
print(f"  Min distance to Moon: {np.min(dist_to_moon):.4f}")
print(f"  Fraction of time inside Hill sphere: {np.mean(inside_hill):.1%}")

if len(entries) > 0:
    print(f"  Number of Hill sphere entries: {len(entries)}")
    for i, ent in enumerate(entries[:10]):
        # find corresponding exit
        later_exits = exits[exits > ent]
        if len(later_exits) > 0:
            ext = later_exits[0]
            duration = sol.t[ext] - sol.t[ent]
            print(f"    Entry {i+1}: t={sol.t[ent]:.2f} → t={sol.t[ext]:.2f}  "
                  f"(duration = {duration:.2f} = {duration / T_lunar:.2f} lunar orbits)")
        else:
            print(f"    Entry {i+1}: t={sol.t[ent]:.2f} → still inside at end")
else:
    print("  No Hill sphere entries detected.")

# Count periapsis passages around Moon (local minima of distance)
from scipy.signal import argrelmin
periapsis_idx = argrelmin(dist_to_moon, order=50)[0]
peri_inside = [i for i in periapsis_idx if inside_hill[i]]
print(f"  Periapsis passages (total): {len(periapsis_idx)}")
print(f"  Periapsis passages inside Hill sphere: {len(peri_inside)}")
if len(peri_inside) > 0:
    for i, idx in enumerate(peri_inside[:8]):
        print(f"    Periapsis {i+1}: t={sol.t[idx]:.2f}, "
              f"dist={dist_to_moon[idx]:.4f}, "
              f"(x,y)=({x[idx]:.4f},{y[idx]:.4f})")

# ── Plotting ───────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle(
    f"CR3BP Earth-Moon Temporary Capture\n"
    f"CJ = {CJ_actual:.4f}  (L1: {CJ_L1:.4f}, L2: {CJ_L2:.4f})",
    fontsize=14,
)

# 1) Full trajectory in rotating frame
ax = axes[0, 0]
ax.plot(x, y, "b-", linewidth=0.3, alpha=0.6, label="Trajectory")
ax.plot(X_EARTH, 0, "co", markersize=10, label="Earth")
ax.plot(X_MOON, 0, "ko", markersize=6, label="Moon")
ax.plot(x_L1, 0, "r^", markersize=8, label="L1")
ax.plot(x_L2, 0, "rv", markersize=8, label="L2")
ax.plot(x[0], y[0], "g*", markersize=12, label="Start")
hill = Circle((X_MOON, 0), R_HILL_MOON, fill=False, color="gray", linestyle="--", label="Hill sphere")
ax.add_patch(hill)
ax.set_xlabel("x (nondim)")
ax.set_ylabel("y (nondim)")
ax.set_title("Full Trajectory (rotating frame)")
ax.legend(fontsize=7, loc="upper left")
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)

# 2) Zoom on Moon neighborhood
ax = axes[0, 1]
ax.plot(x, y, "b-", linewidth=0.5, alpha=0.7)
# Color segments inside Hill sphere red
for ent in entries:
    later_exits = exits[exits > ent]
    if len(later_exits) > 0:
        ext = later_exits[0]
    else:
        ext = len(sol.t) - 1
    ax.plot(x[ent:ext], y[ent:ext], "r-", linewidth=1.0, alpha=0.8)
ax.plot(X_MOON, 0, "ko", markersize=8)
hill2 = Circle((X_MOON, 0), R_HILL_MOON, fill=False, color="gray", linestyle="--")
ax.add_patch(hill2)
ax.plot(x_L1, 0, "r^", markersize=8)
ax.plot(x_L2, 0, "rv", markersize=8)
ax.set_xlim(X_MOON - 0.25, X_MOON + 0.25)
ax.set_ylim(-0.25, 0.25)
ax.set_xlabel("x (nondim)")
ax.set_ylabel("y (nondim)")
ax.set_title("Zoom: Moon Neighborhood (red = inside Hill sphere)")
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)

# 3) Distance to Moon vs time
ax = axes[1, 0]
ax.plot(sol.t / T_lunar, dist_to_moon, "b-", linewidth=0.5)
ax.axhline(R_HILL_MOON, color="gray", linestyle="--", label=f"Hill radius = {R_HILL_MOON:.3f}")
ax.fill_between(
    sol.t / T_lunar, 0, dist_to_moon,
    where=inside_hill, alpha=0.3, color="red", label="Inside Hill sphere"
)
ax.set_xlabel("Time (lunar orbits)")
ax.set_ylabel("Distance to Moon (nondim)")
ax.set_title("Distance to Moon vs Time")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 4) Zero-velocity curves
ax = axes[1, 1]
xgrid = np.linspace(-1.5, 1.5, 500)
ygrid = np.linspace(-1.5, 1.5, 500)
X, Y = np.meshgrid(xgrid, ygrid)
Omega_grid = pseudo_potential(X, Y)
CJ_grid = 2 * Omega_grid
ax.contour(X, Y, CJ_grid, levels=[CJ_actual], colors="red", linewidths=1.5)
ax.contourf(X, Y, CJ_grid, levels=[CJ_actual, 100], colors=["gray"], alpha=0.3)
ax.plot(x, y, "b-", linewidth=0.3, alpha=0.5)
ax.plot(X_EARTH, 0, "co", markersize=10)
ax.plot(X_MOON, 0, "ko", markersize=6)
ax.plot(x_L1, 0, "r^", markersize=8)
ax.plot(x_L2, 0, "rv", markersize=8)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel("x (nondim)")
ax.set_ylabel("y (nondim)")
ax.set_title(f"Zero-Velocity Curve at CJ = {CJ_actual:.4f}")
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figures/cr3bp_capture.png", dpi=150, bbox_inches="tight")
plt.show()

print()
print("Figure saved to figures/cr3bp_capture.png")
