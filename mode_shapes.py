"""
01_waveguide_modes.py
======================
Task 4, step 1 -- the FOUNDATION of everything else.

Goal
----
Solve the guided optical modes of a straight silicon-on-insulator (SOI) strip
waveguide and extract the numbers that every later step (single ring, two-ring
Vernier, fabrication-variability study) will consume:

    * n_eff(lambda)  -- effective index of each mode  (the eigenvalue)
    * the mode profiles                                (the eigenvectors)
    * n_g            -- group index, from the dispersion n_eff(lambda)

Run (inside the femwell venv):
    python 01_waveguide_modes.py
"""

from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")  # headless backend: save figures to file, no screen needed
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import box
from skfem import Basis, ElementTriP0
from skfem.io.meshio import from_meshio

from femwell.maxwell.waveguide import compute_modes
from femwell.mesh import mesh_from_OrderedDict

# ---------------------------------------------------------------------------
# 1. Parameters  (all lengths in micrometres, um)
# ---------------------------------------------------------------------------
WG_WIDTH = 0.50          # silicon core width  (500 nm -- the standard single-mode width)
WG_THICKNESS = 0.22      # silicon core height (220 nm -- the standard SOI device layer)
WVL = 1.55               # design wavelength (um) -- the telecom pump

N_SI = 3.48              # refractive index of silicon at 1.55 um
N_SIO2 = 1.44            # refractive index of the silica cladding at 1.55 um

SIM_HALF = 2.0           # half-size of the simulation window around the core (um)
NUM_MODES = 4            # how many modes to solve for (sorted by n_eff, high -> low)
ORDER = 2                # finite-element order (2 = quadratic, more accurate)


# ---------------------------------------------------------------------------
# 2. Build the cross-section mesh and the permittivity (index^2) map
# ---------------------------------------------------------------------------
def build_basis_and_epsilon(width, thickness):
    """Return (basis, epsilon) for a silicon strip fully clad in silica.

    We draw two rectangles: the silicon 'core' and a large 'clad' box around it.
    gmsh meshes both; each triangle is tagged 'core' or 'clad', which lets us
    assign a different refractive index to each region.
    """
    core = box(-width / 2, -thickness / 2, width / 2, thickness / 2)
    clad = box(-SIM_HALF, -SIM_HALF, SIM_HALF, SIM_HALF)

    polygons = OrderedDict(core=core, clad=clad)
    # Fine mesh inside/near the core (where the field lives), coarser far away.
    resolutions = {"core": {"resolution": 0.02, "distance": 0.5}}
    mesh = from_meshio(
        mesh_from_OrderedDict(polygons, resolutions, default_resolution_max=0.3)
    )

    # epsilon_r = n^2 defined per element (ElementTriP0 = one value per triangle)
    basis = Basis(mesh, ElementTriP0())
    epsilon = basis.zeros()
    for region, n in {"core": N_SI, "clad": N_SIO2}.items():
        epsilon[basis.get_dofs(elements=region)] = n**2
    return basis, epsilon


def solve(width, thickness, wavelength, num_modes=NUM_MODES):
    """Solve the waveguide modes; returns a femwell Modes object (sorted by n_eff)."""
    basis, epsilon = build_basis_and_epsilon(width, thickness)
    return compute_modes(
        basis, epsilon, wavelength=wavelength, num_modes=num_modes, order=ORDER
    )


def polarisation(mode):
    """Label a mode TE-like or TM-like from its TE fraction (0..1)."""
    return "TE" if mode.te_fraction > 0.5 else "TM"


# ---------------------------------------------------------------------------
# 3. Solve at the design wavelength and report
# ---------------------------------------------------------------------------
def modes_at_design_wavelength():
    print(f"\nSOI strip  {WG_WIDTH*1e3:.0f} x {WG_THICKNESS*1e3:.0f} nm   "
          f"@ lambda = {WVL} um   (n_Si={N_SI}, n_SiO2={N_SIO2})")
    print("-" * 60)
    print(f"{'mode':<6}{'n_eff':>10}{'pol.':>8}{'TE frac.':>12}")
    modes = solve(WG_WIDTH, WG_THICKNESS, WVL)

    te_count = tm_count = 0
    labelled = []
    for m in modes:
        pol = polarisation(m)
        idx = te_count if pol == "TE" else tm_count
        label = f"{pol}{idx}"
        labelled.append((label, m))
        if pol == "TE":
            te_count += 1
        else:
            tm_count += 1
        guided = "" if np.real(m.n_eff) > N_SIO2 + 0.02 else "  (~cutoff / not guided)"
        print(f"{label:<6}{np.real(m.n_eff):>10.4f}{pol:>8}{m.te_fraction:>12.3f}{guided}")

    # A_eff of the fundamental mode -- feeds the nonlinear parameter later
    m0 = modes[0]
    print("-" * 60)
    print(f"Fundamental A_eff = {m0.calculate_effective_area():.4f} um^2")
    print("Note: a 500 x 220 nm guide is single-mode for TE at 1550 nm -- any\n"
          "      TE1 sits at/below the cladding index (1.44), i.e. not guided.")
    return labelled


def plot_first_two_modes(labelled):
    """Save a figure with the dominant field component of the first two modes."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (label, m) in zip(axes, labelled[:2]):
        comp = "x" if polarisation(m) == "TE" else "y"  # dominant transverse E component
        m.plot_component("E", comp, part="real", ax=ax, colorbar=True)
        ax.set_title(f"{label}:  E{comp}   (n_eff = {np.real(m.n_eff):.3f})")
        ax.set_xlim(-0.8, 0.8)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
    fig.tight_layout()
    fig.savefig("modes_1550nm.png", dpi=150)
    print("saved -> modes_1550nm.png")


# ---------------------------------------------------------------------------
# 4. Dispersion: sweep wavelength -> n_eff(lambda) -> group index n_g
# ---------------------------------------------------------------------------
def dispersion_sweep(lam_min=1.50, lam_max=1.60, n_points=11):
    """Track the fundamental (TE0) mode vs wavelength and compute the group index.

    Group index:   n_g = n_eff - lambda * d(n_eff)/d(lambda)
    It sets the ring free spectral range later:  FSR = lambda^2 / (n_g * L).
    """
    lambdas = np.linspace(lam_min, lam_max, n_points)
    n_eff = np.empty_like(lambdas)
    for i, lam in enumerate(lambdas):
        modes = solve(WG_WIDTH, WG_THICKNESS, lam, num_modes=1)  # highest n_eff = TE0
        n_eff[i] = np.real(modes[0].n_eff)
        print(f"  lambda = {lam:.3f} um   n_eff(TE0) = {n_eff[i]:.4f}")

    dneff_dlam = np.gradient(n_eff, lambdas)   # numerical derivative
    n_g = n_eff - lambdas * dneff_dlam

    # value at the design wavelength (interpolated)
    ng_design = np.interp(WVL, lambdas, n_g)
    print(f"\nGroup index at {WVL} um:  n_g(TE0) = {ng_design:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    ax1.plot(lambdas * 1e3, n_eff, "o-")
    ax1.set_xlabel("wavelength (nm)")
    ax1.set_ylabel("n_eff (TE0)")
    ax1.set_title("Effective index vs wavelength")
    ax1.grid(alpha=0.3)

    ax2.plot(lambdas * 1e3, n_g, "s-", color="C1")
    ax2.set_xlabel("wavelength (nm)")
    ax2.set_ylabel("group index n_g")
    ax2.set_title("Group index vs wavelength")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("dispersion.png", dpi=150)
    print("saved -> dispersion.png")
    return lambdas, n_eff, n_g


if __name__ == "__main__":
    labelled = modes_at_design_wavelength()
    plot_first_two_modes(labelled)
    print("\nWavelength sweep for dispersion / group index:")
    dispersion_sweep()
    print("\nDone.")
