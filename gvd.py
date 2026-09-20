import json
import math
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shapely
from scipy.interpolate import PchipInterpolator, UnivariateSpline
from skfem import Basis, ElementTriP0
from skfem.io import from_meshio

from femwell.maxwell.waveguide import compute_modes
from femwell.mesh import mesh_from_OrderedDict


# ------------------------------------------------------------
# Search setup
# ------------------------------------------------------------
HEIGHT_NM = 220.0

COARSE_WIDTHS_NM = np.arange(350.0, 431.0, 10.0)
COARSE_WAVELENGTHS_NM = np.linspace(1500.0, 1650.0, 13)

REFINE_WIDTH_HALFSPAN_NM = 10.0
REFINE_WIDTH_STEP_NM = 2.0
REFINE_WAVELENGTH_HALFSPAN_NM = 40.0
REFINE_WAVELENGTH_POINTS = 21

FINAL_WAVELENGTH_HALFSPAN_NM = 40.0
FINAL_WAVELENGTH_POINTS = 41

MIN_POSITIVE_GVD = 1.0
POSITIVE_GVD_MAX = 100.0

NUM_MODES = 3
CACHE_DIR = Path("femwell_gvd_cache")
CACHE_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# Material dispersion
# ------------------------------------------------------------
# Si: H. H. Li, 293 K, tabulated refractive index, 1.2-14 um.
# The interpolation is only used to provide n(lambda) to FEMWELL.
SI_LAMBDA_UM = np.array([
    1.20, 1.22, 1.24, 1.26, 1.28, 1.30, 1.32, 1.34, 1.36, 1.38,
    1.40, 1.45, 1.50, 1.55, 1.60, 1.65, 1.70, 1.80, 1.90, 2.00,
    2.25, 2.50, 2.75, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00,
    10.0, 11.0, 12.0, 13.0, 14.0,
])

SI_N = np.array([
    3.5167, 3.5133, 3.5102, 3.5072, 3.5043, 3.5016, 3.4990,
    3.4965, 3.4941, 3.4918, 3.4896, 3.4845, 3.4799, 3.4757,
    3.4719, 3.4684, 3.4653, 3.4597, 3.4550, 3.4510, 3.4431,
    3.4375, 3.4334, 3.4302, 3.4229, 3.4195, 3.4177, 3.4165,
    3.4158, 3.4153, 3.4150, 3.4147, 3.4145, 3.4144, 3.4142,
])

SI_SPLINE = PchipInterpolator(SI_LAMBDA_UM, SI_N, extrapolate=False)


def n_Si(wavelength_um):
    if wavelength_um < SI_LAMBDA_UM[0] or wavelength_um > SI_LAMBDA_UM[-1]:
        raise ValueError(
            f"Si model valid only from {SI_LAMBDA_UM[0]} to "
            f"{SI_LAMBDA_UM[-1]} um"
        )
    return float(SI_SPLINE(wavelength_um))


def n_SiO2(wavelength_um):
    if wavelength_um < 0.21 or wavelength_um > 6.7:
        raise ValueError("SiO2 wavelength outside Sellmeier range")

    return math.sqrt(
        0.6961663 * wavelength_um**2
        / (wavelength_um**2 - 0.0684043**2)
        + 0.4079426 * wavelength_um**2
        / (wavelength_um**2 - 0.1162414**2)
        + 0.8974794 * wavelength_um**2
        / (wavelength_um**2 - 9.896161**2)
        + 1
    )


# ------------------------------------------------------------
# FEMWELL waveguide model
# ------------------------------------------------------------
def build_mesh(width_nm, height_nm):
    width = width_nm * 1e-3
    height = height_nm * 1e-3

    core = shapely.geometry.box(
        -width / 2,
        0,
        width / 2,
        height,
    )
    cladding = shapely.geometry.box(
        -width * 2,
        0,
        width * 2,
        height * 3,
    )
    buried_oxide = shapely.geometry.box(
        -width * 2,
        -height * 2,
        width * 2,
        0,
    )

    polygons = OrderedDict(
        core=core,
        cladding=cladding,
        buried_oxide=buried_oxide,
    )

    resolutions = {
        "core": {"resolution": 0.02, "distance": 0.3},
        "cladding": {"resolution": 0.05, "distance": 0.3},
        "buried_oxide": {"resolution": 0.05, "distance": 0.3},
    }

    return from_meshio(
        mesh_from_OrderedDict(
            polygons,
            resolutions,
            default_resolution_max=2,
        )
    )


def cache_file(width_nm, height_nm, wavelengths_nm):
    key = (
        f"w{width_nm:.1f}_h{height_nm:.1f}_"
        f"l{wavelengths_nm[0]:.1f}_{wavelengths_nm[-1]:.1f}_"
        f"n{len(wavelengths_nm)}.npz"
    )
    return CACHE_DIR / key.replace(".", "p")


def solve_neff_sweep(width_nm, height_nm, wavelengths_nm):
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)
    cache = cache_file(width_nm, height_nm, wavelengths_nm)

    if cache.exists():
        data = np.load(cache)
        return data["wavelength_nm"], data["neff"]

    mesh = build_mesh(width_nm, height_nm)
    basis0 = Basis(mesh, ElementTriP0())
    epsilon = basis0.zeros()

    neff = []

    for j, wavelength_nm in enumerate(wavelengths_nm, start=1):
        wavelength_um = wavelength_nm * 1e-3

        epsilon[basis0.get_dofs(elements="core")] = n_Si(wavelength_um) ** 2
        epsilon[basis0.get_dofs(elements="cladding")] = n_SiO2(wavelength_um) ** 2
        epsilon[basis0.get_dofs(elements="buried_oxide")] = n_SiO2(wavelength_um) ** 2

        modes = compute_modes(
            basis0,
            epsilon,
            wavelength=wavelength_um,
            num_modes=NUM_MODES,
            order=1,
        )

        modes_sorted = modes.sorted(
            key=lambda mode: -np.real(mode.te_fraction)
        )
        mode = modes_sorted[0]

        neff.append(float(np.real(mode.n_eff)))

        print(
            f"  w={width_nm:6.1f} nm, "
            f"lambda={wavelength_nm:7.2f} nm "
            f"({j:02d}/{len(wavelengths_nm):02d}) "
            f"neff={neff[-1]:.7f}"
        )

    neff = np.asarray(neff)

    np.savez(
        cache,
        wavelength_nm=wavelengths_nm,
        neff=neff,
    )

    return wavelengths_nm, neff


# ------------------------------------------------------------
# Dispersion extraction: same numerical procedure as FEMWELL tutorial
# ------------------------------------------------------------
def dispersion_from_neff(wavelength_nm, neff):
    spline = UnivariateSpline(
        wavelength_nm,
        neff,
        s=0,
        k=3,
    )

    d1 = spline.derivative(1)
    d2 = spline.derivative(2)

    dense_nm = np.linspace(
        wavelength_nm[1],
        wavelength_nm[-2],
        1201,
    )

    neff_dense = spline(dense_nm)
    ng_dense = neff_dense - dense_nm * d1(dense_nm)

    gvd_dense = (
        -dense_nm
        / 2.99792e-7
        * d2(dense_nm)
    )

    return {
        "spline": spline,
        "dense_nm": dense_nm,
        "neff": neff_dense,
        "ng": ng_dense,
        "gvd": gvd_dense,
    }


def best_positive_point(result):
    gvd = result["gvd"]
    lam = result["dense_nm"]

    mask = (
        np.isfinite(gvd)
        & (gvd >= MIN_POSITIVE_GVD)
        & (gvd <= POSITIVE_GVD_MAX)
    )

    if np.any(mask):
        idxs = np.where(mask)[0]
        idx = idxs[np.argmin(gvd[idxs])]
        return {
            "found_target_band": True,
            "index": int(idx),
            "wavelength_nm": float(lam[idx]),
            "gvd": float(gvd[idx]),
            "neff": float(result["neff"][idx]),
            "ng": float(result["ng"][idx]),
            "score": float(gvd[idx]),
        }

    idx = int(np.nanargmin(np.abs(gvd)))
    return {
        "found_target_band": False,
        "index": idx,
        "wavelength_nm": float(lam[idx]),
        "gvd": float(gvd[idx]),
        "neff": float(result["neff"][idx]),
        "ng": float(result["ng"][idx]),
        "score": float(1e6 + abs(gvd[idx])),
    }


def evaluate_width(width_nm, wavelengths_nm):
    wavelength_nm, neff = solve_neff_sweep(
        width_nm,
        HEIGHT_NM,
        wavelengths_nm,
    )
    dispersion = dispersion_from_neff(
        wavelength_nm,
        neff,
    )
    best = best_positive_point(dispersion)

    return {
        "width_nm": float(width_nm),
        "height_nm": float(HEIGHT_NM),
        "sample_wavelength_nm": wavelength_nm,
        "sample_neff": neff,
        "dispersion": dispersion,
        "best": best,
    }


# ------------------------------------------------------------
# Stage 1: coarse width search
# ------------------------------------------------------------
print("=" * 78)
print("SOI FEMWELL GVD SEARCH")
print("=" * 78)
print(f"height fixed at       = {HEIGHT_NM:.1f} nm")
print(f"coarse widths         = {COARSE_WIDTHS_NM[0]:.0f}..."
      f"{COARSE_WIDTHS_NM[-1]:.0f} nm")
print(f"coarse wavelength     = {COARSE_WAVELENGTHS_NM[0]:.0f}..."
      f"{COARSE_WAVELENGTHS_NM[-1]:.0f} nm")
print(
    f"positive GVD target  = smallest value in "
    f"[+{MIN_POSITIVE_GVD:.1f}, +{POSITIVE_GVD_MAX:.1f}] ps/(nm km)"
)
print()

coarse_results = []

for width_nm in COARSE_WIDTHS_NM:
    print(f"\nCOARSE width = {width_nm:.1f} nm")
    result = evaluate_width(
        width_nm,
        COARSE_WAVELENGTHS_NM,
    )
    coarse_results.append(result)

    b = result["best"]
    print(
        f"  candidate: lambda={b['wavelength_nm']:.2f} nm, "
        f"GVD={b['gvd']:+.3f} ps/(nm km), "
        f"ng={b['ng']:.5f}"
    )

coarse_best = min(
    coarse_results,
    key=lambda item: item["best"]["score"],
)

coarse_best_width = coarse_best["width_nm"]
coarse_best_lambda = coarse_best["best"]["wavelength_nm"]


# ------------------------------------------------------------
# Stage 2: refine width and wavelength around coarse optimum
# ------------------------------------------------------------
refine_widths_nm = np.arange(
    coarse_best_width - REFINE_WIDTH_HALFSPAN_NM,
    coarse_best_width + REFINE_WIDTH_HALFSPAN_NM + 0.1,
    REFINE_WIDTH_STEP_NM,
)

refine_lambda_min = max(
    1500.0,
    coarse_best_lambda - REFINE_WAVELENGTH_HALFSPAN_NM,
)
refine_lambda_max = min(
    1650.0,
    coarse_best_lambda + REFINE_WAVELENGTH_HALFSPAN_NM,
)

refine_wavelengths_nm = np.linspace(
    refine_lambda_min,
    refine_lambda_max,
    REFINE_WAVELENGTH_POINTS,
)

print("\n" + "=" * 78)
print("REFINEMENT")
print("=" * 78)
print(
    f"widths = {refine_widths_nm[0]:.1f}..."
    f"{refine_widths_nm[-1]:.1f} nm"
)
print(
    f"wavelength = {refine_wavelengths_nm[0]:.1f}..."
    f"{refine_wavelengths_nm[-1]:.1f} nm"
)

refined_results = []

for width_nm in refine_widths_nm:
    print(f"\nREFINE width = {width_nm:.1f} nm")
    result = evaluate_width(
        width_nm,
        refine_wavelengths_nm,
    )
    refined_results.append(result)

    b = result["best"]
    print(
        f"  candidate: lambda={b['wavelength_nm']:.2f} nm, "
        f"GVD={b['gvd']:+.3f} ps/(nm km), "
        f"ng={b['ng']:.5f}"
    )

refined_best = min(
    refined_results,
    key=lambda item: item["best"]["score"],
)

best_width = refined_best["width_nm"]
best_lambda_pre = refined_best["best"]["wavelength_nm"]


# ------------------------------------------------------------
# Stage 3: dense wavelength validation at the best geometry
# ------------------------------------------------------------
final_lambda_min = max(
    1500.0,
    best_lambda_pre - FINAL_WAVELENGTH_HALFSPAN_NM,
)
final_lambda_max = min(
    1650.0,
    best_lambda_pre + FINAL_WAVELENGTH_HALFSPAN_NM,
)

final_wavelengths_nm = np.linspace(
    final_lambda_min,
    final_lambda_max,
    FINAL_WAVELENGTH_POINTS,
)

print("\n" + "=" * 78)
print("FINAL VALIDATION")
print("=" * 78)
print(f"width  = {best_width:.1f} nm")
print(f"height = {HEIGHT_NM:.1f} nm")
print(
    f"lambda = {final_wavelengths_nm[0]:.1f}..."
    f"{final_wavelengths_nm[-1]:.1f} nm"
)

final_result = evaluate_width(
    best_width,
    final_wavelengths_nm,
)
best = final_result["best"]


# ------------------------------------------------------------
# Print final result
# ------------------------------------------------------------
print("\n" + "=" * 78)
print("BEST SLIGHTLY-POSITIVE-GVD POINT")
print("=" * 78)
print(f"width       = {best_width:.2f} nm")
print(f"height      = {HEIGHT_NM:.2f} nm")
print(f"wavelength  = {best['wavelength_nm']:.3f} nm")
print(f"n_eff       = {best['neff']:.8f}")
print(f"n_g         = {best['ng']:.8f}")
print(f"GVD         = {best['gvd']:+.4f} ps/(nm km)")
print(
    f"target band = +{MIN_POSITIVE_GVD:.1f} ... "
    f"+{POSITIVE_GVD_MAX:.1f} ps/(nm km)"
)

if not best["found_target_band"]:
    print(
        "WARNING: no positive near-zero point was found in the requested "
        "band. The printed point is only the closest-to-zero result."
    )


# ------------------------------------------------------------
# Plots
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.0, 5.8))

for result in coarse_results:
    d = result["dispersion"]
    ax.plot(
        d["dense_nm"],
        d["gvd"],
        lw=1.5,
        label=f'{result["width_nm"]:.0f} nm',
    )

ax.axhline(0.0, ls="--", lw=1.2)
ax.axhspan(
    MIN_POSITIVE_GVD,
    POSITIVE_GVD_MAX,
    alpha=0.08,
    label="slightly-positive target band",
)
ax.set_xlabel("wavelength (nm)")
ax.set_ylabel("GVD [ps/(nm km)]")
ax.set_title(f"SOI coarse GVD scan, height = {HEIGHT_NM:.0f} nm")
ax.grid(alpha=0.3)
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig("gvd_Si_coarse_width_scan.png", dpi=180)
plt.close(fig)


fig, ax = plt.subplots(figsize=(9.0, 5.8))

for result in refined_results:
    d = result["dispersion"]
    ax.plot(
        d["dense_nm"],
        d["gvd"],
        lw=1.5,
        label=f'{result["width_nm"]:.0f} nm',
    )

ax.axhline(0.0, ls="--", lw=1.2)
ax.axhspan(
    MIN_POSITIVE_GVD,
    POSITIVE_GVD_MAX,
    alpha=0.08,
)
ax.scatter(
    [best["wavelength_nm"]],
    [best["gvd"]],
    s=90,
    marker="*",
    label="selected point",
)
ax.set_xlabel("wavelength (nm)")
ax.set_ylabel("GVD [ps/(nm km)]")
ax.set_title(f"SOI refined GVD scan, height = {HEIGHT_NM:.0f} nm")
ax.grid(alpha=0.3)
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig("gvd_Si_refined_width_scan.png", dpi=180)
plt.close(fig)


d = final_result["dispersion"]
fig, axes = plt.subplots(3, 1, figsize=(9.0, 10.0), sharex=True)

axes[0].plot(d["dense_nm"], d["neff"], lw=2)
axes[0].plot(
    final_result["sample_wavelength_nm"],
    final_result["sample_neff"],
    "o",
    ms=3,
    label="FEMWELL samples",
)
axes[0].set_ylabel(r"$n_{\mathrm{eff}}$")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

axes[1].plot(d["dense_nm"], d["ng"], lw=2)
axes[1].set_ylabel(r"$n_g$")
axes[1].grid(alpha=0.3)

axes[2].plot(d["dense_nm"], d["gvd"], lw=2)
axes[2].axhline(0.0, ls="--", lw=1.2)
axes[2].axhspan(
    MIN_POSITIVE_GVD,
    POSITIVE_GVD_MAX,
    alpha=0.08,
)
axes[2].scatter(
    [best["wavelength_nm"]],
    [best["gvd"]],
    s=90,
    marker="*",
)
axes[2].set_xlabel("wavelength (nm)")
axes[2].set_ylabel("GVD [ps/(nm km)]")
axes[2].grid(alpha=0.3)

fig.suptitle(
    f"Best SOI geometry: {best_width:.0f} x {HEIGHT_NM:.0f} nm"
)
fig.tight_layout()
fig.savefig("gvd_Si_best_geometry.png", dpi=180)
plt.close(fig)


# ------------------------------------------------------------
# Save values for the ring/coupler script
# ------------------------------------------------------------
output = {
    "core_material": "Si",
    "cladding_material": "SiO2",
    "width_nm": float(best_width),
    "height_nm": float(HEIGHT_NM),
    "wavelength_nm": float(best["wavelength_nm"]),
    "neff": float(best["neff"]),
    "ng": float(best["ng"]),
    "gvd_ps_nm_km": float(best["gvd"]),
    "min_positive_gvd_ps_nm_km": float(MIN_POSITIVE_GVD),
    "positive_gvd_max_ps_nm_km": float(POSITIVE_GVD_MAX),
}

with open("best_SOI_GVD_design.json", "w") as f:
    json.dump(output, f, indent=2)

print("\nSaved:")
print("  gvd_Si_coarse_width_scan.png")
print("  gvd_Si_refined_width_scan.png")
print("  gvd_Si_best_geometry.png")
print("  best_SOI_GVD_design.json")
