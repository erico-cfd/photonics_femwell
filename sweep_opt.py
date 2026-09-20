import math
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import shapely
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from skfem import Basis, ElementTriP0
from skfem.io import from_meshio

from femwell.maxwell.waveguide import compute_modes
from femwell.mesh import mesh_from_OrderedDict


# ============================================================
# FINAL — direct sweep optimization + physical gap extraction
# ============================================================

c = 299792458.0

# Fixed device parameters
WIDTH_NM = 420.0
HEIGHT_NM = 220.0

LAMBDA0_NM = 1520.0
LAMBDA0_M = LAMBDA0_NM * 1e-9

# Approximation carried from the previous FEMWELL GVD point
NG = 4.33298099

QI = 5.0e5

# Fixed ring radii
R1_UM = 20.0
R2_UM = 10.0

L1 = 2*np.pi*R1_UM*1e-6
L2 = 2*np.pi*R2_UM*1e-6

VG = c / NG
TR1 = L1 / VG
TR2 = L2 / VG

F0 = c / LAMBDA0_M
FSR1_HZ = 1.0 / TR1

FS = F0 + FSR1_HZ
FI = F0 - FSR1_HZ

# Intrinsic loss inferred from Qi
ALPHA_POWER_M = 2*np.pi*NG / (QI*LAMBDA0_M)


# ============================================================
# Sweep ranges from the slide
# ============================================================

K1_MIN, K1_MAX = 0.005, 0.022
K2_MIN, K2_MAX = 0.000315, 0.966
K3_MIN, K3_MAX = 0.005, 0.022

N_PER_AXIS = 71
REJECTION_MIN_DB = 40.0

K1_VALUES = np.geomspace(
    K1_MIN,
    K1_MAX,
    N_PER_AXIS,
)

K2_VALUES = np.geomspace(
    K2_MIN,
    K2_MAX,
    N_PER_AXIS,
)

K3_VALUES = np.geomspace(
    K3_MIN,
    K3_MAX,
    N_PER_AXIS,
)

K1_GRID, K2_GRID, K3_GRID = np.meshgrid(
    K1_VALUES,
    K2_VALUES,
    K3_VALUES,
    indexing="ij",
)

K1_SWEEP = K1_GRID.ravel()
K2_SWEEP = K2_GRID.ravel()
K3_SWEEP = K3_GRID.ravel()


# ============================================================
# Exact coupled-ring model
# ============================================================

def solve(freq_hz, K1, K2, K3):
    """
    Exact field-amplitude solution of the two coupled rings.

    E1a: circulating field in ring 1 immediately after coupler 1.
    E2a: circulating field in ring 2 immediately after coupler 2.

    The buildup factors are
        B1 = |E1a|^2
        B2 = |E2a|^2
    for a unit-amplitude input field.
    """
    k1 = np.sqrt(K1)
    k2 = np.sqrt(K2)
    k3 = np.sqrt(K3)

    t1 = np.sqrt(1.0-K1)
    t2 = np.sqrt(1.0-K2)
    t3 = np.sqrt(1.0-K3)

    domega = 2*np.pi*(freq_hz-F0)

    phi1 = 2*np.pi + domega*TR1
    phi2 = np.pi + domega*TR2

    a1 = (
        np.exp(-ALPHA_POWER_M*L1/4.0)
        * np.exp(1j*phi1/2.0)
    )

    a2 = (
        np.exp(-ALPHA_POWER_M*L2/4.0)
        * np.exp(1j*phi2/2.0)
    )

    # Linear system for the intracavity fields E1a and E2b
    A = 1.0 - t1*t2*a1**2
    B = t1*a1*k2*a2
    C = -t3*a2*k2*a1
    D = 1.0 - t3*t2*a2**2

    det = A*D - B*C

    E1a = -k1*D/det
    E2b = C*k1/det

    # Field in ring 2 after the inter-ring coupler
    E2a = k2*a1*E1a + t2*a2*E2b

    # Drop-port field
    E_drop = k3*a2*E2a

    return {
        "E1a": E1a,
        "E2a": E2a,
        "Tdrop": np.abs(E_drop)**2,
        "B1": np.abs(E1a)**2,
        "B2": np.abs(E2a)**2,
    }


# ============================================================
# Full sweep
# ============================================================

pump_sweep = solve(
    F0,
    K1_SWEEP,
    K2_SWEEP,
    K3_SWEEP,
)

signal_sweep = solve(
    FS,
    K1_SWEEP,
    K2_SWEEP,
    K3_SWEEP,
)

idler_sweep = solve(
    FI,
    K1_SWEEP,
    K2_SWEEP,
    K3_SWEEP,
)

Tp = pump_sweep["Tdrop"]
Ts = signal_sweep["Tdrop"]
Ti = idler_sweep["Tdrop"]

B1 = pump_sweep["B1"]

rejection_db = -10*np.log10(
    np.maximum(Tp, 1e-300)
)

TsTi = Ts*Ti

valid = (
    np.isfinite(rejection_db)
    & np.isfinite(TsTi)
    & np.isfinite(B1)
)

K1_SWEEP = K1_SWEEP[valid]
K2_SWEEP = K2_SWEEP[valid]
K3_SWEEP = K3_SWEEP[valid]

rejection_db = rejection_db[valid]
TsTi = TsTi[valid]
B1 = B1[valid]

feasible = (
    rejection_db
    >= REJECTION_MIN_DB
)

if not np.any(feasible):
    raise RuntimeError(
        "No design satisfies the 40 dB rejection constraint."
    )

feasible_idx = np.flatnonzero(
    feasible
)

idx_best_TsTi = feasible_idx[
    np.argmax(
        TsTi[feasible]
    )
]

idx_best_B1 = feasible_idx[
    np.argmax(
        B1[feasible]
    )
]


# Final selected design:
# maximum Ts*Ti subject to pump rejection >= 40 dB
K1_DESIGN = float(
    K1_SWEEP[idx_best_TsTi]
)

K2_DESIGN = float(
    K2_SWEEP[idx_best_TsTi]
)

K3_DESIGN = float(
    K3_SWEEP[idx_best_TsTi]
)


# ============================================================
# Pareto frontier
# ============================================================

order = np.argsort(
    rejection_db
)[::-1]

pareto_idx = []
best_TsTi_seen = -np.inf

for idx in order:
    if TsTi[idx] > best_TsTi_seen:
        pareto_idx.append(idx)
        best_TsTi_seen = TsTi[idx]

pareto_idx = np.array(
    pareto_idx[::-1],
    dtype=int,
)


# ============================================================
# Final sweep figure
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12.5, 5.2),
)

# Rejection vs TsTi
ax = axes[0]

ax.scatter(
    rejection_db,
    TsTi,
    s=7,
    alpha=0.14,
    label="sweep designs",
)

ax.plot(
    rejection_db[pareto_idx],
    TsTi[pareto_idx],
    lw=2.0,
    label="Pareto frontier",
)

ax.axvline(
    REJECTION_MIN_DB,
    ls="--",
    lw=1.3,
    label="40 dB requirement",
)

ax.scatter(
    rejection_db[idx_best_TsTi],
    TsTi[idx_best_TsTi],
    marker="*",
    s=170,
    zorder=5,
    label=r"max $T_sT_i$, $R_p\geq40$ dB",
)

ax.set_xlabel(
    "Pump rejection (dB)"
)

ax.set_ylabel(
    r"$T_sT_i$"
)

ax.set_title(
    "(a) Rejection / transmission trade-off"
)

ax.grid(
    alpha=0.25
)

ax.legend(
    fontsize=8
)

# Rejection vs buildup
ax = axes[1]

ax.scatter(
    rejection_db,
    B1,
    s=7,
    alpha=0.14,
    label="sweep designs",
)

ax.axvline(
    REJECTION_MIN_DB,
    ls="--",
    lw=1.3,
    label="40 dB requirement",
)

ax.scatter(
    rejection_db[idx_best_B1],
    B1[idx_best_B1],
    marker="*",
    s=170,
    zorder=5,
    label=r"max $B_1$, $R_p\geq40$ dB",
)

ax.set_xlabel(
    "Pump rejection (dB)"
)

ax.set_ylabel(
    r"Pump buildup $B_1$"
)

ax.set_title(
    "(b) Rejection / buildup trade-off"
)

ax.grid(
    alpha=0.25
)

ax.legend(
    fontsize=8
)

fig.suptitle(
    r"Direct sweep optimization — fixed "
    r"$R_1=20\,\mu$m, $R_2=10\,\mu$m",
    y=0.995,
)

fig.tight_layout()

fig.savefig(
    "final_sweep_optimization.png",
    dpi=220,
)

plt.close(fig)


# ============================================================
# Intracavity buildup spectrum of the selected sweep optimum
# ============================================================
#
# B1 = |E1a|^2 : circulating-power buildup in ring 1
# B2 = |E2a|^2 : circulating-power buildup in ring 2

span_hz = 2.6*FSR1_HZ

freqs = np.linspace(
    F0-span_hz,
    F0+span_hz,
    12001,
)

wavelengths_nm = c/freqs*1e9

selected_spectrum = solve(
    freqs,
    K1_DESIGN,
    K2_DESIGN,
    K3_DESIGN,
)

B1_spectrum = selected_spectrum["B1"]
B2_spectrum = selected_spectrum["B2"]

wavelength_order = np.argsort(
    wavelengths_nm
)

x_nm = (
    wavelengths_nm[wavelength_order]
    - LAMBDA0_NM
)

fig, axes = plt.subplots(
    2,
    1,
    figsize=(10.0, 8.0),
    sharex=True,
)

axes[0].semilogy(
    x_nm,
    np.maximum(
        B1_spectrum[wavelength_order],
        1e-14,
    ),
    lw=2,
)

axes[0].set_ylabel(
    r"$B_1=|E_{1a}|^2$"
)

axes[0].set_title(
    "Ring 1 circulating-power buildup"
)

axes[0].grid(
    alpha=0.3,
    which="both",
)

axes[1].semilogy(
    x_nm,
    np.maximum(
        B2_spectrum[wavelength_order],
        1e-14,
    ),
    lw=2,
)

axes[1].set_xlabel(
    r"$\lambda-\lambda_0$ (nm)"
)

axes[1].set_ylabel(
    r"$B_2=|E_{2a}|^2$"
)

axes[1].set_title(
    "Ring 2 circulating-power buildup"
)

axes[1].grid(
    alpha=0.3,
    which="both",
)

fig.tight_layout()

fig.savefig(
    "buildup_selected_design.png",
    dpi=220,
)

plt.close(fig)


# ============================================================
# Drop-port transmission spectrum of the selected design
# ============================================================

Tdrop_spectrum = selected_spectrum["Tdrop"]

fig, ax = plt.subplots(
    figsize=(10.0, 6.0),
)

ax.semilogy(
    x_nm,
    np.maximum(
        Tdrop_spectrum[wavelength_order],
        1e-14,
    ),
    lw=2,
    label=r"$T_{drop}$",
)

ax.axhline(
    10**(-REJECTION_MIN_DB/10.0),
    ls="--",
    lw=1.3,
    label=f"{REJECTION_MIN_DB:.0f} dB requirement",
)

pump_selected = solve(
    F0,
    K1_DESIGN,
    K2_DESIGN,
    K3_DESIGN,
)

signal_selected = solve(
    FS,
    K1_DESIGN,
    K2_DESIGN,
    K3_DESIGN,
)

idler_selected = solve(
    FI,
    K1_DESIGN,
    K2_DESIGN,
    K3_DESIGN,
)

LAMBDA_S_NM = c/FS*1e9
LAMBDA_I_NM = c/FI*1e9

ax.scatter(
    [0.0],
    [pump_selected["Tdrop"]],
    marker="x",
    s=90,
    zorder=5,
    label="pump",
)

ax.scatter(
    [LAMBDA_S_NM-LAMBDA0_NM],
    [signal_selected["Tdrop"]],
    s=55,
    zorder=5,
    label="signal",
)

ax.scatter(
    [LAMBDA_I_NM-LAMBDA0_NM],
    [idler_selected["Tdrop"]],
    s=55,
    zorder=5,
    label="idler",
)

ax.set_xlabel(
    r"$\lambda-\lambda_0$ (nm)"
)

ax.set_ylabel(
    "Drop-port power transmission"
)

ax.set_title(
    "SOI double-ring drop-port spectrum"
)

ax.grid(
    alpha=0.3,
    which="both",
)

ax.legend(
    fontsize=8,
)

fig.tight_layout()

fig.savefig(
    "drop_transmission_selected_design.png",
    dpi=220,
)

plt.close(fig)


# ============================================================
# FEMWELL directional-coupler map: K -> physical gap
# ============================================================
#
# Same method as the attached final design code:
#
#   kappa_z(g) = pi/lambda * [n_even(g) - n_odd(g)]
#   K(g)       = sin^2[kappa_z(g) * Lc]
#
# The FEMWELL map is inverted for the selected sweep optimum.

COUPLER_LENGTH_UM = 5.0

INITIAL_GAPS_NM = np.array([
    300, 350, 400, 425, 450, 475,
    500, 525, 550, 575, 600, 625,
    650, 675, 700, 725, 750, 775,
    800, 850, 900,
], dtype=float)

MAX_GAP_NM = 1300.0
TAIL_STEP_NM = 50.0

TARGET_K_MIN = min(
    K1_DESIGN,
    K2_DESIGN,
    K3_DESIGN,
)


# ============================================================
# Si / SiO2 material models
# ============================================================

SI_LAMBDA_UM = np.array([
    1.20, 1.22, 1.24, 1.26, 1.28,
    1.30, 1.32, 1.34, 1.36, 1.38,
    1.40, 1.45, 1.50, 1.55, 1.60,
    1.65, 1.70, 1.80, 1.90, 2.00,
    2.25, 2.50, 2.75, 3.00, 4.00,
    5.00, 6.00, 7.00, 8.00, 9.00,
    10.0, 11.0, 12.0, 13.0, 14.0,
])

SI_N = np.array([
    3.5167, 3.5133, 3.5102, 3.5072, 3.5043,
    3.5016, 3.4990, 3.4965, 3.4941, 3.4918,
    3.4896, 3.4845, 3.4799, 3.4757, 3.4719,
    3.4684, 3.4653, 3.4597, 3.4550, 3.4510,
    3.4431, 3.4375, 3.4334, 3.4302, 3.4229,
    3.4195, 3.4177, 3.4165, 3.4158, 3.4153,
    3.4150, 3.4147, 3.4145, 3.4144, 3.4142,
])

SI_SPLINE = PchipInterpolator(
    SI_LAMBDA_UM,
    SI_N,
    extrapolate=False,
)


def n_Si(wavelength_um):
    if (
        wavelength_um < SI_LAMBDA_UM[0]
        or wavelength_um > SI_LAMBDA_UM[-1]
    ):
        raise ValueError(
            "wavelength outside Si material-model range"
        )

    return float(
        SI_SPLINE(
            wavelength_um
        )
    )


def n_SiO2(wavelength_um):
    if (
        wavelength_um < 0.21
        or wavelength_um > 6.7
    ):
        raise ValueError(
            "wavelength outside SiO2 Sellmeier range"
        )

    return math.sqrt(
        1
        + 0.6961663*wavelength_um**2
        / (
            wavelength_um**2
            - 0.0684043**2
        )
        + 0.4079426*wavelength_um**2
        / (
            wavelength_um**2
            - 0.1162414**2
        )
        + 0.8974794*wavelength_um**2
        / (
            wavelength_um**2
            - 9.896161**2
        )
    )


# ============================================================
# FEMWELL directional coupler
# ============================================================

def build_coupler_mesh(gap_nm):
    width = WIDTH_NM*1e-3
    height = HEIGHT_NM*1e-3
    gap = gap_nm*1e-3

    core1 = shapely.geometry.box(
        -gap/2-width,
        0,
        -gap/2,
        height,
    )

    core2 = shapely.geometry.box(
        gap/2,
        0,
        gap/2+width,
        height,
    )

    span = (
        2*width
        + gap
    )

    cladding = shapely.geometry.box(
        -2.5*span,
        0,
        2.5*span,
        5*height,
    )

    buried_oxide = shapely.geometry.box(
        -2.5*span,
        -5*height,
        2.5*span,
        0,
    )

    polygons = OrderedDict(
        core1=core1,
        core2=core2,
        cladding=cladding,
        buried_oxide=buried_oxide,
    )

    resolutions = {
        "core1": {
            "resolution": 0.02,
            "distance": 0.3,
        },
        "core2": {
            "resolution": 0.02,
            "distance": 0.3,
        },
        "cladding": {
            "resolution": 0.05,
            "distance": 0.3,
        },
        "buried_oxide": {
            "resolution": 0.05,
            "distance": 0.3,
        },
    }

    return from_meshio(
        mesh_from_OrderedDict(
            polygons,
            resolutions,
            default_resolution_max=2,
        )
    )


_COUPLER_CACHE = {}


def solve_coupler_gap(gap_nm):
    key = round(
        float(gap_nm),
        6,
    )

    if key in _COUPLER_CACHE:
        return _COUPLER_CACHE[key]

    mesh_c = build_coupler_mesh(
        gap_nm
    )

    basis_c = Basis(
        mesh_c,
        ElementTriP0(),
    )

    epsilon_c = basis_c.zeros()

    wavelength_um = (
        LAMBDA0_NM
        * 1e-3
    )

    materials = {
        "core1": n_Si,
        "core2": n_Si,
        "cladding": n_SiO2,
        "buried_oxide": n_SiO2,
    }

    for subdomain, n_model in materials.items():
        epsilon_c[
            basis_c.get_dofs(
                elements=subdomain
            )
        ] = n_model(
            wavelength_um
        )**2

    modes = compute_modes(
        basis_c,
        epsilon_c,
        wavelength=wavelength_um,
        num_modes=4,
        order=1,
    )

    te_modes = sorted(
        modes,
        key=lambda mode:
        -float(
            np.real(
                mode.te_fraction
            )
        ),
    )

    n_pair = sorted(
        [
            float(
                np.real(
                    te_modes[0].n_eff
                )
            ),
            float(
                np.real(
                    te_modes[1].n_eff
                )
            ),
        ],
        reverse=True,
    )

    n_even = n_pair[0]
    n_odd = n_pair[1]

    kappa_z = (
        np.pi
        / (LAMBDA0_NM*1e-3)
        * (n_even-n_odd)
    )

    if kappa_z <= 0:
        raise RuntimeError(
            f"Non-positive supermode splitting "
            f"at gap={gap_nm:.1f} nm"
        )

    K_value = np.sin(
        kappa_z
        * COUPLER_LENGTH_UM
    )**2

    result = {
        "gap_nm": float(gap_nm),
        "n_even": n_even,
        "n_odd": n_odd,
        "kappa_z": kappa_z,
        "K": K_value,
    }

    _COUPLER_CACHE[key] = result

    return result


# ============================================================
# Build and invert K(g)
# ============================================================

gap_samples = list(
    INITIAL_GAPS_NM
)

coupler_samples = []

for gap_nm in gap_samples:
    coupler_samples.append(
        solve_coupler_gap(
            gap_nm
        )
    )

while (
    coupler_samples[-1]["K"]
    > 0.5*TARGET_K_MIN
    and gap_samples[-1]
    < MAX_GAP_NM
):
    next_gap = (
        gap_samples[-1]
        + TAIL_STEP_NM
    )

    gap_samples.append(
        next_gap
    )

    coupler_samples.append(
        solve_coupler_gap(
            next_gap
        )
    )

COUPLER_GAPS_NM = np.array(
    [
        s["gap_nm"]
        for s in coupler_samples
    ]
)

COUPLER_KAPPA_Z = np.array(
    [
        s["kappa_z"]
        for s in coupler_samples
    ]
)

weak_mask = (
    COUPLER_KAPPA_Z
    * COUPLER_LENGTH_UM
    <= np.pi/2
)

weak_gaps = (
    COUPLER_GAPS_NM[
        weak_mask
    ]
)

weak_kappa = (
    COUPLER_KAPPA_Z[
        weak_mask
    ]
)

if len(weak_gaps) < 4:
    raise RuntimeError(
        "Not enough FEMWELL samples "
        "on the weak-coupling branch."
    )

keep = [0]

for i in range(
    1,
    len(weak_gaps),
):
    if (
        weak_kappa[i]
        < weak_kappa[
            keep[-1]
        ]
    ):
        keep.append(i)

weak_gaps = weak_gaps[
    keep
]

weak_kappa = weak_kappa[
    keep
]

KAPPA_GAP_SPLINE = PchipInterpolator(
    weak_gaps,
    np.log(
        weak_kappa
    ),
    extrapolate=False,
)


def kappa_z_of_gap(gap_nm):
    gap_nm = np.asarray(
        gap_nm,
        dtype=float,
    )

    if (
        np.any(
            gap_nm
            < weak_gaps[0]
        )
        or np.any(
            gap_nm
            > weak_gaps[-1]
        )
    ):
        raise ValueError(
            "gap outside FEMWELL weak branch"
        )

    value = np.exp(
        KAPPA_GAP_SPLINE(
            gap_nm
        )
    )

    return (
        float(value)
        if np.ndim(value) == 0
        else value
    )


def K_of_gap(gap_nm):
    return np.sin(
        kappa_z_of_gap(
            gap_nm
        )
        * COUPLER_LENGTH_UM
    )**2


K_WEAK_MAX = float(
    K_of_gap(
        weak_gaps[0]
    )
)

K_WEAK_MIN = float(
    K_of_gap(
        weak_gaps[-1]
    )
)


def gap_for_K(K_target):
    if (
        K_target < K_WEAK_MIN
        or K_target > K_WEAK_MAX
    ):
        raise RuntimeError(
            f"K={K_target:.8e} "
            "is outside the FEMWELL map."
        )

    return brentq(
        lambda gap_nm:
        K_of_gap(gap_nm)
        - K_target,
        float(
            weak_gaps[0]
        ),
        float(
            weak_gaps[-1]
        ),
    )


# Exact physical gaps for the selected sweep optimum
G1_FEMWELL_NM = gap_for_K(
    K1_DESIGN
)

G12_FEMWELL_NM = gap_for_K(
    K2_DESIGN
)

G3_FEMWELL_NM = gap_for_K(
    K3_DESIGN
)


def quantize_gap_10nm(gap_nm):
    return (
        10.0
        * np.round(
            float(gap_nm)
            / 10.0
        )
    )


# Fabricable mask gaps: multiples of 10 nm
G1_DESIGN_NM = quantize_gap_10nm(
    G1_FEMWELL_NM
)

G12_DESIGN_NM = quantize_gap_10nm(
    G12_FEMWELL_NM
)

G3_DESIGN_NM = quantize_gap_10nm(
    G3_FEMWELL_NM
)

K1_MASK = float(
    K_of_gap(
        G1_DESIGN_NM
    )
)

K2_MASK = float(
    K_of_gap(
        G12_DESIGN_NM
    )
)

K3_MASK = float(
    K_of_gap(
        G3_DESIGN_NM
    )
)


# ============================================================
# Save compact result
# ============================================================

with open(
    "final_sweep_result.txt",
    "w",
) as f:
    f.write(
        "DIRECT SWEEP OPTIMIZATION + FEMWELL GAPS\n"
    )

    f.write(
        f"R1_um = {R1_UM:.1f}\n"
    )

    f.write(
        f"R2_um = {R2_UM:.1f}\n"
    )

    f.write(
        f"constraint_dB = {REJECTION_MIN_DB:.1f}\n"
    )

    f.write(
        "\nSELECTED DESIGN — MAX TsTi WITH REJECTION >= 40 dB\n"
    )

    f.write(
        f"TsTi = {TsTi[idx_best_TsTi]:.10f}\n"
    )

    f.write(
        f"rejection_dB = "
        f"{rejection_db[idx_best_TsTi]:.10f}\n"
    )

    f.write(
        f"B1 = {B1[idx_best_TsTi]:.10f}\n"
    )

    f.write(
        f"K1 = {K1_DESIGN:.10e}\n"
    )

    f.write(
        f"K2 = {K2_DESIGN:.10e}\n"
    )

    f.write(
        f"K3 = {K3_DESIGN:.10e}\n"
    )

    selected_pump = solve(
        F0,
        K1_DESIGN,
        K2_DESIGN,
        K3_DESIGN,
    )

    f.write(
        "\nINTRACAVITY FIELDS AT THE PUMP\n"
    )
    f.write(
        f"E1a = {selected_pump['E1a']:.10e}\n"
    )
    f.write(
        f"E2a = {selected_pump['E2a']:.10e}\n"
    )
    f.write(
        f"B1_absE1a2 = {selected_pump['B1']:.10f}\n"
    )
    f.write(
        f"B2_absE2a2 = {selected_pump['B2']:.10f}\n"
    )

    f.write(
        "\nFEMWELL PHYSICAL GAPS\n"
    )

    f.write(
        f"g1_exact_nm = {G1_FEMWELL_NM:.6f}\n"
    )

    f.write(
        f"g12_exact_nm = {G12_FEMWELL_NM:.6f}\n"
    )

    f.write(
        f"g3_exact_nm = {G3_FEMWELL_NM:.6f}\n"
    )

    f.write(
        f"g1_mask_nm = {G1_DESIGN_NM:.0f}\n"
    )

    f.write(
        f"g12_mask_nm = {G12_DESIGN_NM:.0f}\n"
    )

    f.write(
        f"g3_mask_nm = {G3_DESIGN_NM:.0f}\n"
    )

    f.write(
        "\nCOUPLING AFTER 10 nm GAP QUANTIZATION\n"
    )

    f.write(
        f"K1_mask = {K1_MASK:.10e}\n"
    )

    f.write(
        f"K2_mask = {K2_MASK:.10e}\n"
    )

    f.write(
        f"K3_mask = {K3_MASK:.10e}\n"
    )


# ============================================================
# Clean console output
# ============================================================

print(
    "\n"
    + "="*72
)

print(
    "DIRECT SWEEP OPTIMIZATION + FEMWELL GAP EXTRACTION"
)

print(
    "="*72
)

print(
    "\nSweep"
)

print(
    f"  grid                    : "
    f"{N_PER_AXIS} x {N_PER_AXIS} x {N_PER_AXIS}"
)

print(
    f"  evaluated points        : "
    f"{len(rejection_db):,}"
)

print(
    f"  feasible points         : "
    f"{np.count_nonzero(feasible):,}"
)

print(
    f"  rejection constraint    : "
    f">= {REJECTION_MIN_DB:.1f} dB"
)

print(
    "\nSelected design — maximum Ts*Ti"
)

print(
    f"  Ts*Ti                   : "
    f"{TsTi[idx_best_TsTi]:.6f}"
)

print(
    f"  rejection               : "
    f"{rejection_db[idx_best_TsTi]:.3f} dB"
)

print(
    f"  buildup B1              : "
    f"{B1[idx_best_TsTi]:.6f}"
)

print(
    f"  K1                      : "
    f"{K1_DESIGN:.8e}"
)

print(
    f"  K2                      : "
    f"{K2_DESIGN:.8e}"
)

print(
    f"  K3                      : "
    f"{K3_DESIGN:.8e}"
)

selected_pump = solve(
    F0,
    K1_DESIGN,
    K2_DESIGN,
    K3_DESIGN,
)

print(
    "\nIntracavity fields at pump"
)

print(
    f"  E1a                     : "
    f"{selected_pump['E1a']:.6e}"
)

print(
    f"  E2a                     : "
    f"{selected_pump['E2a']:.6e}"
)

print(
    f"  B1 = |E1a|^2            : "
    f"{selected_pump['B1']:.6f}"
)

print(
    f"  B2 = |E2a|^2            : "
    f"{selected_pump['B2']:.6f}"
)

print(
    "\nPhysical gaps from FEMWELL"
)

print(
    f"  g1                      : "
    f"{G1_FEMWELL_NM:.2f} nm "
    f"-> {G1_DESIGN_NM:.0f} nm mask"
)

print(
    f"  g12                     : "
    f"{G12_FEMWELL_NM:.2f} nm "
    f"-> {G12_DESIGN_NM:.0f} nm mask"
)

print(
    f"  g3                      : "
    f"{G3_FEMWELL_NM:.2f} nm "
    f"-> {G3_DESIGN_NM:.0f} nm mask"
)

print(
    "\nCoupling after 10 nm mask quantization"
)

print(
    f"  K1(mask)                : "
    f"{K1_MASK:.8e} "
    f"(target {K1_DESIGN:.8e})"
)

print(
    f"  K2(mask)                : "
    f"{K2_MASK:.8e} "
    f"(target {K2_DESIGN:.8e})"
)

print(
    f"  K3(mask)                : "
    f"{K3_MASK:.8e} "
    f"(target {K3_DESIGN:.8e})"
)

print(
    "\nSaved"
)

print(
    "  final_sweep_optimization.png"
)

print(
    "  buildup_selected_design.png"
)

print(
    "  drop_transmission_selected_design.png"
)

print(
    "  final_sweep_result.txt"
)

print(
    "="*72
)
