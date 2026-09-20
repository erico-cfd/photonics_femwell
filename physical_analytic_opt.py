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
# FINAL DESIGN — coupled-microring photon-pair source
# ============================================================

c = 299792458.0

# General characteristics
WIDTH_NM = 420.0
HEIGHT_NM = 220.0
LAMBDA0_NM = 1520.0
LAMBDA0_M = LAMBDA0_NM * 1e-9
NG = 4.33298099  # approximation carried from the 418 nm FEMWELL point

# Intrinsic-Q design assumption
QI = 5.0e5

# Ring geometry
R1_UM = 20.0
R2_UM = 10.0

L1 = 2*np.pi*R1_UM*1e-6
L2 = 2*np.pi*R2_UM*1e-6

# Normalized inter-ring coupling choice
MU_OVER_GAMMA_I = 8.87


# ============================================================
# Round-trip times and intrinsic rate
# ============================================================

F0 = c / LAMBDA0_M
OMEGA0 = 2*np.pi*F0
VG = c / NG

TR1 = L1 / VG
TR2 = L2 / VG

FSR1_HZ = 1.0 / TR1
FSR2_HZ = 1.0 / TR2

gamma_i = OMEGA0 / (2*QI)


# ============================================================
# Coupler 1 — critical coupling at the pump
# gamma1 = gamma_i
# K1 = 1 - exp(-2*gamma_i*TR1)
# ============================================================

K1 = 1.0 - np.exp(-2.0*gamma_i*TR1)
kappa1 = np.sqrt(K1)

gamma1 = -np.log(1.0-K1) / (2.0*TR1)
Gamma1 = gamma_i + gamma1


# ============================================================
# Coupler 2 — normalized inter-ring coupling
# mu/gamma_i = 8.87
# mu = kappa2 / sqrt(TR1*TR2)
# ============================================================

mu = MU_OVER_GAMMA_I * gamma_i

kappa2 = mu * np.sqrt(TR1*TR2)
K2 = kappa2**2


# ============================================================
# Coupler 3 — extraction optimum
# gamma2* = sqrt[gamma_i * (gamma_i + mu^2/Gamma1)]
# K3 = 1 - exp(-2*gamma2*TR2)
# ============================================================

gamma2_star = np.sqrt(
    gamma_i * (
        gamma_i + mu**2/Gamma1
    )
)

K3 = 1.0 - np.exp(-2.0*gamma2_star*TR2)
kappa3 = np.sqrt(K3)

gamma2 = -np.log(1.0-K3) / (2.0*TR2)
Gamma2 = gamma_i + gamma2


# ============================================================
# Source-aware metrics
# ============================================================

eta_esc = (
    mu**2 / (Gamma1*Gamma2 + mu**2)
    * gamma2/Gamma2
)

eta_pair = eta_esc**2

Q_loaded = OMEGA0 / (2.0*Gamma1)

splitting_hz = 2.0*mu / (2*np.pi)


# ============================================================
# Ideal Vernier phase model
# ============================================================
# Pump:
#   ring1 resonant
#   ring2 anti-resonant
#
# dphi/domega = TR

# Intrinsic power-loss coefficient inferred from Qi
ALPHA_POWER_M = 2*np.pi*NG / (QI*LAMBDA0_M)


def series_rings_frequency(
    freq_hz,
    K1_value=K1,
    K2_value=K2,
    K3_value=K3,
):
    k1 = np.sqrt(K1_value)
    k2 = np.sqrt(K2_value)
    k3 = np.sqrt(K3_value)

    t1 = np.sqrt(1.0-K1_value)
    t2 = np.sqrt(1.0-K2_value)
    t3 = np.sqrt(1.0-K3_value)

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

    M = np.array([
        [1-t1*t2*a1**2,       t1*a1*k2*a2],
        [-t3*a2*k2*a1,        1-t3*t2*a2**2],
    ], dtype=complex)

    E1a, E2b = np.linalg.solve(
        M,
        [-k1, 0.0],
    )

    E1b = t2*a1*E1a - k2*a2*E2b
    E2a = k2*a1*E1a + t2*a2*E2b

    E_through = t1 + k1*a1*E1b
    E_drop = k3*a2*E2a

    return {
        "Tthrough": abs(E_through)**2,
        "Tdrop": abs(E_drop)**2,
        "B1": abs(E1a)**2,
        "B2": abs(E2a)**2,
    }


# ============================================================
# Pump / signal / idler
# ============================================================

FS = F0 + FSR1_HZ
FI = F0 - FSR1_HZ

LAMBDA_S_NM = c/FS * 1e9
LAMBDA_I_NM = c/FI * 1e9

pump = series_rings_frequency(F0)
signal = series_rings_frequency(FS)
idler = series_rings_frequency(FI)

pump_rejection_dB = -10*np.log10(
    max(pump["Tdrop"], 1e-300)
)

through_null_dB = 10*np.log10(
    max(pump["Tthrough"], 1e-300)
)

Ts = signal["Tdrop"]
Ti = idler["Tdrop"]


# ============================================================
# Console summary
# ============================================================

def print_design_summary():
    print("\n" + "="*72)
    print("COUPLED-MICRORING PHOTON-PAIR SOURCE — FINAL DESIGN")
    print("="*72)

    print("\n[1] Device")
    print(f"  SOI waveguide          : {WIDTH_NM:.0f} x {HEIGHT_NM:.0f} nm")
    print(f"  pump wavelength        : {LAMBDA0_NM:.1f} nm")
    print(f"  group index            : {NG:.8f}")
    print(f"  intrinsic Q            : {QI:.2e}")
    print(f"  ring radii             : R1={R1_UM:.1f} um, R2={R2_UM:.1f} um")

    print("\n[2] Couplings")
    print(f"  K1 = kappa1^2          : {K1:.8f}   (kappa1={kappa1:.6f})")
    print(f"  K2 = kappa2^2          : {K2:.8e}   (kappa2={kappa2:.6f})")
    print(f"  K3 = kappa3^2          : {K3:.8f}   (kappa3={kappa3:.6f})")
    print(f"  mu/gamma_i             : {mu/gamma_i:.3f}")

    print("\n[3] Performance")
    print(f"  pump buildup B1        : {pump['B1']:.2f} x")
    print(f"  pump buildup B2        : {pump['B2']:.5f} x")
    print(f"  drop pump rejection    : {pump_rejection_dB:.2f} dB")
    print(f"  escape efficiency      : {eta_esc:.4f} per photon")
    print(f"  pair extraction        : {eta_pair:.4f}")
    print(f"  loaded Q               : {Q_loaded:.3e}")
    print(f"  mode splitting         : {splitting_hz*1e-9:.3f} GHz")
    print(f"  signal wavelength      : {LAMBDA_S_NM:.3f} nm")
    print(f"  idler wavelength       : {LAMBDA_I_NM:.3f} nm")
    print(f"  Ts*Ti                  : {Ts*Ti:.4f}")

    print("\n[4] Physical coupler gaps (10 nm mask grid)")
    print(f"  g1                     : {G1_FEMWELL_NM:.1f} nm exact -> {G1_DESIGN_NM:.0f} nm mask")
    print(f"  g12                    : {G12_FEMWELL_NM:.1f} nm exact -> {G12_DESIGN_NM:.0f} nm mask")
    print(f"  g3                     : {G3_FEMWELL_NM:.1f} nm exact -> {G3_DESIGN_NM:.0f} nm mask")

    print("\n[5] Coupling after mask quantization")
    print(f"  K1(mask)               : {K_of_gap(G1_DESIGN_NM):.8f}   target={K1:.8f}")
    print(f"  K2(mask)               : {K_of_gap(G12_DESIGN_NM):.8e}   target={K2:.8e}")
    print(f"  K3(mask)               : {K_of_gap(G3_DESIGN_NM):.8f}   target={K3:.8f}")

# ============================================================
# Plot 1 — drop-port spectrum
# ============================================================

span_hz = 2.6*FSR1_HZ

freqs = np.linspace(
    F0-span_hz,
    F0+span_hz,
    12001,
)

wavelengths_nm = c/freqs*1e9

spec = [
    series_rings_frequency(f)
    for f in freqs
]

Tdrop = np.array([
    s["Tdrop"] for s in spec
])

B1_spec = np.array([
    s["B1"] for s in spec
])

B2_spec = np.array([
    s["B2"] for s in spec
])

order = np.argsort(wavelengths_nm)
x_nm = wavelengths_nm[order] - LAMBDA0_NM

fig, ax = plt.subplots(figsize=(10.0, 6.0))

ax.semilogy(
    x_nm,
    np.maximum(Tdrop[order], 1e-14),
    lw=2,
    label=r"$T_{drop}$",
)

ax.axhline(
    1e-3,
    ls="--",
    lw=1.2,
    label="30 dB requirement",
)

ax.scatter(
    [0.0],
    [pump["Tdrop"]],
    marker="x",
    s=90,
    label="pump",
)

ax.scatter(
    [LAMBDA_S_NM-LAMBDA0_NM],
    [Ts],
    s=55,
    label="signal",
)

ax.scatter(
    [LAMBDA_I_NM-LAMBDA0_NM],
    [Ti],
    s=55,
    label="idler",
)

ax.set_xlabel(r"$\lambda-\lambda_0$ (nm)")
ax.set_ylabel("Power transmission")
ax.set_title(
    "SOI double-ring drop-port spectrum"
)

ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(
    "spectrum.png",
    dpi=180,
)
plt.close(fig)


# ============================================================
# Plot 2 — buildup
# ============================================================

fig, axes = plt.subplots(
    2,
    1,
    figsize=(10.0, 8.0),
    sharex=True,
)

axes[0].semilogy(
    x_nm,
    np.maximum(B1_spec[order], 1e-14),
    lw=2,
)

axes[0].set_ylabel(r"$B_1$")
axes[0].set_title(
    "Ring 1 circulating-power buildup"
)
axes[0].grid(alpha=0.3, which="both")

axes[1].semilogy(
    x_nm,
    np.maximum(B2_spec[order], 1e-14),
    lw=2,
)

axes[1].set_xlabel(
    r"$\lambda-\lambda_0$ (nm)"
)
axes[1].set_ylabel(r"$B_2$")
axes[1].set_title(
    "Ring 2 circulating-power buildup"
)
axes[1].grid(alpha=0.3, which="both")

fig.tight_layout()
fig.savefig(
    "buildup.png",
    dpi=180,
)
plt.close(fig)


# ============================================================
# Plot 3 — dependence of K1, K2 and K3 on the three constraints
# ============================================================

SWEEP_FACTOR = np.geomspace(
    0.1,
    10.0,
    241,
)

nominal = {
    "K1": K1,
    "K2": K2,
    "K3": K3,
}


def source_metrics(
    K1_value,
    K2_value,
    K3_value,
):
    gamma1_value = (
        -np.log(1-K1_value)
        / (2*TR1)
    )

    gamma2_value = (
        -np.log(1-K3_value)
        / (2*TR2)
    )

    mu_value = (
        np.sqrt(K2_value)
        / np.sqrt(TR1*TR2)
    )

    Gamma1_value = (
        gamma_i + gamma1_value
    )

    Gamma2_value = (
        gamma_i + gamma2_value
    )

    eta_value = (
        mu_value**2
        / (
            Gamma1_value*Gamma2_value
            + mu_value**2
        )
        * gamma2_value/Gamma2_value
    )

    return eta_value


cross = {}

for name, nominal_value in nominal.items():
    values = nominal_value*SWEEP_FACTOR

    B1_values = []
    eta_values = []
    rejection_values = []

    for value in values:
        k1v = K1
        k2v = K2
        k3v = K3

        if name == "K1":
            k1v = value
        elif name == "K2":
            k2v = value
        else:
            k3v = value

        pump_value = series_rings_frequency(
            F0,
            k1v,
            k2v,
            k3v,
        )

        eta_value = source_metrics(
            k1v,
            k2v,
            k3v,
        )

        B1_values.append(
            pump_value["B1"]
        )

        eta_values.append(
            eta_value
        )

        rejection_values.append(
            -10*np.log10(
                max(
                    pump_value["Tdrop"],
                    1e-300,
                )
            )
        )

    cross[name] = {
        "factor": SWEEP_FACTOR,
        "B1": np.asarray(B1_values),
        "eta": np.asarray(eta_values),
        "rej": np.asarray(rejection_values),
    }


fig, axes = plt.subplots(
    3,
    3,
    figsize=(13.0, 10.0),
    sharex="col",
)

for col, name in enumerate(
    ("K1", "K2", "K3")
):
    d = cross[name]

    axes[0, col].semilogx(
        d["factor"],
        d["B1"],
        lw=2,
    )

    axes[1, col].semilogx(
        d["factor"],
        d["eta"],
        lw=2,
    )

    axes[2, col].semilogx(
        d["factor"],
        d["rej"],
        lw=2,
    )

    for row in range(3):
        axes[row, col].axvline(
            1.0,
            ls=":",
            lw=1.5,
        )

        axes[row, col].grid(
            alpha=0.3,
            which="both",
        )

    axes[0, col].set_title(
        rf"vary ${name}$"
    )

    axes[2, col].axhline(
        30.0,
        ls="--",
        lw=1.3,
    )

    axes[2, col].set_xlabel(
        rf"${name}/{name}_{{opt}}$"
    )

axes[0, 0].set_ylabel(
    r"pump buildup $B_1$"
)

axes[1, 0].set_ylabel(
    r"$\eta_{esc}$"
)

axes[2, 0].set_ylabel(
    "pump rejection (dB)"
)

fig.suptitle(
    "Design: coupling dependence on source constraints",
    y=0.995,
)

fig.tight_layout()
fig.savefig(
    "cross_dependence.png",
    dpi=180,
)
plt.close(fig)



# ============================================================
# Plot 4 — design metrics
# ============================================================
# (a) K1: critical coupling -> maximize pump buildup / through null
# (b) K3: extraction optimum -> maximize eta_esc
# (c) K2: free lever -> extraction / pump-rejection trade-off

fig, axes = plt.subplots(
    3,
    1,
    figsize=(9.5, 12.0),
)

# ------------------------------------------------------------
# (a) Coupler 1
# ------------------------------------------------------------
# Critical coupling is shown only through the intracavity pump
# buildup. The through-port curve is intentionally omitted here.
K1_sweep = np.geomspace(
    max(1e-5, 0.15*K1),
    min(0.95, 5.0*K1),
    401,
)

B1_K1 = np.empty_like(K1_sweep)

for j, K1_value in enumerate(K1_sweep):
    p = series_rings_frequency(
        F0,
        K1_value,
        K2,
        K3,
    )

    B1_K1[j] = p["B1"]

ax = axes[0]

ax.plot(
    100*K1_sweep,
    B1_K1,
    lw=2,
    label=r"$B_1$",
)

ax.axvline(
    100*K1,
    ls=":",
    lw=1.5,
    label="selected $K_1$",
)

ax.scatter(
    [100*K1],
    [pump["B1"]],
    s=55,
    zorder=5,
    label="design point",
)

ax.set_xscale("log")
ax.set_xlabel(r"$K_1=\kappa_1^2$ (%)")
ax.set_ylabel(r"pump buildup $B_1$")
ax.set_title(
    "(a) Coupler 1 — critical coupling maximizes pump buildup"
)
ax.grid(alpha=0.3, which="both")
ax.legend(
    fontsize=8,
    loc="best",
)


# ------------------------------------------------------------
# (b) Coupler 3
# ------------------------------------------------------------
K3_sweep = np.geomspace(
    max(1e-5, 0.15*K3),
    min(0.95, 5.0*K3),
    401,
)

eta_K3 = np.empty_like(K3_sweep)
eta_pair_K3 = np.empty_like(K3_sweep)

for j, K3_value in enumerate(K3_sweep):
    gamma2_value = (
        -np.log(1-K3_value)
        / (2*TR2)
    )

    Gamma2_value = gamma_i + gamma2_value

    eta_value = (
        mu**2
        / (
            Gamma1*Gamma2_value
            + mu**2
        )
        * gamma2_value/Gamma2_value
    )

    eta_K3[j] = eta_value
    eta_pair_K3[j] = eta_value**2

ax = axes[1]

ax.plot(
    100*K3_sweep,
    eta_K3,
    lw=2,
    label=r"$\eta_{esc}$ per photon",
)

ax.plot(
    100*K3_sweep,
    eta_pair_K3,
    lw=1.8,
    label=r"$\eta_{esc}^2$ per pair",
)

ax.axvline(
    100*K3,
    ls=":",
    lw=1.5,
    label="selected $K_3$",
)

ax.scatter(
    [100*K3],
    [eta_esc],
    s=55,
    zorder=5,
)

ax.set_xscale("log")
ax.set_xlabel(r"$K_3=\kappa_3^2$ (%)")
ax.set_ylabel("escape efficiency")
ax.set_title(
    "(b) Coupler 3 — maximum extraction"
)
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8, loc="best")


# ------------------------------------------------------------
# (c) Coupler 2
# ------------------------------------------------------------
mu_ratio_sweep = np.linspace(
    3.0,
    18.0,
    401,
)

eta_mu = np.empty_like(mu_ratio_sweep)
eta_pair_mu = np.empty_like(mu_ratio_sweep)
rej_mu = np.empty_like(mu_ratio_sweep)

for j, ratio in enumerate(mu_ratio_sweep):
    mu_value = ratio*gamma_i

    K2_value = (
        mu_value**2
        * TR1*TR2
    )

    gamma2_opt = np.sqrt(
        gamma_i * (
            gamma_i
            + mu_value**2/Gamma1
        )
    )

    K3_opt = (
        1.0
        - np.exp(
            -2.0*gamma2_opt*TR2
        )
    )

    gamma2_value = (
        -np.log(1-K3_opt)
        / (2*TR2)
    )

    Gamma2_value = (
        gamma_i + gamma2_value
    )

    eta_value = (
        mu_value**2
        / (
            Gamma1*Gamma2_value
            + mu_value**2
        )
        * gamma2_value/Gamma2_value
    )

    p = series_rings_frequency(
        F0,
        K1,
        K2_value,
        K3_opt,
    )

    eta_mu[j] = eta_value
    eta_pair_mu[j] = eta_value**2

    rej_mu[j] = (
        -10*np.log10(
            max(
                p["Tdrop"],
                1e-300,
            )
        )
    )

ax = axes[2]
ax_r = ax.twinx()

p1 = ax.plot(
    mu_ratio_sweep,
    eta_mu,
    lw=2,
    label=r"$\eta_{esc}$ per photon",
)

p2 = ax.plot(
    mu_ratio_sweep,
    eta_pair_mu,
    lw=1.8,
    label=r"$\eta_{esc}^2$ per pair",
)

p3 = ax_r.plot(
    mu_ratio_sweep,
    rej_mu,
    lw=1.8,
    ls="--",
    label="ring-stack pump rejection",
)

ax.axvspan(
    9.0,
    15.0,
    alpha=0.10,
    label=r"PDF useful window $9$--$15$",
)

ax.axvline(
    MU_OVER_GAMMA_I,
    ls=":",
    lw=1.5,
    label="selected point",
)

ax_r.axhline(
    30.0,
    ls=":",
    lw=1.2,
    label="30 dB requirement",
)

ax.set_xlabel(r"$\mu/\gamma_i$")
ax.set_ylabel("escape efficiency")
ax_r.set_ylabel("pump rejection at drop (dB)")
ax.set_title(
    "(c) Coupler 2 — extraction / pump-rejection trade-off"
)
ax.grid(alpha=0.3)

lines = (
    p1
    + p2
    + p3
    + [ax.lines[-1]]
    + [ax_r.lines[-1]]
)

ax.legend(
    lines,
    [line.get_label() for line in lines],
    fontsize=8,
    loc="best",
)

fig.suptitle(
    "Coupling-design metrics",
    y=0.995,
)

fig.tight_layout()

fig.savefig(
    "metrics.png",
    dpi=180,
)

plt.close(fig)


# ============================================================
# FEMWELL directional-coupler map: K -> physical gap
# ============================================================
# This section removes the previous assumed gaps.  It solves the
# even/odd TE-like supermodes of two identical Si waveguides and uses
#
#   kappa_z(g) = pi/lambda * [n_even(g) - n_odd(g)]
#   K(g)       = sin^2[kappa_z(g) * Lc]
#
# The resulting map is inverted to obtain g1, g12 and g3 for the
# current K1, K2 and K3.

COUPLER_LENGTH_UM = 5.0

# Coarse FEMWELL samples.  The map is extended automatically if the
# weakest target coupling is not yet reached.
INITIAL_GAPS_NM = np.array([
    300, 350, 400, 425, 450, 475, 500, 525, 550, 575,
    600, 625, 650, 675, 700, 725, 750, 775, 800, 850, 900
], dtype=float)

MAX_GAP_NM = 1300.0
TAIL_STEP_NM = 50.0

TARGET_K_MIN = min(K1, K2, K3)
TARGET_K_MAX = max(K1, K2, K3)

# Same dispersive Si model used in the standalone SOI GVD search.
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
        raise ValueError("wavelength outside Si material-model range")
    return float(SI_SPLINE(wavelength_um))


def n_SiO2(wavelength_um):
    if wavelength_um < 0.21 or wavelength_um > 6.7:
        raise ValueError("wavelength outside SiO2 Sellmeier range")

    return math.sqrt(
        1
        + 0.6961663*wavelength_um**2
        / (wavelength_um**2 - 0.0684043**2)
        + 0.4079426*wavelength_um**2
        / (wavelength_um**2 - 0.1162414**2)
        + 0.8974794*wavelength_um**2
        / (wavelength_um**2 - 9.896161**2)
    )


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

    span = 2*width + gap

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
        "core1": {"resolution": 0.02, "distance": 0.3},
        "core2": {"resolution": 0.02, "distance": 0.3},
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


_COUPLER_CACHE = {}


def solve_coupler_gap(gap_nm):
    key = round(float(gap_nm), 6)

    if key in _COUPLER_CACHE:
        return _COUPLER_CACHE[key]

    mesh_c = build_coupler_mesh(gap_nm)
    basis_c = Basis(mesh_c, ElementTriP0())
    epsilon_c = basis_c.zeros()

    wavelength_um = LAMBDA0_NM*1e-3

    materials = {
        "core1": n_Si,
        "core2": n_Si,
        "cladding": n_SiO2,
        "buried_oxide": n_SiO2,
    }

    for subdomain, n_model in materials.items():
        epsilon_c[basis_c.get_dofs(elements=subdomain)] = (
            n_model(wavelength_um)**2
        )

    modes = compute_modes(
        basis_c,
        epsilon_c,
        wavelength=wavelength_um,
        num_modes=4,
        order=1,
    )

    te_modes = sorted(
        modes,
        key=lambda mode: -float(np.real(mode.te_fraction)),
    )

    n_pair = sorted(
        [
            float(np.real(te_modes[0].n_eff)),
            float(np.real(te_modes[1].n_eff)),
        ],
        reverse=True,
    )

    n_even = n_pair[0]
    n_odd = n_pair[1]

    kappa_z = (
        np.pi/(LAMBDA0_NM*1e-3)
        * (n_even-n_odd)
    )

    if kappa_z <= 0:
        raise RuntimeError(
            f"Non-positive supermode splitting at gap={gap_nm:.1f} nm"
        )

    K_value = np.sin(
        kappa_z*COUPLER_LENGTH_UM
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


gap_samples = list(INITIAL_GAPS_NM)

coupler_samples = []

for gap_nm in gap_samples:
    result = solve_coupler_gap(gap_nm)
    coupler_samples.append(result)

# Extend only the weak-coupling tail if K2 is not yet covered.
while (
    coupler_samples[-1]["K"] > 0.5*TARGET_K_MIN
    and gap_samples[-1] < MAX_GAP_NM
):
    next_gap = gap_samples[-1] + TAIL_STEP_NM
    gap_samples.append(next_gap)

    result = solve_coupler_gap(next_gap)
    coupler_samples.append(result)

COUPLER_GAPS_NM = np.array(
    [s["gap_nm"] for s in coupler_samples]
)

COUPLER_KAPPA_Z = np.array(
    [s["kappa_z"] for s in coupler_samples]
)

COUPLER_K = np.array(
    [s["K"] for s in coupler_samples]
)

# Keep only the weak branch where K decreases monotonically with gap.
weak_mask = (
    COUPLER_KAPPA_Z*COUPLER_LENGTH_UM
    <= np.pi/2
)

weak_gaps = COUPLER_GAPS_NM[weak_mask]
weak_kappa = COUPLER_KAPPA_Z[weak_mask]

if len(weak_gaps) < 4:
    raise RuntimeError(
        "Not enough FEMWELL samples on the weak-coupling branch."
    )

# Remove any numerical non-monotonic tail produced by near-degenerate
# supermode ordering.  Keep the longest prefix with decreasing kappa_z.
keep = [0]
for i in range(1, len(weak_gaps)):
    if weak_kappa[i] < weak_kappa[keep[-1]]:
        keep.append(i)

weak_gaps = weak_gaps[keep]
weak_kappa = weak_kappa[keep]

KAPPA_GAP_SPLINE = PchipInterpolator(
    weak_gaps,
    np.log(weak_kappa),
    extrapolate=False,
)


def kappa_z_of_gap(gap_nm):
    gap_nm = np.asarray(gap_nm, dtype=float)

    if (
        np.any(gap_nm < weak_gaps[0])
        or np.any(gap_nm > weak_gaps[-1])
    ):
        raise ValueError(
            f"gap outside FEMWELL weak branch "
            f"[{weak_gaps[0]:.1f}, {weak_gaps[-1]:.1f}] nm"
        )

    value = np.exp(
        KAPPA_GAP_SPLINE(gap_nm)
    )

    return (
        float(value)
        if np.ndim(value) == 0
        else value
    )


def K_of_gap(gap_nm):
    return np.sin(
        kappa_z_of_gap(gap_nm)
        * COUPLER_LENGTH_UM
    )**2


K_WEAK_MAX = float(K_of_gap(weak_gaps[0]))
K_WEAK_MIN = float(K_of_gap(weak_gaps[-1]))


def gap_for_K(K_target):
    if K_target < K_WEAK_MIN or K_target > K_WEAK_MAX:
        raise RuntimeError(
            f"K={K_target:.8e} is outside the FEMWELL map "
            f"[{K_WEAK_MIN:.8e}, {K_WEAK_MAX:.8e}]"
        )

    return brentq(
        lambda gap_nm: K_of_gap(gap_nm)-K_target,
        float(weak_gaps[0]),
        float(weak_gaps[-1]),
    )


G1_FEMWELL_NM = gap_for_K(K1)
G12_FEMWELL_NM = gap_for_K(K2)
G3_FEMWELL_NM = gap_for_K(K3)


def quantize_gap_10nm(gap_nm):
    return 10.0 * np.round(float(gap_nm) / 10.0)


# Mask/layout rule for the sensitivity graph:
# gaps are restricted to 10 nm increments only.
G1_DESIGN_NM = quantize_gap_10nm(G1_FEMWELL_NM)
G12_DESIGN_NM = quantize_gap_10nm(G12_FEMWELL_NM)
G3_DESIGN_NM = quantize_gap_10nm(G3_FEMWELL_NM)

# ============================================================
# Plot 5 — physical-gap sensitivity using the FEMWELL map
# ============================================================

GAP_STEP_NM = 10.0
GAP_HALF_SPAN_NM = 30.0
FAB_WINDOW_NM = 10.0


def evaluate_gap_point(
    g1_nm,
    g12_nm,
    g3_nm,
):
    K1_value = float(K_of_gap(g1_nm))
    K2_value = float(K_of_gap(g12_nm))
    K3_value = float(K_of_gap(g3_nm))

    pump_value = series_rings_frequency(
        F0,
        K1_value,
        K2_value,
        K3_value,
    )

    signal_value = series_rings_frequency(
        FS,
        K1_value,
        K2_value,
        K3_value,
    )

    idler_value = series_rings_frequency(
        FI,
        K1_value,
        K2_value,
        K3_value,
    )

    Tp_value = pump_value["Tdrop"]
    Ts_value = signal_value["Tdrop"]
    Ti_value = idler_value["Tdrop"]

    pump_rejection_value = (
        -10*np.log10(
            max(Tp_value, 1e-300)
        )
    )

    useful_single = np.sqrt(
        max(Ts_value*Ti_value, 1e-300)
    )

    extinction_ratio_value = (
        10*np.log10(
            useful_single
            / max(Tp_value, 1e-300)
        )
    )

    eta_value = source_metrics(
        K1_value,
        K2_value,
        K3_value,
    )

    return {
        "K1": K1_value,
        "K2": K2_value,
        "K3": K3_value,
        "pump_rejection": pump_rejection_value,
        "extinction_ratio": extinction_ratio_value,
        "TsTi": Ts_value*Ti_value,
        "eta_esc": eta_value,
    }


def bounded_gap_sweep(design_gap):
    # The graph is restricted to mask-realizable values:
    # ..., 400, 410, 420, ... nm only.
    design_gap = quantize_gap_10nm(design_gap)

    lo = max(
        weak_gaps[0],
        design_gap-GAP_HALF_SPAN_NM,
    )

    hi = min(
        weak_gaps[-1],
        design_gap+GAP_HALF_SPAN_NM,
    )

    first = 10.0*np.ceil(lo/10.0)
    last = 10.0*np.floor(hi/10.0)

    return np.arange(
        first,
        last + 0.1,
        10.0,
    )


gap_cases = [
    (
        "g1",
        G1_DESIGN_NM,
        bounded_gap_sweep(G1_DESIGN_NM),
    ),
    (
        "g12",
        G12_DESIGN_NM,
        bounded_gap_sweep(G12_DESIGN_NM),
    ),
    (
        "g3",
        G3_DESIGN_NM,
        bounded_gap_sweep(G3_DESIGN_NM),
    ),
]

gap_sensitivity = {}

for gap_name, gap_design, gap_values in gap_cases:
    pump_rejection_values = []
    extinction_values = []
    pair_values = []
    eta_values = []

    for gap_value in gap_values:
        g1_value = G1_DESIGN_NM
        g12_value = G12_DESIGN_NM
        g3_value = G3_DESIGN_NM

        if gap_name == "g1":
            g1_value = gap_value
        elif gap_name == "g12":
            g12_value = gap_value
        else:
            g3_value = gap_value

        metrics = evaluate_gap_point(
            g1_value,
            g12_value,
            g3_value,
        )

        pump_rejection_values.append(
            metrics["pump_rejection"]
        )

        extinction_values.append(
            metrics["extinction_ratio"]
        )

        pair_values.append(
            metrics["TsTi"]
        )

        eta_values.append(
            metrics["eta_esc"]
        )

    gap_sensitivity[gap_name] = {
        "design": gap_design,
        "gap": np.asarray(gap_values),
        "pump_rejection": np.asarray(
            pump_rejection_values
        ),
        "extinction": np.asarray(
            extinction_values
        ),
        "TsTi": np.asarray(
            pair_values
        ),
        "eta": np.asarray(
            eta_values
        ),
    }


fig, axes = plt.subplots(
    2,
    3,
    figsize=(15.5, 8.8),
)

fig.suptitle(
    "FEMWELL gap sensitivity -- pump rejection, extinction, "
    "useful pair transmission and escape efficiency\n"
    "(10 nm mask steps; shaded = ±10 nm fabrication window; "
    "dotted = selected 10 nm mask gap)",
    y=0.995,
)

for col, gap_name in enumerate(
    ("g1", "g12", "g3")
):
    d = gap_sensitivity[gap_name]
    gap_design = d["design"]

    ax = axes[0, col]

    ax.plot(
        d["gap"],
        d["pump_rejection"],
        marker="o",
        lw=1.8,
        label="pump rejection",
    )

    ax.plot(
        d["gap"],
        d["extinction"],
        marker="s",
        lw=1.8,
        label="extinction ratio",
    )

    ax.axvspan(
        gap_design-FAB_WINDOW_NM,
        gap_design+FAB_WINDOW_NM,
        alpha=0.12,
    )

    ax.axvline(
        gap_design,
        ls=":",
        lw=1.4,
    )

    ax.set_title(
        f"({chr(97+col)}) {gap_name}   "
        f"design {gap_design:.1f} nm"
    )

    ax.set_xlabel(
        f"{gap_name} gap (nm)"
    )

    if col == 0:
        ax.set_ylabel("dB")

    ax.grid(alpha=0.3)

    if col == 0:
        ax.legend(
            fontsize=8,
            loc="best",
        )

    ax = axes[1, col]
    ax_r = ax.twinx()

    p1 = ax.plot(
        d["gap"],
        d["TsTi"],
        marker="^",
        lw=1.8,
        label=r"$T_sT_i$ (left)",
    )

    p2 = ax_r.plot(
        d["gap"],
        d["eta"],
        marker="v",
        ls="--",
        lw=1.8,
        label=r"$\eta_{esc}$ (right)",
    )

    ax.axvspan(
        gap_design-FAB_WINDOW_NM,
        gap_design+FAB_WINDOW_NM,
        alpha=0.12,
    )

    ax.axvline(
        gap_design,
        ls=":",
        lw=1.4,
    )

    ax.set_title(
        f"({chr(97+col)}) {gap_name}   "
        f"design {gap_design:.1f} nm"
    )

    ax.set_xlabel(
        f"{gap_name} gap (nm)"
    )

    ax.set_ylabel(r"$T_sT_i$")
    ax_r.set_ylabel(r"$\eta_{esc}$")

    ax.grid(alpha=0.3)

    if col == 0:
        lines = p1 + p2
        ax.legend(
            lines,
            [line.get_label() for line in lines],
            fontsize=8,
            loc="best",
        )

fig.tight_layout(
    rect=[0, 0, 1, 0.95]
)

fig.savefig(
    "gap_sensitivity_FEMWELL.png",
    dpi=180,
)

plt.close(fig)

# ============================================================
# Save compact result
# ============================================================

with open("result.txt", "w") as f:
    f.write("COUPLED-MICRORING PHOTON-PAIR SOURCE — FINAL DESIGN\n")
    f.write(f"width_nm = {WIDTH_NM:.1f}\n")
    f.write(f"height_nm = {HEIGHT_NM:.1f}\n")
    f.write(f"lambda0_nm = {LAMBDA0_NM:.1f}\n")
    f.write(f"ng = {NG:.8f}\n")
    f.write(f"Qi = {QI:.6e}\n")
    f.write(f"K1 = {K1:.10f}\n")
    f.write(f"K2 = {K2:.10e}\n")
    f.write(f"K3 = {K3:.10f}\n")
    f.write(f"kappa1 = {kappa1:.10f}\n")
    f.write(f"kappa2 = {kappa2:.10f}\n")
    f.write(f"kappa3 = {kappa3:.10f}\n")
    f.write(f"eta_esc = {eta_esc:.10f}\n")
    f.write(f"eta_pair = {eta_pair:.10f}\n")
    f.write(f"Q_loaded = {Q_loaded:.10e}\n")
    f.write(f"pump_rejection_dB = {pump_rejection_dB:.10f}\n")
    f.write(f"splitting_GHz = {splitting_hz*1e-9:.10f}\n")
    f.write(f"g1_exact_nm = {G1_FEMWELL_NM:.6f}\n")
    f.write(f"g12_exact_nm = {G12_FEMWELL_NM:.6f}\n")
    f.write(f"g3_exact_nm = {G3_FEMWELL_NM:.6f}\n")
    f.write(f"g1_mask_nm = {G1_DESIGN_NM:.0f}\n")
    f.write(f"g12_mask_nm = {G12_DESIGN_NM:.0f}\n")
    f.write(f"g3_mask_nm = {G3_DESIGN_NM:.0f}\n")

print_design_summary()

print("\n[6] Saved files")
print("  spectrum.png")
print("  buildup.png")
print("  cross_dependence.png")
print("  metrics.png")
print("  gap_sensitivity_FEMWELL.png")
print("  result.txt")
print("="*72)
