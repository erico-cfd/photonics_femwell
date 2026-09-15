"""
task4_FIXED_TEAM_OPTIMUM_10nm.py
================================
Focused final post-processing script for the double-ring Vernier device.

Purpose
-------
Use the three couplings selected by the teammate design note as FIXED design
inputs, map them to physical gaps with the FEMWELL supermode method, quantise
the gaps to a 10-nm mask grid, recompute the realised couplings, and generate
all final spectra/sensitivity plots from the realised geometry.

Imported coupling targets (power coupling K_i = |kappa_i|^2):
    K1 = 0.00427      bus1 <-> ring1
    K2 = 1.8e-4       ring1 <-> ring2
    K3 = 0.0136       ring2 <-> bus2

IMPORTANT PROVENANCE / SCOPE
----------------------------
The numerical K1/K2/K3 values above are imported from the teammate's final
coupling-design table. They were originally derived under that teammate's own
source-aware design assumptions. Here they are NOT re-optimised; they are
simply evaluated inside this project's current geometry/model:
    w = 400 nm, h = 220 nm, lambda_p = 1580 nm,
    A = 0.5 dB/cm, target FSR = 100 GHz, R2 = R1/2,
    straight directional-coupler length Lc = 5 um.

Physical implementation chain
-----------------------------
    imported target K_i
      -> inverse of FEMWELL-derived K(g)
      -> continuous ideal gaps g_i
      -> 10-nm mask quantisation
      -> FEMWELL-derived kappa_z(g_mask)
      -> realised K_i = sin^2(kappa_z Lc)
      -> exact two-ring transfer matrix
      -> pump rejection, Ts, Ti, Ts*Ti and spectra.

No extrapolation of K(g) is allowed. The FEMWELL coupling sweep is explicitly
extended to 800 nm because K2 = 1.8e-4 is weaker than the former 700-nm-domain
minimum (~3.15e-4 at Lc=5 um).

The script keeps EXACTLY TWO physical rings. No cascaded double-ring stages are
used.
"""

import math
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import shapely
from scipy.interpolate import PchipInterpolator, UnivariateSpline
from scipy.optimize import brentq
from skfem import Basis, ElementTriP0
from skfem.io import from_meshio

from femwell.maxwell.waveguide import compute_modes
from femwell.mesh import mesh_from_OrderedDict


# =============================================================================
# 0. FIXED PROJECT INPUTS + IMPORTED COUPLING TARGETS
# =============================================================================
c = 299792458.0

WIDTH_NM = 400.0
HEIGHT_NM = 220.0
PUMP_NM = 1580.0
A_DB_CM = 0.5
FSR_TARGET_GHZ = 100.0
R2_OVER_R1 = 0.5
L_COUPLE_UM = 5.0
MASK_GRID_NM = 10.0

# Imported from teammate final table. POWER couplings: K_i = |kappa_i|^2.
K_TARGET = np.array([0.00427, 1.8e-4, 0.0136], dtype=float)
COUPLER_NAMES = ("K1 bus1-ring1", "K2 ring1-ring2", "K3 ring2-bus2")
GAP_NAMES = ("g1", "g12", "g3")

w_um = WIDTH_NM * 1e-3
h_um = HEIGHT_NM * 1e-3

print("=" * 88)
print("FIXED TEAM-MEMBER COUPLING TARGETS")
print("=" * 88)
for name, K in zip(COUPLER_NAMES, K_TARGET):
    print(f"{name:18s}: K = {K:.8g} = {100*K:.5f}% ; |kappa| = {np.sqrt(K):.6f}")
print(f"Project geometry: {WIDTH_NM:.0f}x{HEIGHT_NM:.0f} nm, pump={PUMP_NM:.0f} nm, "
      f"A={A_DB_CM:.2f} dB/cm, target FSR={FSR_TARGET_GHZ:.0f} GHz, Lc={L_COUPLE_UM:.1f} um")


# =============================================================================
# 1. MATERIAL MODELS + STRAIGHT-WAVEGUIDE FEMWELL DISPERSION
# =============================================================================
def n_Si(l_um):
    """Crystalline-Si Sellmeier fit used in the previous Task-4 model."""
    l2 = l_um**2
    return math.sqrt(
        1
        + 10.6684293 * l2 / (l2 - 0.301516485**2)
        + 0.0030434748 * l2 / (l2 - 1.13475115**2)
        + 1.54133408 * l2 / (l2 - 1104**2)
    )


def n_SiO2(l_um):
    """Fused-silica Sellmeier fit used in the previous Task-4 model."""
    l2 = l_um**2
    return math.sqrt(
        1
        + 0.6961663 * l2 / (l2 - 0.0684043**2)
        + 0.4079426 * l2 / (l2 - 0.1162414**2)
        + 0.8974794 * l2 / (l2 - 9.896161**2)
    )


def strip_mesh(w, h):
    core = shapely.geometry.box(-w/2, 0, w/2, h)
    clad = shapely.geometry.box(-4*w, 0, 4*w, 6*h)
    box = shapely.geometry.box(-4*w, -6*h, 4*w, 0)
    res = dict(
        core={"resolution": 0.02, "distance": 0.3},
        clad={"resolution": 0.05, "distance": 0.3},
        box={"resolution": 0.05, "distance": 0.3},
    )
    return from_meshio(
        mesh_from_OrderedDict(
            OrderedDict(core=core, clad=clad, box=box),
            res,
            default_resolution_max=2,
        )
    )


def te0(mesh, lam_um):
    """Fundamental TE-like mode: mode with largest real TE fraction."""
    b = Basis(mesh, ElementTriP0())
    eps = b.zeros()
    for dom, nf in {"core": n_Si, "clad": n_SiO2, "box": n_SiO2}.items():
        eps[b.get_dofs(elements=dom)] = nf(lam_um)**2
    modes = compute_modes(b, eps, wavelength=lam_um, num_modes=3, order=1)
    return modes.sorted(key=lambda m: -np.real(m.te_fraction))[0]


print("\n" + "=" * 88)
print("1. STRAIGHT-WAVEGUIDE FEMWELL MODEL")
print("=" * 88)
mesh0 = strip_mesh(w_um, h_um)

# Same spectral domain as the validated final model; no spline extrapolation.
wl_nm = np.linspace(1500.0, 1600.0, 50)
neff_samples = []
for lam_nm in wl_nm:
    neff_samples.append(np.real(te0(mesh0, lam_nm*1e-3).n_eff))
neff_samples = np.asarray(neff_samples)

spl = UnivariateSpline(wl_nm, neff_samples, s=0, k=3, ext=2)
spl_d1 = spl.derivative(1)
n_eff = lambda lam_nm: spl(lam_nm)
n_g = lambda lam_nm: spl(lam_nm) - lam_nm*spl_d1(lam_nm)

print(f"n_eff({PUMP_NM:.0f} nm) = {n_eff(PUMP_NM):.6f}")
print(f"n_g  ({PUMP_NM:.0f} nm) = {n_g(PUMP_NM):.6f}")


# =============================================================================
# 2. RING GEOMETRY FROM FSR, WITH ODD PUMP ORDER AND R2 = R1/2
# =============================================================================
def geometry_from_fsr_target(fsr_target_GHz):
    """Build a pump-resonant 2:1 Vernier geometry from the requested FSR."""
    L_guess = c / (n_g(PUMP_NM) * fsr_target_GHz * 1e9)
    m1 = round(n_eff(PUMP_NM) * L_guess / (PUMP_NM * 1e-9))

    # Pump on an odd R1 longitudinal order so R2=L1/2 is anti-resonant at pump
    # and resonant at adjacent R1 orders in the ideal equal-index 2:1 model.
    if m1 % 2 == 0:
        m_lo, m_hi = m1 - 1, m1 + 1
        L_lo = m_lo * PUMP_NM * 1e-9 / n_eff(PUMP_NM)
        L_hi = m_hi * PUMP_NM * 1e-9 / n_eff(PUMP_NM)
        m1 = m_lo if abs(L_lo - L_guess) <= abs(L_hi - L_guess) else m_hi

    L1 = m1 * PUMP_NM * 1e-9 / n_eff(PUMP_NM)
    L2 = R2_OVER_R1 * L1
    R1_um = L1/(2*np.pi)*1e6
    R2_um = L2/(2*np.pi)*1e6
    fsr_GHz = c/(n_g(PUMP_NM)*L1)*1e-9
    fsr_nm = PUMP_NM**2/(n_g(PUMP_NM)*L1*1e9)

    return dict(
        L1=L1, L2=L2,
        R1_um=R1_um, R2_um=R2_um,
        m1=int(m1),
        fsr_target_GHz=float(fsr_target_GHz),
        fsr_GHz=float(fsr_GHz),
        fsr_nm=float(fsr_nm),
    )


def resonance(L_m, order_m, guess_nm):
    """Solve n_eff(lambda)*L/lambda = m inside the FEMWELL spline domain."""
    f = lambda lam_nm: n_eff(lam_nm)*L_m/(lam_nm*1e-9) - order_m
    fsr_guess_nm = guess_nm**2/(n_g(guess_nm)*L_m*1e9)
    return brentq(f, guess_nm - 0.6*fsr_guess_nm, guess_nm + 0.6*fsr_guess_nm)


geom = geometry_from_fsr_target(FSR_TARGET_GHZ)
geom["lam_s"] = resonance(geom["L1"], geom["m1"] + 1, PUMP_NM - geom["fsr_nm"])
geom["lam_i"] = resonance(geom["L1"], geom["m1"] - 1, PUMP_NM + geom["fsr_nm"])

print("\n" + "=" * 88)
print("2. CURRENT PROJECT RING GEOMETRY")
print("=" * 88)
print(f"target/actual FSR = {FSR_TARGET_GHZ:.3f} / {geom['fsr_GHz']:.3f} GHz")
print(f"R1 = {geom['R1_um']:.3f} um ; R2 = {geom['R2_um']:.3f} um ; m1 = {geom['m1']}")
print(f"signal = {geom['lam_s']:.6f} nm ; pump = {PUMP_NM:.6f} nm ; idler = {geom['lam_i']:.6f} nm")


# =============================================================================
# 3. LOSS MODEL + EXACT TWO-RING TRANSFER MATRIX
# =============================================================================
def B_field_per_m(A_dB_cm):
    """Field attenuation coefficient B: E(z)=E0 exp(-B z)."""
    return A_dB_cm*np.log(10)/20.0*100.0


def alpha_power_per_m(A_dB_cm):
    """Power attenuation coefficient alpha_power = 2B."""
    return A_dB_cm*np.log(10)/10.0*100.0


def alpha_rt_field(A_dB_cm, L_m):
    """Full-round-trip field attenuation."""
    return np.exp(-B_field_per_m(A_dB_cm)*L_m)


def theta(lam_nm, L_m):
    return 2*np.pi*n_eff(lam_nm)*L_m/(lam_nm*1e-9)


def couplers_independent(K1, K2, K3):
    """Lossless reciprocal couplers: Ki=|kappa_i|^2, |t_i|^2+Ki=1."""
    K = np.asarray([K1, K2, K3], dtype=float)
    if np.any(K < 0) or np.any(K >= 1):
        raise ValueError("All power couplings must satisfy 0 <= K < 1.")
    k = np.sqrt(K)
    t = np.sqrt(1-K)
    return k[0], t[0], k[1], t[1], k[2], t[2]


def series_rings(lam_nm, geom_local, K1, K2, K3, A_dB_cm, dn2=0.0):
    """Exact series-coupled two-ring linear system used in the final Task-4 model."""
    k1, t1, k2, t2, k3, t3 = couplers_independent(K1, K2, K3)
    L1, L2 = geom_local["L1"], geom_local["L2"]

    # The Rabus field equations use half-round-trip propagation factors.
    a1 = np.sqrt(alpha_rt_field(A_dB_cm, L1)) * np.exp(1j*theta(lam_nm, L1)/2)
    th2 = 2*np.pi*(n_eff(lam_nm)+dn2)*L2/(lam_nm*1e-9)
    a2 = np.sqrt(alpha_rt_field(A_dB_cm, L2)) * np.exp(1j*th2/2)

    M = np.array([
        [1 - t1*t2*a1**2,       t1*a1*k2*a2],
        [-t3*a2*k2*a1,          1 - t3*t2*a2**2],
    ], dtype=complex)

    E1a, E2b = np.linalg.solve(M, [-k1, 0.0])
    E1b = t2*a1*E1a - k2*a2*E2b
    E2a = k2*a1*E1a + t2*a2*E2b

    E_through = t1 + k1*a1*E1b
    E_drop = k3*a2*E2a
    return E_through, E_drop, E1a, E2a


def kpis(geom_local, K1, K2, K3, A_dB_cm, dn2=0.0):
    vals = {}
    for name, lam_nm in (("p", PUMP_NM), ("s", geom_local["lam_s"]), ("i", geom_local["lam_i"])):
        Eth, Ed, _, _ = series_rings(lam_nm, geom_local, K1, K2, K3, A_dB_cm, dn2)
        vals[f"Tthrough_{name}"] = abs(Eth)**2
        vals[f"Tdrop_{name}"] = abs(Ed)**2

    Tp = vals["Tdrop_p"]
    Ts = vals["Tdrop_s"]
    Ti = vals["Tdrop_i"]
    vals.update(
        rejection_dB=-10*np.log10(max(Tp, 1e-300)),
        Ts=Ts,
        Ti=Ti,
        TsTi=Ts*Ti,
        ERs_dB=10*np.log10(max(Ts, 1e-300)/max(Tp, 1e-300)),
        ERi_dB=10*np.log10(max(Ti, 1e-300)/max(Tp, 1e-300)),
    )
    return vals


# =============================================================================
# 4. FEMWELL COUPLING MAP: gap -> supermodes -> kappa_z -> K
# =============================================================================
def coupled_mesh(w, h, gap_um):
    c1 = shapely.geometry.box(-gap_um/2-w, 0, -gap_um/2, h)
    c2 = shapely.geometry.box(gap_um/2, 0, gap_um/2+w, h)
    span = 2*w + gap_um
    clad = shapely.geometry.box(-2*span, 0, 2*span, 6*h)
    box = shapely.geometry.box(-2*span, -6*h, 2*span, 0)
    res = {
        "core1": {"resolution": 0.02, "distance": 0.3},
        "core2": {"resolution": 0.02, "distance": 0.3},
        "clad": {"resolution": 0.05, "distance": 0.3},
        "box": {"resolution": 0.05, "distance": 0.3},
    }
    return from_meshio(
        mesh_from_OrderedDict(
            OrderedDict(core1=c1, core2=c2, clad=clad, box=box),
            res,
            default_resolution_max=2,
        )
    )


def kappa_z_femwell(gap_nm, lam_um=PUMP_NM*1e-3):
    """kappa_z = pi*(n_even-n_odd)/lambda [rad/um]."""
    msh = coupled_mesh(w_um, h_um, gap_nm*1e-3)
    b = Basis(msh, ElementTriP0())
    eps = b.zeros()
    for dom, nf in {"core1": n_Si, "core2": n_Si, "clad": n_SiO2, "box": n_SiO2}.items():
        eps[b.get_dofs(elements=dom)] = nf(lam_um)**2
    modes = compute_modes(b, eps, wavelength=lam_um, num_modes=2, order=1)
    ne, no = sorted((np.real(modes[0].n_eff), np.real(modes[1].n_eff)), reverse=True)
    return np.pi/lam_um*(ne-no)


print("\n" + "=" * 88)
print("4. FEMWELL GAP -> COUPLING MAP")
print("=" * 88)

# Explicitly extended beyond 700 nm because imported K2 is weaker than the
# previous domain minimum. No extrapolation is used later.
GAP_SAMPLES_NM = np.array([
    100, 125, 150, 175, 200, 250, 300, 350,
    400, 450, 500, 550, 600, 650, 700,
    725, 750, 775, 800,
], dtype=float)

kz_samples = np.array([kappa_z_femwell(g) for g in GAP_SAMPLES_NM])
for g, kz in zip(GAP_SAMPLES_NM, kz_samples):
    print(f"gap {g:6.1f} nm -> kappa_z = {kz:.8e} rad/um")

# Shape-preserving interpolation of log(kappa_z), only inside simulated points.
log_kz_interp = PchipInterpolator(GAP_SAMPLES_NM, np.log(kz_samples), extrapolate=False)


def kappa_z_interp(gap_nm):
    g = np.asarray(gap_nm, dtype=float)
    if np.any(g < GAP_SAMPLES_NM.min()) or np.any(g > GAP_SAMPLES_NM.max()):
        raise ValueError(
            f"gap={gap_nm} nm outside FEMWELL-simulated domain "
            f"[{GAP_SAMPLES_NM.min():.0f},{GAP_SAMPLES_NM.max():.0f}] nm"
        )
    val = np.exp(log_kz_interp(g))
    return float(val) if val.ndim == 0 else val


def K_of_gap(gap_nm):
    """Power coupling for the fixed straight coupling length Lc."""
    return np.sin(kappa_z_interp(gap_nm)*L_COUPLE_UM)**2


GAP_MIN_NM = float(GAP_SAMPLES_NM.min())
GAP_MAX_NM = float(GAP_SAMPLES_NM.max())
K_MIN = min(float(K_of_gap(GAP_MIN_NM)), float(K_of_gap(GAP_MAX_NM)))
K_MAX = max(float(K_of_gap(GAP_MIN_NM)), float(K_of_gap(GAP_MAX_NM)))
print(f"Physical K domain at Lc={L_COUPLE_UM:.1f} um: {K_MIN:.6e} <= K <= {K_MAX:.6e}")


def gap_for_K(K_target):
    """Inverse K(g), strictly inside the physically simulated FEMWELL domain."""
    if not (K_MIN <= K_target <= K_MAX):
        raise RuntimeError(
            f"Target K={K_target:.6e} is outside the current physical K(g) domain "
            f"[{K_MIN:.6e},{K_MAX:.6e}]. Extend GAP_SAMPLES_NM and rerun FEMWELL; "
            "do not extrapolate."
        )
    return brentq(lambda g: K_of_gap(g)-K_target, GAP_MIN_NM, GAP_MAX_NM)


def snap_gap_to_mask(gap_nm, step_nm=MASK_GRID_NM):
    """Nearest 10-nm mask value; half-up rounding, clipped to FEMWELL domain."""
    snapped = np.floor(float(gap_nm)/step_nm + 0.5)*step_nm
    return float(np.clip(snapped, GAP_MIN_NM, GAP_MAX_NM))


# =============================================================================
# 5. TARGET K -> IDEAL GAP -> 10-nm MASK GAP -> REALISED K
# =============================================================================
g_ideal = np.array([gap_for_K(K) for K in K_TARGET], dtype=float)
g_mask = np.array([snap_gap_to_mask(g) for g in g_ideal], dtype=float)
kz_real = np.array([kappa_z_interp(g) for g in g_mask], dtype=float)
K_real = np.array([K_of_gap(g) for g in g_mask], dtype=float)

print("\n" + "=" * 88)
print("5. IMPORTED TARGET -> PHYSICAL GAP -> 10-nm MASK -> REALISED COUPLING")
print("=" * 88)
print(f"{'coupler':18s} {'K target (%)':>14s} {'g ideal (nm)':>14s} {'g mask (nm)':>12s} "
      f"{'kappa_z':>13s} {'K real (%)':>13s} {'dK/K':>10s}")
for name, Kt, gi, gm, kz, Kr in zip(COUPLER_NAMES, K_TARGET, g_ideal, g_mask, kz_real, K_real):
    print(f"{name:18s} {100*Kt:14.6f} {gi:14.3f} {gm:12.0f} {kz:13.6e} "
          f"{100*Kr:13.6f} {100*(Kr/Kt-1):+9.2f}%")

ideal_perf = kpis(geom, *K_TARGET, A_DB_CM)
real_perf = kpis(geom, *K_real, A_DB_CM)

print("\nTransfer-function KPIs in CURRENT project geometry:")
print(f"  imported continuous target: rejection={ideal_perf['rejection_dB']:.3f} dB, "
      f"Ts={ideal_perf['Ts']:.6f}, Ti={ideal_perf['Ti']:.6f}, TsTi={ideal_perf['TsTi']:.6f}")
print(f"  10-nm mask realised      : rejection={real_perf['rejection_dB']:.3f} dB, "
      f"Ts={real_perf['Ts']:.6f}, Ti={real_perf['Ti']:.6f}, TsTi={real_perf['TsTi']:.6f}")
print("  IMPORTANT: these KPIs evaluate the imported coupling values in THIS project's")
print("  400x220-nm, 1580-nm, ~100-GHz geometry; they need not reproduce the teammate")
print("  note's 20/10-um, 1550-nm numerical performance.")


# =============================================================================
# 6. SOURCE-AWARE DIAGNOSTIC (SEPARATE FROM LINEAR Ts*Ti)
# =============================================================================
def source_aware_rates(K1, K2, K3, geom_local, A_dB_cm):
    """Coupled-mode escape metric from the teammate design note, evaluated here."""
    vg = c/n_g(PUMP_NM)
    alpha_p = alpha_power_per_m(A_dB_cm)
    gamma_i1 = alpha_p*vg/2
    gamma_i2 = alpha_p*vg/2
    TR1 = geom_local["L1"]/vg
    TR2 = geom_local["L2"]/vg
    gamma1 = -np.log(max(1-K1, 1e-300))/(2*TR1)
    gamma2 = -np.log(max(1-K3, 1e-300))/(2*TR2)
    mu = np.sqrt(max(K2, 0.0))/np.sqrt(TR1*TR2)
    Gamma1 = gamma_i1 + gamma1
    Gamma2 = gamma_i2 + gamma2
    eta_esc = (mu**2/(Gamma1*Gamma2 + mu**2))*(gamma2/Gamma2)
    K1_crit = 1 - np.exp(-alpha_p*geom_local["L1"])
    gamma2_star = np.sqrt(gamma_i2*(gamma_i2 + mu**2/Gamma1))
    K3_star = 1 - np.exp(-2*gamma2_star*TR2)
    return dict(
        eta_esc=eta_esc,
        eta_pair_proxy=eta_esc**2,
        K1_crit=K1_crit,
        K3_star=K3_star,
        mu_over_gamma_i=mu/gamma_i1,
    )


sa_target = source_aware_rates(*K_TARGET, geom, A_DB_CM)
sa_real = source_aware_rates(*K_real, geom, A_DB_CM)
print("\n" + "=" * 88)
print("6. SOURCE-AWARE DIAGNOSTIC IN CURRENT PROJECT GEOMETRY")
print("=" * 88)
print(f"continuous target eta_esc = {sa_target['eta_esc']:.6f}")
print(f"10-nm mask eta_esc        = {sa_real['eta_esc']:.6f}")
print(f"10-nm mask eta_esc^2      = {sa_real['eta_pair_proxy']:.6f}")
print(f"derived critical K1*      = {100*sa_real['K1_crit']:.6f}%")
print(f"derived extraction K3*    = {100*sa_real['K3_star']:.6f}%")
print("Ts*Ti is a linear filter-transmission proxy; eta_esc is an internal-photon escape metric.")


# =============================================================================
# 7. FIGURE 1 -- PHYSICAL COUPLING MAP AND FOUND GAPS
# =============================================================================
g_dense = np.linspace(GAP_MIN_NM, GAP_MAX_NM, 500)
K_dense = np.array([K_of_gap(g) for g in g_dense])

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.semilogy(g_dense, K_dense, label=r"$K(g)=\sin^2[\kappa_z(g)L_c]$")
ax.semilogy(GAP_SAMPLES_NM, [K_of_gap(g) for g in GAP_SAMPLES_NM], "o", ms=4,
            label="FEMWELL-supported samples")
for i, (Kt, gi, gm) in enumerate(zip(K_TARGET, g_ideal, g_mask)):
    ax.axhline(Kt, ls="--", alpha=.45)
    ax.axvline(gi, ls="--", alpha=.6, label=f"{GAP_NAMES[i]} ideal = {gi:.1f} nm")
    ax.axvline(gm, ls=":", alpha=.9, label=f"{GAP_NAMES[i]} mask = {gm:.0f} nm")
ax.set_xlabel("coupler gap (nm)")
ax.set_ylabel(r"power coupling $K=|\kappa|^2$")
ax.set_title("Fixed teammate couplings mapped to physical gaps")
ax.grid(alpha=.3, which="both")
ax.legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig("team_fixed_fig1_gap_map.png", dpi=180)
plt.close(fig)


# =============================================================================
# 8. FIGURE 2 -- FINAL TWO-RING SPECTRUM FROM 10-nm MASK-REALISED COUPLINGS
# =============================================================================
span_nm = 2.6*geom["fsr_nm"]
lams = np.linspace(PUMP_NM-span_nm, PUMP_NM+span_nm, 20001)

out_real = np.array([
    series_rings(l, geom, *K_real, A_DB_CM)
    for l in lams
])
out_target = np.array([
    series_rings(l, geom, *K_TARGET, A_DB_CM)
    for l in lams
])

fig, ax = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

B1 = abs(out_real[:, 2])**2
B2 = abs(out_real[:, 3])**2
ax[0].plot(lams-PUMP_NM, B1/max(B1.max(), 1e-300), lw=.9)
ax[0].set_ylabel("ring 1\nbuildup (norm.)")
ax[0].set_title(f"(a) ring 1 field buildup, R1 = {geom['R1_um']:.1f} um")

ax[1].plot(lams-PUMP_NM, B2/max(B2.max(), 1e-300), lw=.9)
ax[1].set_ylabel("ring 2\nbuildup (norm.)")
ax[1].set_title(f"(b) ring 2 field buildup, R2 = {geom['R2_um']:.1f} um")

ax[2].semilogy(lams-PUMP_NM, np.maximum(abs(out_real[:, 1])**2, 1e-14),
               label=r"$T_{drop}$, 10-nm mask")
ax[2].semilogy(lams-PUMP_NM, np.maximum(abs(out_real[:, 0])**2, 1e-14),
               label=r"$T_{through}$, 10-nm mask")
ax[2].semilogy(lams-PUMP_NM, np.maximum(abs(out_target[:, 1])**2, 1e-14),
               ls="--", alpha=.75, label=r"$T_{drop}$, continuous target")
ax[2].axhline(1e-10, ls=":", label="-100 dB reference")
ax[2].scatter([0], [real_perf["Tdrop_p"]], marker="x", s=55, label="pump")
ax[2].scatter([geom["lam_s"]-PUMP_NM], [real_perf["Ts"]], marker="o", s=30, label="signal")
ax[2].scatter([geom["lam_i"]-PUMP_NM], [real_perf["Ti"]], marker="o", s=30, label="idler")
ax[2].set_ylim(1e-12, 3)
ax[2].set_ylabel("transmission")
ax[2].set_xlabel(r"$\lambda-\lambda_p$ (nm)")
ax[2].set_title(f"(c) realised spectrum: rejection={real_perf['rejection_dB']:.2f} dB, "
                f"Ts={real_perf['Ts']:.3f}, Ti={real_perf['Ti']:.3f}")
ax[2].legend(fontsize=7, ncol=2)

for a in ax:
    for lam, ls, col in ((PUMP_NM, ":", "k"), (geom["lam_s"], ":", "r"), (geom["lam_i"], ":", "b")):
        a.axvline(lam-PUMP_NM, ls=ls, color=col, lw=1)
    a.grid(alpha=.3)

fig.suptitle("Exactly two rings: spectrum from teammate-selected couplings after 10-nm gap quantisation")
fig.tight_layout()
fig.savefig("team_fixed_fig2_spectrum.png", dpi=180)
plt.close(fig)


# =============================================================================
# 9. FIGURE 3 -- TARGET VS MASK-REALISED DESIGN
# =============================================================================
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
x = np.arange(3)

ax[0].semilogy(x, 100*K_TARGET, "o-", label="imported target")
ax[0].semilogy(x, 100*K_real, "s--", label="10-nm mask realised")
ax[0].set_xticks(x, ["K1", "K2", "K3"])
ax[0].set_ylabel("power coupling K (%)")
ax[0].set_title("(a) target vs realised coupling")
ax[0].grid(alpha=.3, which="both")
ax[0].legend(fontsize=8)

bw = 0.34
ax[1].bar(x-bw/2, g_ideal, width=bw, label="continuous ideal gap")
ax[1].bar(x+bw/2, g_mask, width=bw, label="10-nm mask gap")
ax[1].set_xticks(x, GAP_NAMES)
ax[1].set_ylabel("gap (nm)")
ax[1].set_title("(b) physical gap quantisation")
ax[1].grid(alpha=.3, axis="y")
ax[1].legend(fontsize=8)

ax[2].semilogy(lams-PUMP_NM, np.maximum(abs(out_target[:,1])**2, 1e-14), label="continuous target")
ax[2].semilogy(lams-PUMP_NM, np.maximum(abs(out_real[:,1])**2, 1e-14), ls="--", label="10-nm mask realised")
ax[2].axhline(1e-10, ls=":", label="-100 dB")
ax[2].set_ylim(1e-12, 2)
ax[2].set_xlabel(r"$\lambda-\lambda_p$ (nm)")
ax[2].set_ylabel(r"$T_{drop}$")
ax[2].set_title("(c) transfer-function impact")
ax[2].grid(alpha=.3)
ax[2].legend(fontsize=8)

fig.suptitle("Imported optimum -> physical gaps -> 10-nm mask-realised design")
fig.tight_layout()
fig.savefig("team_fixed_fig3_mask_impact.png", dpi=180)
plt.close(fig)


# =============================================================================
# 10. FIGURE 4 -- GAP SENSITIVITY AROUND THE MASK DESIGN
# =============================================================================
# This is NOT a claim that fabrication error is +/-30 nm. It is a deterministic
# sensitivity plot sampled on the same 10-nm geometric grid used for the mask.
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

for j, a in enumerate(axes):
    center = g_mask[j]
    g_test = np.arange(center-30, center+30.1, MASK_GRID_NM)
    g_test = g_test[(g_test >= GAP_MIN_NM) & (g_test <= GAP_MAX_NM)]

    rej = []
    pair = []
    eta = []
    for gv in g_test:
        Kvec = K_real.copy()
        Kvec[j] = K_of_gap(gv)
        perf = kpis(geom, *Kvec, A_DB_CM)
        sa = source_aware_rates(*Kvec, geom, A_DB_CM)
        rej.append(perf["rejection_dB"])
        pair.append(perf["TsTi"])
        eta.append(sa["eta_esc"])

    a.plot(g_test, rej, "o-", label="pump rejection")
    a.axvline(center, ls=":", color="k", label=f"mask {center:.0f} nm")
    a.set_xlabel(f"{GAP_NAMES[j]} (nm)")
    a.set_ylabel("pump rejection (dB)")
    a.grid(alpha=.3)
    b = a.twinx()
    b.plot(g_test, pair, "s--", label=r"$T_sT_i$")
    b.plot(g_test, eta, "^--", label=r"$\eta_{esc}$")
    b.set_ylabel("transmission / escape metric")
    b.set_ylim(0, 1)
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = b.get_legend_handles_labels()
    a.legend(h1+h2, l1+l2, fontsize=7, loc="best")
    a.set_title(f"({chr(97+j)}) sensitivity to {GAP_NAMES[j]}")

fig.suptitle("Deterministic 10-nm-step gap sensitivity around the selected mask design")
fig.tight_layout()
fig.savefig("team_fixed_fig4_gap_sensitivity.png", dpi=180)
plt.close(fig)


# =============================================================================
# 11. FINAL CONSOLE SUMMARY
# =============================================================================
print("\n" + "=" * 88)
print("FINAL SUMMARY")
print("=" * 88)
print("Imported target power couplings:")
for nm, K in zip(("K1", "K2", "K3"), K_TARGET):
    print(f"  {nm} = {K:.8g} ({100*K:.6f}%)")
print("\nPhysical implementation from FEMWELL K(g):")
for nm, gi, gm, Kr in zip(GAP_NAMES, g_ideal, g_mask, K_real):
    print(f"  {nm}: ideal {gi:.3f} nm -> mask {gm:.0f} nm -> realised K={Kr:.8g} ({100*Kr:.6f}%)")
print("\nFinal plots use the MASK-REALISED K values, not the imported continuous targets.")
print(f"  pump rejection = {real_perf['rejection_dB']:.3f} dB")
print(f"  Ts = {real_perf['Ts']:.6f}")
print(f"  Ti = {real_perf['Ti']:.6f}")
print(f"  Ts*Ti = {real_perf['TsTi']:.6f}")
print(f"  source-aware eta_esc = {sa_real['eta_esc']:.6f}")
print("\nGenerated figures:")
print("  team_fixed_fig1_gap_map.png")
print("  team_fixed_fig2_spectrum.png")
print("  team_fixed_fig3_mask_impact.png")
print("  team_fixed_fig4_gap_sensitivity.png")
