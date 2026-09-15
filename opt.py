"""
mask_gaps.py -- from design couplings to REAL, fabricable gaps
===============================================================
The design gives target coupling coefficients; a mask gives gaps on a discrete
grid. This module closes that loop: it inverts the coupling-vs-gap relation,
snaps the result to the mask grid, and returns the couplings that are ACTUALLY
realised -- which are the ones that should feed the transfer function.

Chain, all from the supplied material:
  kappa_z(g) = pi (n_even - n_odd)/lambda      [CHROST15 Eq. 4.3]  (Femwell supermodes)
  kappa_z(g) = kappa_0 exp(-g/decay)           [fit; exponential form of RABUS07 Eq. 2.66,
                                                confirmed by CHROST15 Eq. 4.7]
  |kappa|^2  = sin^2(kappa_z * L_c)            [CHROST15 Eq. 4.8]
  |t|^2 + |kappa|^2 = 1                        [RABUS07 Eq. 2.2]

Usage
  python3 mask_gaps.py                     # design point of the lean model
  python3 mask_gaps.py --step 5            # 5 nm mask grid
  python3 mask_gaps.py --K1 0.018 --Lc 1.0 # another operating point

If the supermode fit of your own geometry differs, pass --kappa0 and --decay
(both printed by task4_lean.py in section [T4-4b]).
"""
import argparse

import numpy as np
from scipy.optimize import brentq

# ---------------------------------------------------------------- CLI
ap = argparse.ArgumentParser()
ap.add_argument("--kappa0", type=float, default=0.247,
                help="prefactor of the supermode fit, rad/um (default: lean 500x220 @1550)")
ap.add_argument("--decay", type=float, default=113.0,
                help="decay length of the fit, nm")
ap.add_argument("--Lc", type=float, default=5.0, help="straight coupling length, um")
ap.add_argument("--K1", type=float, default=0.03, help="bus power coupling (design)")
ap.add_argument("--ratio", type=float, default=0.5,
                help="r2/r1 (0.5 for the 2:1 Vernier)")
ap.add_argument("--step", type=float, default=10.0, help="mask grid step, nm")
ap.add_argument("--gmin", type=float, default=100.0, help="smallest fabricable gap, nm")
ap.add_argument("--gmax", type=float, default=900.0, help="largest usable gap, nm")
ap.add_argument("--dw", type=float, default=10.0,
                help="fabrication sensitivity window, +/- nm of width (default 10 = stress test)")
a = ap.parse_args()

kappa_z = lambda g: a.kappa0 * np.exp(-g / a.decay)          # CHROST15 Eq. 4.3 + fit
K_of_gap = lambda g: np.sin(kappa_z(g) * a.Lc) ** 2          # CHROST15 Eq. 4.8


def check_monotonic():
    """K(g) = sin^2(kappa_z Lc) is monotonic in g only while kappa_z*Lc < pi/2.
    Past that the sine wraps, K comes back down, and inverting K -> g is ambiguous
    (CHROST15 makes the same point: a 15 um coupler at a 100 nm gap couples almost
    nothing because the light has gone through a full cycle). Refuse to continue."""
    arg_max = kappa_z(a.gmin) * a.Lc
    if arg_max >= np.pi / 2:
        raise SystemExit(
            f"kappa_z*Lc = {arg_max:.3f} rad at g = {a.gmin:.0f} nm exceeds pi/2 = 1.571:\n"
            f"K(g) is NOT monotonic on [{a.gmin:.0f}, {a.gmax:.0f}] nm and the inversion is\n"
            f"ambiguous. Reduce --Lc (currently {a.Lc} um) or raise --gmin.")
    return arg_max


def gap_for_K(K_target):
    """Invert Eq. 4.8. Returns None if the target lies outside the fabricable window,
    rather than extrapolating a fit outside the range it was measured on."""
    lo, hi = K_of_gap(a.gmax), K_of_gap(a.gmin)          # K decreases with g
    if not (lo <= K_target <= hi):
        return None
    return brentq(lambda g: K_of_gap(g) - K_target, a.gmin, a.gmax)


def snap(g):
    """Nearest point on the mask grid."""
    return round(g / a.step) * a.step


# ---------------------------------------------------------------- design point
# Butterworth, general form for r1 != r2 [RABUS07 Eqs. 2.74-2.75]:
#   kappa_2 = (kappa_1^2/2) sqrt(r2/r1);  reduces to Eq. 2.76 when r1 = r2.
# kappa_3 = kappa_1 [RABUS07 sec 2.2.1].
k1_t = np.sqrt(a.K1)
k2_t = (k1_t ** 2 / 2) * np.sqrt(a.ratio)
k3_t = k1_t
TARGETS = [("bus <-> R1", k1_t), ("R1  <-> R2", k2_t), ("R2  <-> drop", k3_t)]

arg_max = check_monotonic()
print("=" * 74)
print("MASK GAPS -- design couplings -> fabricable gaps -> realised couplings")
print("=" * 74)
print(f"fit: kappa_z = {a.kappa0:.3f} exp(-g/{a.decay:.0f} nm) rad/um | "
      f"L_c = {a.Lc:.1f} um | mask step = {a.step:.0f} nm")
print(f"monotonicity OK: kappa_z*Lc = {arg_max:.3f} rad at g = {a.gmin:.0f} nm (< pi/2)")
print(f"fabricable window: gap [{a.gmin:.0f}, {a.gmax:.0f}] nm -> "
      f"|kappa|^2 [{K_of_gap(a.gmax):.2e}, {K_of_gap(a.gmin):.2e}]")

print(f"\n{'coupler':14}{'|k|^2 target':>14}{'gap ideal':>11}{'gap MASK':>10}"
      f"{'|k|^2 real':>13}{'|k| real':>11}{'t real':>10}{'err':>9}")
realised = {}
for name, kt in TARGETS:
    Kt = kt ** 2
    g = gap_for_K(Kt)
    if g is None:
        print(f"{name:14}{Kt:14.4e}{'--':>11}{'--':>10}{'OUTSIDE fabricable window':>43}")
        realised[name] = None
        continue
    gq = snap(g)
    gq = min(max(gq, a.gmin), a.gmax)
    Kr = K_of_gap(gq)
    kr, tr = np.sqrt(Kr), np.sqrt(1 - Kr)
    realised[name] = dict(gap=gq, K=Kr, k=kr, t=tr, K_target=Kt)
    print(f"{name:14}{Kt:14.4e}{g:9.1f}nm{gq:8.0f}nm{Kr:13.4e}{kr:11.5f}{tr:10.6f}"
          f"{(Kr/Kt-1)*100:+8.1f}%")

if any(v is None for v in realised.values()):
    raise SystemExit("\nAt least one coupling is not reachable with this L_c. "
                     "Reduce --Lc: K scales roughly as (kappa_z L_c)^2 for small argument, "
                     "so halving L_c divides the whole reachable range by ~4.")

print("\nSanity check |t|^2 + |kappa|^2 = 1 [RABUS07 Eq. 2.2]:")
for name, v in realised.items():
    print(f"  {name:14}: {v['t']**2 + v['k']**2:.12f}")

# ---------------------------------------------------------------- Butterworth ratio
k1r = realised["bus <-> R1"]["k"]
k2r = realised["R1  <-> R2"]["k"]
k3r = realised["R2  <-> drop"]["k"]
ratio_t = 0.5 * np.sqrt(a.ratio)
ratio_r = k2r / k1r ** 2
print(f"\nButterworth condition kappa_2/kappa_1^2 [RABUS07 Eq. 2.75]:")
print(f"  target    = {ratio_t:.5f}")
print(f"  realised  = {ratio_r:.5f}   ({ratio_r/ratio_t-1:+.2%})")
print(f"  kappa_3/kappa_1 = {k3r/k1r:.5f}  (target 1.000 per RABUS07 sec 2.2.1)")
if ratio_r < ratio_t:
    print("  -> below target: the inter-ring coupling is weaker than maximally flat,")
    print("     i.e. FURTHER from the mode-splitting threshold. The safe direction.")
else:
    print("  -> above target: closer to mode splitting; check the drop-band shape.")

# ---------------------------------------------------------------- sensitivity
dlnK_dg = -2.0 / a.decay
print(f"\nSensitivity of the mask grid:")
print(f"  d ln|kappa|^2/dg = {dlnK_dg*100:+.2f} %/nm")
print(f"  worst-case rounding (+/-{a.step/2:.0f} nm) -> "
      f"{(np.exp(-dlnK_dg*a.step/2)-1)*100:+.0f} % / "
      f"{(np.exp(dlnK_dg*a.step/2)-1)*100:+.0f} % in |kappa|^2")
print("  A gap error changes the COUPLING, not the resonance wavelength: it moves Q_e")
print("  and the notch depth, but does not detune the Vernier coincidence. Compare with")
print("  a width error, which shifts resonances by 585 pm/nm and is what actually breaks")
print("  the design (differential tolerance ~0.007 nm of width in the lean model).")

# ---------------------------------------------------------------- KPI impact
print("\n" + "=" * 74)
print("KPI impact of the mask grid (transfer function, RABUS07 Eqs. 2.58-2.63)")
print("=" * 74)
try:
    R1_um, m1, A_dB = 113.9, 1129, 1.0          # lean design point
    L1 = 2 * np.pi * R1_um * 1e-6
    L2 = L1 * a.ratio
    al = lambda A, L: np.exp(-(A * np.log(10) / 20) * L * 100)
    a1m, a2m = al(A_dB, L1), al(A_dB, L2)

    def T_drop(th1, th2, K1, K2, K3):
        k1, k2, k3 = np.sqrt(K1), np.sqrt(K2), np.sqrt(K3)
        t1, t2, t3 = np.sqrt(1 - K1), np.sqrt(1 - K2), np.sqrt(1 - K3)
        A1 = np.sqrt(a1m) * np.exp(1j * th1 / 2)
        A2 = np.sqrt(a2m) * np.exp(1j * th2 / 2)
        M = np.array([[1 - t1 * t2 * A1 ** 2, t1 * A1 * k2 * A2],
                      [-t3 * A2 * k2 * A1, 1 - t3 * t2 * A2 ** 2]])
        E1a, E2b = np.linalg.solve(M, [-k1, 0.0])
        E2a = k2 * A1 * E1a + t2 * A2 * E2b
        return abs(k3 * A2 * E2a) ** 2

    # with L2 = L1/2 and one index, theta2 = theta1/2 exactly; m1 odd -> pump
    # lands on an anti-resonance of ring 2, signal (m1+1, even) on a resonance.
    thp = (2 * np.pi * m1, np.pi * m1)
    ths = (2 * np.pi * (m1 + 1), np.pi * (m1 + 1))

    def kpi(K1, K2, K3):
        Tp = T_drop(*thp, K1, K2, K3)
        Ts = T_drop(*ths, K1, K2, K3)
        return -10 * np.log10(Tp), Ts, 10 * np.log10(Ts / Tp)

    id_ = kpi(k1_t ** 2, k2_t ** 2, k3_t ** 2)
    ma_ = kpi(realised["bus <-> R1"]["K"], realised["R1  <-> R2"]["K"],
              realised["R2  <-> drop"]["K"])
    print(f"{'':26}{'rejection':>12}{'T_drop(s)':>12}{'extinction':>13}")
    print(f"{'ideal (not fabricable)':26}{id_[0]:10.2f} dB{id_[1]:12.4f}{id_[2]:11.2f} dB")
    print(f"{'mask grid':26}{ma_[0]:10.2f} dB{ma_[1]:12.4f}{ma_[2]:11.2f} dB")
    print(f"{'change':26}{ma_[0]-id_[0]:+10.2f} dB{(ma_[1]/id_[1]-1)*100:+11.2f} %"
          f"{ma_[2]-id_[2]:+11.2f} dB")

    ideals = [v["K_target"] for v in realised.values()]
    gaps_i = [gap_for_K(K) for K in ideals]
    lo = hi = None
    for d1 in (-a.step / 2, a.step / 2):
        for d2 in (-a.step / 2, a.step / 2):
            for d3 in (-a.step / 2, a.step / 2):
                r = kpi(K_of_gap(gaps_i[0] + d1), K_of_gap(gaps_i[1] + d2),
                        K_of_gap(gaps_i[2] + d3))
                lo = r if lo is None or r[1] < lo[1] else lo
                hi = r if hi is None or r[1] > hi[1] else hi
    print(f"\nWorst case over all +/-{a.step/2:.0f} nm rounding combinations:")
    print(f"  T_drop(signal): {lo[1]:.4f} .. {hi[1]:.4f}   (ideal {id_[1]:.4f})")
    print(f"  rejection:      {min(lo[0], hi[0]):.2f} .. {max(lo[0], hi[0]):.2f} dB "
          f"(ideal {id_[0]:.2f})")
except Exception as e:                       # pragma: no cover
    print(f"  (KPI block skipped: {e})")

print(f"\nMASK GAPS TO DRAW: " + ", ".join(
    f"{n.strip()} = {v['gap']:.0f} nm" for n, v in realised.items()))

# =============================================================================
# FABRICATION VARIATION -- distinct from mask quantization
# =============================================================================
# Two effects that must NOT be conflated:
#   (1) MASK QUANTIZATION (above): a layout choice. The drawn gap sits on a grid.
#   (2) FABRICATION VARIATION (here): a process effect. The FABRICATED dimensions
#       differ from the drawn ones.
#
# Scale of (2) [REF: THOMSON16, "Roadmap on silicon photonics"]: for best-in-class
# 193 nm immersion DUV, observed LINEWIDTH fluctuations are of order 5 nm and
# silicon-thickness fluctuations of order 1 nm. Note carefully: that is a width
# (critical-dimension) figure. It is NOT a measured gap standard deviation, and it
# is used here only to set the scale of the sensitivity window.
#
# The coupling: a lithographic CD excursion that makes the waveguides WIDER eats
# into the space between them, so the gap shrinks by the same amount:
#         delta_g = -delta_w
# [REF: CHROST15 sec 4.1 -- "when the waveguide width shrinks, the gap increases,
#  and vice versa", and their fabrication-sensitivity procedure applies the same
#  but opposite change to the gap; PRINZEN13, Opt. Express 21, 17212 (2013), a
#  3D-FEM study of exactly this for SOI directional couplers and ring resonators
#  (rib waveguides, so the numbers transfer only qualitatively to our strip).]
#
# NOT modelled: sidewall angle, etch-depth and thickness excursions, which
# PRINZEN13 also covers and which would each add their own term.

DN_DW = 1.58e-3      # dn_eff/dw per nm of width [Femwell, 500x220 strip @1550]
N_G = 4.1932         # group index at the pump   [same model]
LAM_NM = 1550.0
FSR_NM = 0.801       # design FSR of the lean model
FWHM_PM = 6.0        # design linewidth

print("\n" + "=" * 74)
print("FABRICATION VARIATION (process), with delta_g = -delta_w")
print("=" * 74)
print(f"scale: |delta_w| ~ 5 nm  [THOMSON16, DUV linewidth fluctuation -- a WIDTH")
print(f"       figure, not a measured gap sigma]; thickness ~ 1 nm (not modelled)")
print(f"coupling: delta_g = -delta_w  [CHROST15 sec 4.1; PRINZEN13]")
print(f"shift law: dlambda/lambda = dn_eff/n_g -> "
      f"{LAM_NM*DN_DW/N_G*1e3:.0f} pm per nm of width\n")

DW = a.dw
print(f"SENSITIVITY WINDOW USED: +/-{DW:.0f} nm of width")
if DW > 5:
    print(f"  This is a STRESS TEST, not a foundry tolerance: {DW/5:.0f}x the ~5 nm")
    print(f"  linewidth fluctuation THOMSON16 reports for 193 nm immersion DUV.")
print()
print(f"{'dw (nm)':>8}{'dg (nm)':>9}{'lambda shift':>14}{'in FSR':>9}"
      f"{'in linewidths':>15}{'K1 change':>12}")
for dw in (-DW, -DW / 2, -1.0, 1.0, DW / 2, DW):
    dg = -dw
    dlam_pm = LAM_NM * DN_DW * dw / N_G * 1e3
    g_new = realised["bus <-> R1"]["gap"] + dg
    K_new = K_of_gap(g_new) if a.gmin <= g_new <= a.gmax else float("nan")
    dK = (K_new / realised["bus <-> R1"]["K"] - 1) * 100
    print(f"{dw:8.1f}{dg:9.1f}{dlam_pm:12.0f} pm{dlam_pm*1e-3/FSR_NM:9.2f}"
          f"{dlam_pm/FWHM_PM:15.0f}{dK:+11.1f}%")

# ---- KPI impact, common vs differential, WITH the gap anticorrelation --------
print(f"\nKPI impact at +/-{DW:.0f} nm [transfer function, RABUS07 Eqs. 2.58-2.63]")
N_EFF = 2.44557
_clip = lambda g: min(max(g, a.gmin), a.gmax)


def _Ks(dw):
    """The three couplings after a width excursion dw, with delta_g = -dw."""
    return (K_of_gap(_clip(realised["bus <-> R1"]["gap"] - dw)),
            K_of_gap(_clip(realised["R1  <-> R2"]["gap"] - dw)),
            K_of_gap(_clip(realised["R2  <-> drop"]["gap"] - dw)))


base = kpi(realised["bus <-> R1"]["K"], realised["R1  <-> R2"]["K"],
           realised["R2  <-> drop"]["K"])
print(f"  nominal: rejection {base[0]:6.2f} dB   T_drop(s) {base[1]:.4f}")
print(f"\n  DIFFERENTIAL -- ring 2 alone off by dw (gaps follow -dw):")
for dw in (-DW, -1.0, -0.05, 0.05, 1.0, DW):
    f2 = 1 + DN_DW * dw / N_EFF
    K1n, K2n, K3n = _Ks(dw)
    Tp = T_drop(thp[0], thp[1] * f2, K1n, K2n, K3n)
    Ts = T_drop(ths[0], ths[1] * f2, K1n, K2n, K3n)
    print(f"    dw = {dw:+6.2f} nm: rejection {-10*np.log10(max(Tp,1e-300)):6.2f} dB"
          f"   T_drop(s) {Ts:.4f}  ({Ts/base[1]*100:5.1f} % of nominal)")
print(f"\n  COMMON MODE -- both rings, AFTER thermal retuning of the phase")
print(f"  (a heater restores the resonance; it cannot restore the coupling):")
for dw in (-DW, -DW / 2, DW / 2, DW):
    r, t, _ = kpi(*_Ks(dw))
    print(f"    dw = {dw:+6.1f} nm: rejection {r:6.2f} dB   T_drop(s) {t:.4f}"
          f"  ({t/base[1]*100:5.1f} % of nominal)")
print("""
Reading of the table
  COMMON mode (both rings see the same excursion): the resonance shift is large
  in linewidths but both combs move together, so a heater can follow it. The
  COUPLING change, however, is NOT correctable by a heater -- it is frozen at
  fabrication. That is a qualitatively different failure mode from the resonance
  shift, and it is the part that the delta_g = -delta_w coupling introduces.

  DIFFERENTIAL mode (the two rings differ): this is what destroys the Vernier
  coincidence, and the tolerance is set by the linewidth, not by the FSR.

  Orders of magnitude for this design: ONE nanometre of width detunes a resonance
  by ~97 linewidths, while the same excursion changes K1 by only ~1.8 %. The two
  effects are not comparable: the detuning is catastrophic and the coupling change
  is noise. That asymmetry is why the design needs per-ring thermal trimming and
  yet tolerates a coarse mask grid.
""")
print("Provenance of each number used above:")
print("  5 nm width fluctuation .... THOMSON16 (observed, 193 nm immersion DUV)")
print("  1 nm thickness ............ THOMSON16 (observed; not modelled here)")
print("  delta_g = -delta_w ........ CHROST15 sec 4.1; PRINZEN13 (3D FEM, SOI rib)")
print("  dn_eff/dw = 1.58e-3 /nm ... our own Femwell solves at 495 and 505 nm")
print("  min. feature 50 nm (EBL) .. MEDINA24 -- a RESOLUTION limit, NOT a tolerance")
print("  min. feature 100-120 nm ... MEDINA24 (DUV) -- likewise a resolution limit")
