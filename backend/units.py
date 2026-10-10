"""The only place in the codebase where units are converted.

The API speaks laboratory units (kV, pF, uF, uH, ohm, us). The physics engine will
speak SI (V, F, H, ohm, s). Mixing the two is the easiest way to get a plausible but
wrong answer out of an impulse calculation, so every conversion goes through a named
function here and the physics module is expected to accept SI only.
"""

from __future__ import annotations

KV_PER_V = 1e-3
V_PER_KV = 1e3
F_PER_PF = 1e-12
PF_PER_F = 1e12
F_PER_UF = 1e-6
UF_PER_F = 1e6
H_PER_UH = 1e-6
UH_PER_H = 1e6
S_PER_US = 1e-6
US_PER_S = 1e6


def kv_to_v(kv: float) -> float:
    return kv * V_PER_KV


def v_to_kv(volts: float) -> float:
    return volts * KV_PER_V


def pf_to_f(pf: float) -> float:
    return pf * F_PER_PF


def f_to_pf(farads: float) -> float:
    return farads * PF_PER_F


def uf_to_f(uf: float) -> float:
    return uf * F_PER_UF


def f_to_uf(farads: float) -> float:
    return farads * UF_PER_F


def uh_to_h(uh: float) -> float:
    return uh * H_PER_UH


def h_to_uh(henries: float) -> float:
    return henries * UH_PER_H


def us_to_s(us: float) -> float:
    return us * S_PER_US


def s_to_us(seconds: float) -> float:
    return seconds * US_PER_S


def pf_to_uf(pf: float) -> float:
    return pf * 1e-6
