"""Cross-check runner: reads the runtime-agnostic case JSON (shared with the
cross-check suite at ../cross-check) and prices every case with quantjax,
emitting results JSON on stdout.

Usage: quantjax/.venv/bin/python -m quantjax.cli < cases.json [--backend cpu|rocm]
Models mirror quantcandle's src/bin/crosscheck.rs dispatch.
"""
import argparse
import json
import sys

import jax.numpy as jnp

from quantjax import (american_option, asian_option, bermudan_option,
                      binomial_tree_model, black_scholes_merton, bond_pricer,
                      cir, exotics, expected_shortfall, greeks, heston,
                      hull_white, sabr, vasicek, var)
from quantjax.heston import HestonParams
from quantjax.hull_white import HullWhiteModel
from quantjax.black_karasinski import BlackKarasinski


def _t(x):
    return jnp.array(x, dtype=jnp.float64)


def price_case(model, p):
    if model == "bsm":
        return float(black_scholes_merton.black_scholes_merton(
            p["option_type"], p["option_style"],
            float(p["spot"]), float(p["strike"]), float(p["expiry"]),
            float(p["volatility"]), float(p["rate"]), float(p["dividend"])))
    if model == "binomial":
        return float(binomial_tree_model.price_binomial_tree(
            p["option_type"], p["option_style"],
            float(p["spot"]), float(p["strike"]), float(p["expiry"]),
            float(p["volatility"]), float(p["rate"]), int(p["steps"]), None))
    if model == "heston_cf":
        h = heston.HestonParams(
            kappa=p["kappa"], theta=p["theta"], sigma_v=p["sigma_v"],
            rho=p["rho"], v0=p["v0"], mu=p.get("mu", 0.0))
        return float(heston.characteristic_function_price(
            h, p["spot"], p["strike"], p["expiry"], p["rate"],
            p["option_type"], p["dividend"],
            p.get("integration_limit", 100.0),
            int(p.get("integration_points", 2000))))
    if model == "heston_mc":
        h = HestonParams(
            kappa=p["kappa"], theta=p["theta"], sigma_v=p["sigma_v"],
            rho=p["rho"], v0=p["v0"], mu=p.get("mu", 0.0))
        return float(heston.monte_carlo_option_price(
            h, p["spot"], p["strike"], p["expiry"], p["rate"],
            p["option_type"], p["dividend"],
            int(p["n_paths"]), int(p["steps"]), int(p.get("seed", 2007))))
    if model == "vasicek_zcb":
        return float(vasicek.zero_coupon_bond_price(
            p["kappa"], p["theta"], p["sigma"], p["r0"], p["maturity"]))
    if model == "cir_zcb":
        return float(cir.zero_coupon_bond_price(
            p["kappa"], p["theta"], p["sigma"], p["r0"], p["maturity"]))
    if model == "hull_white_zcb":
        hw = HullWhiteModel(a=p["a"], sigma=p["sigma"], r0=p["r0"])
        return float(hw.zero_coupon_bond_price(p["maturity"], None))
    if model == "black_karasinski_zcb":
        bk = BlackKarasinski(a=p["a"], sigma=p["sigma"], r0=p["r0"])
        return float(bk.zero_coupon_bond_price(p["maturity"], None, 100))
    if model == "american_option":
        return float(american_option.american_option(
            p["option_type"], p["spot"], p["strike"], p["expiry"],
            p["volatility"], p["rate"], p["steps"]))
    if model == "bermudan_option":
        return float(bermudan_option.bermudan_option(
            p["option_type"], p["spot"], p["strike"], p["expiry"],
            p["volatility"], p["rate"], p["steps"], p.get("exercise_dates") or []))
    if model == "asian_option":
        return float(asian_option.asian_option(
            p["option_type"], p["spot"], p["strike"], p["expiry"],
            p["volatility"], p["rate"], p["steps"],
            average_type=p.get("average_type", "arithmetic"),
            num_paths=p.get("num_paths"), seed=int(p.get("seed", 2007))))
    if model == "black76":
        return float(sabr.black76_price(
            p["option_type"], p["forward"], p["strike"], p["expiry"],
            p["volatility"], p["rate"]))
    if model == "sabr_iv":
        sp = sabr.SabrParams(
            alpha=p["alpha"], beta=p["beta"], rho=p["rho"], nu=p["nu"],
            f0=p["f0"], shift=p.get("shift", 0.0))
        return float(sabr.hagan_lognormal_implied_volatility(
            sp, p["strike"], p["expiry"],
            p.get("forward"), p.get("shift")))
    if model == "sabr_price":
        sp = sabr.SabrParams(
            alpha=p["alpha"], beta=p["beta"], rho=p["rho"], nu=p["nu"],
            f0=p["f0"], shift=p.get("shift", 0.0))
        return float(sabr.option_price(
            sp, p["strike"], p["expiry"], p["rate"], p["option_type"],
            p.get("dividend", 0.0), p.get("model", "lognormal"), p.get("shift")))
    if model == "digital_option":
        return float(exotics.digital_option(
            p["option_type"], p["spot"], p["strike"], p["expiry"],
            p["volatility"], p["rate"], p["payout"],
            int(p["n_paths"]), int(p.get("seed", 2007))))
    if model == "barrier_option":
        return float(exotics.barrier_option(
            p["option_type"], p["barrier_type"], p["spot"], p["strike"],
            p["barrier"], p["expiry"], p["volatility"], p["rate"],
            int(p["n_paths"]), int(p["steps"]), int(p.get("seed", 2007))))
    if model == "lookback_option":
        return float(exotics.lookback_option(
            p["option_type"], p["spot"], p["strike"], p["expiry"],
            p["volatility"], p["rate"], int(p["n_paths"]),
            p.get("strike_type", "fixed"), int(p.get("seed", 2007))))
    if model == "chooser_option":
        return float(exotics.chooser_option(
            p["spot"], p["strike"], p["expiry"], p["volatility"],
            p["rate"], p["choose_time"]))
    if model == "bond_zero_coupon":
        return float(bond_pricer.zero_coupon_bond(
            p["face_value"], p["rate"], p["maturity"]))
    if model == "bond_coupon":
        return float(bond_pricer.coupon_bond(
            p["face_value"], p["coupon_rate"], p["rate"], int(p["periods"])))
    if model == "bond_callable":
        return float(bond_pricer.callable_bond(
            p["face_value"], p["coupon_rate"], p["rate"], int(p["periods"]),
            p["call_price"], int(p["call_period"])))
    if model == "bond_putable":
        return float(bond_pricer.putable_bond(
            p["face_value"], p["coupon_rate"], p["rate"], int(p["periods"]),
            p["put_price"], int(p["put_period"])))
    if model == "greeks_european":
        g = greeks.european_option_greeks(
            p["spot"], p["strike"], p["expiry"], p["rate"], p["volatility"],
            int(p.get("n_paths", 65536)), int(p.get("seed", 2007)))
        return {"delta": g.delta, "gamma": g.gamma, "vega": g.vega,
                "theta": g.theta, "rho": g.rho}
    if model == "var":
        return float(var.calculate_var(list(map(float, p["returns"])),
                                       float(p["confidence_level"])))
    if model == "es":
        return float(expected_shortfall.calculate_es(
            list(map(float, p["returns"])), float(p["confidence_level"])))
    raise ValueError(f"unknown model '{model}'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--cases", default=None, help="path to cases JSON (default stdin)")
    args = ap.parse_args()

    if args.cases:
        with open(args.cases) as f:
            cases = json.load(f)
    else:
        cases = json.load(sys.stdin)
    results = []
    for c in cases["cases"]:
        entry = {"id": c["id"], "ok": False}
        try:
            out = price_case(c["model"], c["params"])
            if isinstance(out, dict):
                entry.update(ok=True, values=out)
            else:
                entry.update(ok=True, price=float(out))
        except Exception as e:
            entry.update(ok=False, error=f"{type(e).__name__}: {e}")
        results.append(entry)
    print(json.dumps({"backend": args.backend, "results": results}, indent=2))


if __name__ == "__main__":
    main()
