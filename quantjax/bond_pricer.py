"""Port of quantburn/src/bond_pricer.rs (originally
torchquantlib/core/asset_pricing/bond/bond_pricer.py).

Closed-form and lattice bond pricing: zero-coupon, coupon bonds,
callable/putable bonds on a short-rate lattice, convertible bonds on a
stock binomial tree (with call/put exercise schedules), and a
stochastic-rate coupon bond. All math is scalar Python floats (pure
lattice / closed-form, no tensors needed).
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

EPSILON = 1e-8


def _period_count(periods: int) -> int:
    return max(int(periods), 1)


def _short_rate_tree(initial_rate: float, count: int, rate_volatility: float) -> List[List[float]]:
    sigma = max(rate_volatility, 0.0)
    if sigma == 0.0:
        return [[initial_rate] * (n + 1) for n in range(count)]
    up = math.exp(sigma)
    tree = []
    for n in range(count):
        exponents = range(-n, n + 1, 2)
        tree.append([max(initial_rate * up ** e, 0.0) for e in exponents])
    return tree


def _embedded_option_bond_lattice(
    face_value: float,
    coupon_rate: float,
    rate: float,
    periods: int,
    exercise_price: Optional[float],
    exercise_period: Optional[int],
    exercise_kind: Optional[str],
    rate_volatility: Optional[float],
) -> float:
    n = _period_count(periods)
    rate_tree = _short_rate_tree(rate, n, rate_volatility if rate_volatility is not None else 0.0)
    coupon = face_value * coupon_rate
    continuation = [face_value] * (n + 1)

    exercise_index = None
    if exercise_period is not None:
        exercise_index = min(max(int(exercise_period), 1), n)

    for step in range(n - 1, -1, -1):
        expected_next = [0.5 * (continuation[i] + continuation[i + 1]) for i in range(len(continuation) - 1)]
        node_values = [
            (coupon + v) / max(1.0 + r, EPSILON)
            for v, r in zip(expected_next, rate_tree[step])
        ]
        if exercise_index is not None and exercise_price is not None and step + 1 == exercise_index:
            if exercise_kind == "call":
                node_values = [min(v, exercise_price) for v in node_values]
            elif exercise_kind == "put":
                node_values = [max(v, exercise_price) for v in node_values]
        continuation = node_values
    return continuation[0]


def zero_coupon_bond(face_value: float, rate: float, maturity: float) -> float:
    return face_value / max((1.0 + rate) ** maturity, EPSILON)


def coupon_bond(face_value: float, coupon_rate: float, rate: float, periods: int) -> float:
    n = _period_count(periods)
    coupon_cashflow = face_value * coupon_rate
    total = 0.0
    for t in range(1, n + 1):
        total += coupon_cashflow / max((1.0 + rate) ** t, EPSILON)
    return total + face_value / max((1.0 + rate) ** n, EPSILON)


def callable_bond_lattice(
    face_value: float,
    coupon_rate: float,
    rate: float,
    periods: int,
    call_price: float,
    call_period: int,
    rate_volatility: Optional[float] = None,
) -> float:
    lattice_value = _embedded_option_bond_lattice(
        face_value, coupon_rate, rate, periods,
        call_price, call_period, "call", rate_volatility,
    )
    return min(lattice_value, coupon_bond(face_value, coupon_rate, rate, periods))


def callable_bond(
    face_value: float,
    coupon_rate: float,
    rate: float,
    periods: int,
    call_price: float,
    call_period: int,
) -> float:
    return callable_bond_lattice(face_value, coupon_rate, rate, periods, call_price, call_period, None)


def putable_bond_lattice(
    face_value: float,
    coupon_rate: float,
    rate: float,
    periods: int,
    put_price: float,
    put_period: int,
    rate_volatility: Optional[float] = None,
) -> float:
    lattice_value = _embedded_option_bond_lattice(
        face_value, coupon_rate, rate, periods,
        put_price, put_period, "put", rate_volatility,
    )
    return max(lattice_value, coupon_bond(face_value, coupon_rate, rate, periods))


def putable_bond(
    face_value: float,
    coupon_rate: float,
    rate: float,
    periods: int,
    put_price: float,
    put_period: int,
) -> float:
    return putable_bond_lattice(face_value, coupon_rate, rate, periods, put_price, put_period, None)


def convertible_bond(
    face_value: float,
    coupon_rate: float,
    rate: float,
    periods: int,
    conversion_ratio: float,
    conversion_price: float,
) -> float:
    straight_value = coupon_bond(face_value, coupon_rate, rate, periods)
    conversion_value = conversion_ratio * conversion_price
    return max(straight_value, conversion_value)


Schedule = Dict[int, float]


def _normalize_schedule(
    period_count: int,
    schedule: Optional[Sequence[Tuple[int, float]]],
    single_price: Optional[float],
    single_period: Optional[int],
) -> Schedule:
    out: Schedule = {}
    if schedule is not None:
        for p, price in schedule:
            idx = min(max(int(p), 1), period_count)
            out[idx] = price
    if single_price is not None and single_period is not None:
        idx = min(max(int(single_period), 1), period_count)
        out[idx] = single_price
    return out


def convertible_bond_lattice(
    face_value: float,
    coupon_rate: float,
    rate: float,
    periods: int,
    conversion_ratio: float,
    stock_price: float,
    stock_volatility: float,
    dividend_yield: Optional[float] = None,
    call_price: Optional[float] = None,
    call_period: Optional[int] = None,
    put_price: Optional[float] = None,
    put_period: Optional[int] = None,
    call_schedule: Optional[Sequence[Tuple[int, float]]] = None,
    put_schedule: Optional[Sequence[Tuple[int, float]]] = None,
) -> float:
    n = _period_count(periods)
    coupon = face_value * coupon_rate
    stock0 = stock_price
    volatility = max(stock_volatility, 0.0)
    dividend = dividend_yield if dividend_yield is not None else 0.0
    dt = 1.0

    call_schedule_map = _normalize_schedule(n, call_schedule, call_price, call_period)
    put_schedule_map = _normalize_schedule(n, put_schedule, put_price, put_period)

    if volatility == 0.0:
        growth = math.exp((rate - dividend) * dt)
        terminal_stock = stock0 * growth ** n
        continuation = [max(face_value, conversion_ratio * terminal_stock)] * (n + 1)
        if n in call_schedule_map:
            p = call_schedule_map[n]
            continuation = [min(v, p) for v in continuation]
        if n in put_schedule_map:
            p = put_schedule_map[n]
            continuation = [max(v, p) for v in continuation]
        for step in range(n - 1, -1, -1):
            continuation = [
                (coupon + continuation[i]) / max(1.0 + rate, EPSILON)
                for i in range(len(continuation) - 1)
            ]
            exercise_date = step + 1
            if exercise_date in call_schedule_map:
                p = call_schedule_map[exercise_date]
                continuation = [min(v, p) for v in continuation]
            if exercise_date in put_schedule_map:
                p = put_schedule_map[exercise_date]
                continuation = [max(v, p) for v in continuation]
        return continuation[0]

    up = math.exp(volatility * math.sqrt(dt))
    down = 1.0 / up
    growth = math.exp((rate - dividend) * dt)
    probability = min(max((growth - down) / max(up - down, EPSILON), 0.0), 1.0)

    def stock_nodes(step: int) -> List[float]:
        return [stock0 * up ** (step - j) * down ** j for j in range(step + 1)]

    continuation = [max(face_value, conversion_ratio * s) for s in stock_nodes(n)]
    if n in call_schedule_map:
        p = call_schedule_map[n]
        continuation = [min(v, p) for v in continuation]
    if n in put_schedule_map:
        p = put_schedule_map[n]
        continuation = [max(v, p) for v in continuation]

    for step in range(n - 1, -1, -1):
        nodes = stock_nodes(step)
        expected_next = [
            probability * continuation[i] + (1.0 - probability) * continuation[i + 1]
            for i in range(len(continuation) - 1)
        ]
        node_value = [
            max((coupon + h) / max(1.0 + rate, EPSILON), conversion_ratio * s)
            for h, s in zip(expected_next, nodes)
        ]
        exercise_date = step + 1
        if exercise_date in call_schedule_map:
            p = call_schedule_map[exercise_date]
            node_value = [min(v, p) for v in node_value]
        if exercise_date in put_schedule_map:
            p = put_schedule_map[exercise_date]
            node_value = [max(v, p) for v in node_value]
        continuation = node_value

    return continuation[0]


def callable_convertible_bond_lattice(
    face_value: float,
    coupon_rate: float,
    rate: float,
    periods: int,
    conversion_ratio: float,
    stock_price: float,
    stock_volatility: float,
    call_price: float,
    call_period: int,
    dividend_yield: Optional[float] = None,
    call_schedule: Optional[Sequence[Tuple[int, float]]] = None,
) -> float:
    return convertible_bond_lattice(
        face_value, coupon_rate, rate, periods,
        conversion_ratio, stock_price, stock_volatility,
        dividend_yield, call_price, call_period, None, None,
        call_schedule, None,
    )


def putable_convertible_bond_lattice(
    face_value: float,
    coupon_rate: float,
    rate: float,
    periods: int,
    conversion_ratio: float,
    stock_price: float,
    stock_volatility: float,
    put_price: float,
    put_period: int,
    dividend_yield: Optional[float] = None,
    put_schedule: Optional[Sequence[Tuple[int, float]]] = None,
) -> float:
    return convertible_bond_lattice(
        face_value, coupon_rate, rate, periods,
        conversion_ratio, stock_price, stock_volatility,
        dividend_yield, None, None, put_price, put_period,
        None, put_schedule,
    )


def callable_putable_convertible_bond_lattice(
    face_value: float,
    coupon_rate: float,
    rate: float,
    periods: int,
    conversion_ratio: float,
    stock_price: float,
    stock_volatility: float,
    call_price: float,
    call_period: int,
    put_price: float,
    put_period: int,
    dividend_yield: Optional[float] = None,
    call_schedule: Optional[Sequence[Tuple[int, float]]] = None,
    put_schedule: Optional[Sequence[Tuple[int, float]]] = None,
) -> float:
    return convertible_bond_lattice(
        face_value, coupon_rate, rate, periods,
        conversion_ratio, stock_price, stock_volatility,
        dividend_yield, call_price, call_period, put_price, put_period,
        call_schedule, put_schedule,
    )


def stochastic_rate_bond(face_value: float, coupon_rate: float, rates: Sequence[float]) -> float:
    """`rates` is the per-period short rate path `rate[:period_count]`."""
    coupon = face_value * coupon_rate
    discount = 1.0
    total = 0.0
    for r in rates:
        discount *= 1.0 + r
        total += coupon / discount
    return total + face_value / discount


# ---------------------------------------------------------------------------
# Tests (expected values from quantburn's scripts/gen_bond_pricer.py, float64)
# ---------------------------------------------------------------------------

def _assert_close(actual: float, expected: float, tol: float = 1e-8) -> None:
    diff = abs(actual - expected)
    assert diff <= max(tol, tol * abs(expected)), f"actual {actual} vs expected {expected} (diff {diff})"


def test_zero_coupon_bond():
    _assert_close(zero_coupon_bond(1000.0, 0.05, 3.0), 863.837598531476)


def test_coupon_bond():
    _assert_close(coupon_bond(1000.0, 0.05, 0.04, 5), 1044.5182233101618)


def test_callable_bond_lattice():
    _assert_close(
        callable_bond_lattice(1000.0, 0.06, 0.05, 5, 1050.0, 3, 0.1),
        1041.0660615540885,
    )
    _assert_close(
        callable_bond(1000.0, 0.06, 0.05, 5, 1050.0, 3),
        callable_bond_lattice(1000.0, 0.06, 0.05, 5, 1050.0, 3, None),
    )


def test_putable_bond_lattice():
    _assert_close(
        putable_bond_lattice(1000.0, 0.06, 0.05, 5, 950.0, 3, 0.1),
        1043.2947667063079,
    )


def test_convertible_bond():
    _assert_close(convertible_bond(100.0, 0.05, 0.05, 4, 2.0, 60.0), 120.0)


def test_convertible_bond_lattice_with_vol():
    calls = [(3, 110.0)]
    puts = [(2, 90.0)]
    _assert_close(
        convertible_bond_lattice(
            100.0, 0.05, 0.05, 4, 2.0, 60.0, 0.2, 0.01, 110.0, 3,
            90.0, 2, calls, puts,
        ),
        127.7103299966812,
    )


def test_convertible_bond_lattice_zero_vol():
    calls = [(3, 110.0)]
    puts = [(2, 90.0)]
    _assert_close(
        convertible_bond_lattice(
            100.0, 0.05, 0.05, 4, 2.0, 60.0, 0.0, 0.01, 110.0, 3,
            90.0, 2, calls, puts,
        ),
        109.07029478458048,
    )


def test_stochastic_rate_bond():
    _assert_close(
        stochastic_rate_bond(1000.0, 0.04, [0.03, 0.035, 0.04, 0.045]),
        1010.083333337649,
    )
