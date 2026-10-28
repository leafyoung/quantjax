"""Deterministic normal sampler, bit-identical to quantburn's Xorshift64Star
(and to quantburn's scripts/refgen.py reference harness), so all four
libraries (quantorch via reference harness, quantburn, quantcandle, quantjax)
draw identical standard normals for the same seed."""

_M64 = (1 << 64) - 1
_MULT = 0x2545F4914F6CDD1D


class Xorshift64Star:
    def __init__(self, seed: int):
        self.state = max(int(seed), 1) & _M64

    def next_u64(self) -> int:
        x = self.state
        x ^= x >> 12
        x ^= (x << 25) & _M64
        x ^= x >> 27
        self.state = x & _M64
        return (x * _MULT) & _M64

    def next_f64(self) -> float:
        # Uniform in (0, 1), same bit recipe as the Rust sampler.
        return (self.next_u64() >> 11) / float(1 << 53)

    def next_std_normal(self) -> float:
        import math

        u1 = max(self.next_f64(), 1e-300)
        u2 = self.next_f64()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def normals(self, n: int) -> list:
        return [self.next_std_normal() for _ in range(n)]


def test_matches_rust_reference():
    # quantburn/src/rng.rs::matches_python_reference (seed 42)
    r = Xorshift64Star(42)
    assert abs(r.next_std_normal() - 0.2960330150871692) < 1e-12
    assert abs(r.next_std_normal() - 0.6443779794453868) < 1e-12
