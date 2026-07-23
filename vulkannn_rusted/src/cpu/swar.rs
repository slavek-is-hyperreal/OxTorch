//! SWAR (SIMD-within-a-register) primitives — verified core (Stage 1).
//!
//! GPR-only fallbacks for targets with 64-bit integer registers but no vector
//! unit (32-bit ARM w/o NEON, RISC-V w/o V, old MIPS). On x86_64/aarch64 these
//! NEVER win against SSE2/NEON and must stay behind a `not(vector)` gate.
//!
//! Every formula is transcribed from `docs/research/swar_catalog.md` (spot-check
//! confirmed 2026-07-23: musl HASZERO, Chessprogramming A1/A2, RTN Eq.8). Each
//! card's source is cited on its function. **Gate: every 8-bit-lane primitive is
//! tested EXHAUSTIVELY (all 2^16 (a,b) pairs per lane) vs a scalar reference,
//! plus u64 fuzz for lane independence — the test that would have caught the
//! legacy add_i8 carry-leak in one second.**

#![allow(dead_code)] // primitives are wired per-op incrementally

/// High-bit-of-each-byte mask.
pub const H: u64 = 0x8080_8080_8080_8080;
/// Low-bit-of-each-byte mask.
pub const L: u64 = 0x0101_0101_0101_0101;
/// Sign-bit-of-each-16-bit-lane mask (IEEE half / bf16).
pub const S16: u64 = 0x8000_8000_8000_8000;

// --- GROUP A: partitioned 8×byte arithmetic --------------------------------

/// A1 — wrapping add, carry-blocked. Catalog A1 (Chessprogramming/Warren §2-18).
/// `z = ((x & ~H) + (y & ~H)) ^ ((x ^ y) & H)`. Sign-agnostic (i8==u8 wrapping).
#[inline]
pub fn add_bytes(x: u64, y: u64) -> u64 {
    ((x & !H).wrapping_add(y & !H)) ^ ((x ^ y) & H)
}

/// A2 — wrapping sub, borrow-blocked. Catalog A2 (Chessprogramming/Warren §2-18).
/// `z = ((x | H) - (y & ~H)) ^ ((x ^ ~y) & H)`.
#[inline]
pub fn sub_bytes(x: u64, y: u64) -> u64 {
    ((x | H).wrapping_sub(y & !H)) ^ ((x ^ !y) & H)
}

/// A5 — negate = sub_bytes(0, x). Catalog A5. `neg(-128)=-128` (wraps).
#[inline]
pub fn neg_bytes(x: u64) -> u64 {
    sub_bytes(0, x)
}

/// Per-byte sign spread: 0xFF in bytes whose sign bit (bit7) is set, else 0x00.
/// Smear bit7 down within each byte (shifts stay lane-local). Warren §2-18 idea.
#[inline]
fn sign_spread(x: u64) -> u64 {
    let mut m = x & H;
    m |= m >> 1;
    m |= m >> 2;
    m |= m >> 4;
    m
}

/// A4 — per-byte signed absolute value. Catalog A4 (Warren §2-18/§2-4):
/// `abs = (x ^ m) - m` with `m` = per-byte sign spread, the subtract borrow-
/// blocked. `abs(-128) = -128` (wraps), matches `i8::wrapping_abs`.
#[inline]
pub fn abs_bytes(x: u64) -> u64 {
    let m = sign_spread(x);
    sub_bytes(x ^ m, m)
}

/// A3 — unsigned per-byte average, round-down. Catalog A3 (Chessprogramming/
/// Warren §2-5). `z = (x & y) + (((x ^ y) & ~L) >> 1)`. Unsigned only.
#[inline]
pub fn avg_u8(x: u64, y: u64) -> u64 {
    (x & y).wrapping_add(((x ^ y) & !L) >> 1)
}

// A7 (eq mask) / A8 (lt mask): DEFERRED — TODO, deliberately NOT shipped.
// First-attempt formulas (haszero-of-XOR + spread for A7; borrow-marker for A8)
// FAILED the exhaustive gate: musl HASZERO is a *detector* (its borrow cascades
// through a field of zero bytes), not a per-lane-exact mask, so an eq/lt mask
// built on it is wrong when several lanes match. A per-lane-exact byte compare
// needs a borrow-CONTAINED construction (each lane's borrow absorbed by a
// pre-set boundary bit, à la A2). Per protocol — a bit-trick that fails the
// exhaustive test does not ship "on faith" (cf. the legacy add_i8 carry-leak).
// To be revisited with a contained formula + exhaustive test (Stage 2).

// --- GROUP B: IEEE half sign-domain (4×u16) --------------------------------

/// B1 — negate 4 packed halves (f16/bf16): flip sign bit. Exact incl. NaN/±0.
#[inline]
pub fn neg_half(x: u64) -> u64 {
    x ^ S16
}

/// B2 — abs of 4 packed halves: clear sign bit. Exact incl. NaN.
#[inline]
pub fn abs_half(x: u64) -> u64 {
    x & !S16
}

/// B3 — copysign for 4 packed halves: magnitude from `mag`, sign from `sgn`.
#[inline]
pub fn copysign_half(mag: u64, sgn: u64) -> u64 {
    (mag & !S16) | (sgn & S16)
}

// --- GROUP C / D / E: popcount, horizontal sum, broadcast, scan ------------

/// C1 — parallel popcount of a u64 (Anderson; HAKMEM). Fallback when no POPCNT.
/// (On real targets prefer `u64::count_ones()`.)
#[inline]
pub fn popcount(mut x: u64) -> u32 {
    x -= (x >> 1) & 0x5555_5555_5555_5555;
    x = (x & 0x3333_3333_3333_3333) + ((x >> 2) & 0x3333_3333_3333_3333);
    x = (x + (x >> 4)) & 0x0f0f_0f0f_0f0f_0f0f;
    ((x.wrapping_mul(L)) >> 56) as u32
}

/// C2 — horizontal byte-sum via multiply by L. Valid ONLY if the total ≤ 255
/// (else the top byte overflows). Catalog C2 (Anderson/Lemire).
#[inline]
pub fn hsum_bytes(x: u64) -> u8 {
    (x.wrapping_mul(L) >> 56) as u8
}

/// D1 — broadcast a byte into all 8 lanes. Catalog D1 (Lemire).
#[inline]
pub fn broadcast_byte(b: u8) -> u64 {
    (b as u64).wrapping_mul(L)
}

/// E1 — musl HASZERO: verbatim `((x)-ONES & ~(x) & HIGHS)` (ONES=L, HIGHS=H).
/// **This is a DETECTOR, not a per-lane mask**: the result is nonzero iff `x`
/// contains at least one zero byte (musl's strlen use). A borrow from a zero
/// byte can cascade into higher bytes, so an individual lane's marker is NOT
/// guaranteed per-lane-exact when multiple low/zero bytes are present. Do not
/// build a per-lane equality mask on this (see the A7/A8 TODO above).
#[inline]
pub fn haszero(x: u64) -> u64 {
    x.wrapping_sub(L) & !x & H
}

/// `hasvalue(x, n)`: nonzero iff `x` has at least one byte equal to `n`. musl.
/// Detector semantics, as `haszero`.
#[inline]
pub fn hasvalue(x: u64, n: u8) -> u64 {
    haszero(x ^ broadcast_byte(n))
}

/// ReLU for 8 packed i8 lanes: max(x, 0). Composed from the sign-spread mask
/// (A4 component) + AND — zero out negative lanes, pass through non-negatives.
/// NOTE: like `x.max(0)`, this is `-0 -> 0`; for i8 there is no NaN, so unlike
/// the half-float relu-select (catalog B4) there is no NaN divergence here.
#[inline]
pub fn relu_i8_bytes(x: u64) -> u64 {
    x & !sign_spread(x)
}

// ---------------------------------------------------------------------------
// EXHAUSTIVE tests — the gate.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    /// Run `f` on every (a,b) byte pair placed in all 8 lanes, checking each
    /// output lane equals `refr(a,b)`. Then fuzz random u64 pairs for lane
    /// independence. 2^16 pairs — genuinely exhaustive for 8-bit lanes.
    fn exhaustive_bin(f: impl Fn(u64, u64) -> u64, refr: impl Fn(u8, u8) -> u8) {
        for a in 0u16..=255 {
            for b in 0u16..=255 {
                let (a, b) = (a as u8, b as u8);
                let x = broadcast_byte(a);
                let y = broadcast_byte(b);
                let got = f(x, y).to_le_bytes();
                let want = refr(a, b);
                for (lane, &g) in got.iter().enumerate() {
                    assert_eq!(g, want, "lane {lane}: op({a:#04x},{b:#04x}) = {g:#04x} != {want:#04x}");
                }
            }
        }
        // lane independence: mixed lanes must not interfere.
        let mut state = 0x1234_5678_9abc_def0u64;
        for _ in 0..200_000 {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17;
            let x = state;
            state ^= state << 13; state ^= state >> 7; state ^= state << 17;
            let y = state;
            let got = f(x, y).to_le_bytes();
            let xb = x.to_le_bytes();
            let yb = y.to_le_bytes();
            for i in 0..8 {
                assert_eq!(got[i], refr(xb[i], yb[i]), "fuzz lane {i}");
            }
        }
    }

    fn exhaustive_un(f: impl Fn(u64) -> u64, refr: impl Fn(u8) -> u8) {
        for a in 0u16..=255 {
            let a = a as u8;
            let got = f(broadcast_byte(a)).to_le_bytes();
            let want = refr(a);
            for (lane, &g) in got.iter().enumerate() {
                assert_eq!(g, want, "lane {lane}: op({a:#04x}) = {g:#04x} != {want:#04x}");
            }
        }
        let mut state = 0xdead_beef_cafe_babeu64;
        for _ in 0..200_000 {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17;
            let got = f(state).to_le_bytes();
            let sb = state.to_le_bytes();
            for i in 0..8 { assert_eq!(got[i], refr(sb[i]), "fuzz lane {i}"); }
        }
    }

    #[test] fn a1_add() { exhaustive_bin(add_bytes, |a, b| a.wrapping_add(b)); }
    #[test] fn a2_sub() { exhaustive_bin(sub_bytes, |a, b| a.wrapping_sub(b)); }
    #[test] fn a5_neg() { exhaustive_un(neg_bytes, |a| (a as i8).wrapping_neg() as u8); }
    #[test] fn a4_abs() { exhaustive_un(abs_bytes, |a| (a as i8).wrapping_abs() as u8); }
    #[test] fn a3_avg() { exhaustive_bin(avg_u8, |a, b| ((a as u16 + b as u16) >> 1) as u8); }
    // A7/A8 (eq/lt masks) intentionally not implemented — see the TODO in the
    // module body; their first formulas failed this exhaustive gate.
    #[test] fn relu_i8() { exhaustive_un(relu_i8_bytes, |a| (a as i8).max(0) as u8); }

    // B-family: exhaustive over all 2^16 half bit-patterns per lane.
    fn exhaustive_half_un(f: impl Fn(u64) -> u64, refr: impl Fn(u16) -> u16) {
        for p in 0u32..=0xFFFF {
            let p = p as u16;
            let x = (p as u64) * 0x0001_0001_0001_0001; // broadcast 16-bit lane
            let got = f(x);
            let lanes = [
                (got & 0xFFFF) as u16,
                ((got >> 16) & 0xFFFF) as u16,
                ((got >> 32) & 0xFFFF) as u16,
                ((got >> 48) & 0xFFFF) as u16,
            ];
            let want = refr(p);
            for (i, &g) in lanes.iter().enumerate() {
                assert_eq!(g, want, "half lane {i}: op({p:#06x}) = {g:#06x} != {want:#06x}");
            }
        }
    }
    #[test] fn b1_neg_half() { exhaustive_half_un(neg_half, |p| p ^ 0x8000); }
    #[test] fn b2_abs_half() { exhaustive_half_un(abs_half, |p| p & 0x7FFF); }
    #[test]
    fn b3_copysign_half() {
        for m in [0x0000u16, 0x3C00, 0xBC00, 0x7C00, 0xFC00, 0x7E00, 0x8000, 0x1234] {
            for s in [0x0000u16, 0x8000, 0x7FFF, 0xFFFF] {
                let mag = (m as u64) * 0x0001_0001_0001_0001;
                let sgn = (s as u64) * 0x0001_0001_0001_0001;
                let got = (copysign_half(mag, sgn) & 0xFFFF) as u16;
                let want = (m & 0x7FFF) | (s & 0x8000);
                assert_eq!(got, want);
            }
        }
    }

    #[test]
    fn c1_popcount() {
        for &v in &[0u64, u64::MAX, 1, 0x8000_0000_0000_0000, L, H, 0xdead_beef_cafe_babe] {
            assert_eq!(popcount(v), v.count_ones());
        }
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..1_000_000 {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            assert_eq!(popcount(s), s.count_ones());
        }
    }

    #[test]
    fn c2_hsum_and_d1() {
        // hsum valid when total <= 255
        for &bytes in &[[0u8; 8], [1, 2, 3, 4, 5, 6, 7, 8], [30, 30, 30, 30, 30, 30, 30, 15]] {
            let x = u64::from_le_bytes(bytes);
            let want: u16 = bytes.iter().map(|&b| b as u16).sum();
            assert!(want <= 255);
            assert_eq!(hsum_bytes(x) as u16, want);
        }
        for b in 0u16..=255 { assert_eq!(broadcast_byte(b as u8), (b as u64) * L); }
    }

    #[test]
    fn e1_haszero_hasvalue_detector() {
        // Detector semantics (musl): nonzero result iff >=1 zero byte present.
        for bytes in [
            [1u8, 2, 3, 4, 5, 6, 7, 8],   // no zero
            [1, 2, 0, 4, 5, 6, 7, 8],     // one zero
            [1, 1, 1, 1, 1, 1, 1, 1],     // no zero, all ones
            [0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80], // 0x80 must NOT false-positive
        ] {
            let x = u64::from_le_bytes(bytes);
            let has = bytes.iter().any(|&b| b == 0);
            assert_eq!(haszero(x) != 0, has, "haszero detector for {bytes:?}");
        }
        // hasvalue detector
        let x = u64::from_le_bytes([1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(hasvalue(x, 5) != 0);
        assert!(hasvalue(x, 9) == 0);
    }
}
