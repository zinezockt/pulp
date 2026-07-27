//! Low-level 32-bit ARM NEON API (pulp `nightly` feature).
//!
//! Rust's 32-bit ARM NEON intrinsics are still unstable; this module is only
//! available with `--features nightly` on a nightly toolchain. Mirrors the
//! `pulp::aarch64::Neon` token shape used by exrs ZIP/RLE reconstruct.
//!
//! High-level helpers return pulp [`u8x16`] (Pod) so downstream crates can stay
//! free of `core::arch::arm` / unstable features. Conversions use `transmute`
//! because `core::arch::arm::uint8x16_t` does not implement bytemuck `Pod`
//! (unlike the aarch64 equivalents pulp usually `cast!`s).

use super::*;

simd_type!({
	/// NEON token for 32-bit ARM (`target_feature = "neon"`).
	#[allow(missing_docs)]
	pub struct Neon {
		pub neon: f!("neon"),
	}
});

#[inline(always)]
fn to_arch(v: u8x16) -> core::arch::arm::uint8x16_t {
	// SAFETY: both are 16-byte plain bit patterns; u8x16 is Pod.
	unsafe { core::mem::transmute(v) }
}

#[inline(always)]
fn from_arch(v: core::arch::arm::uint8x16_t) -> u8x16 {
	// SAFETY: both are 16-byte plain bit patterns; u8x16 is Pod.
	unsafe { core::mem::transmute(v) }
}

impl Neon {
	/// `vdupq_n_u8`
	#[inline(always)]
	pub fn splat_u8x16(self, value: u8) -> u8x16 {
		from_arch(self.neon.vdupq_n_u8(value))
	}

	/// Wrapping byte add (`vaddq_u8`).
	#[inline(always)]
	pub fn wrapping_add_u8x16(self, a: u8x16, b: u8x16) -> u8x16 {
		from_arch(self.neon.vaddq_u8(to_arch(a), to_arch(b)))
	}

	/// Byte extract / shift used for log-depth prefix sum
	/// (`vextq_u8(a, b, N)` — same as AArch64 / OpenEXR reconstruct).
	#[inline(always)]
	pub fn ext_u8x16<const N: i32>(self, a: u8x16, b: u8x16) -> u8x16 {
		from_arch(self.neon.vextq_u8::<N>(to_arch(a), to_arch(b)))
	}

	/// Table lookup (`vqtbl1q_u8` semantics; software on AArch32).
	#[inline(always)]
	pub fn tbl_u8x16(self, table: u8x16, idx: u8x16) -> u8x16 {
		from_arch(self.neon.vqtbl1q_u8(to_arch(table), to_arch(idx)))
	}

	/// Broadcast lane `IMM5` of `v` to every lane.
	#[inline(always)]
	pub fn broadcast_lane_u8x16<const IMM5: i32>(self, v: u8x16) -> u8x16 {
		from_arch(self.neon.vdupq_laneq_u8::<IMM5>(to_arch(v)))
	}
}
