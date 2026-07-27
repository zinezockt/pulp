//! 32-bit ARM NEON feature tokens + safe intrinsic wrappers.
//!
//! Gated on pulp `nightly` + `target_arch = "arm"` because Rust's
//! `core::arch::arm` NEON surface (`stdarch_arm_neon_intrinsics`) and the
//! `neon` target feature are still unstable on the stable channel.
//!
//! Subset focused on ZIP/RLE-style byte reconstruct (OpenEXR log-depth
//! prefix sum): `vdupq_n_u8`, `vaddq_u8`, `vextq_u8`, lane extract, and a
//! software `vqtbl1q_u8` (AArch32 has no 128-bit `vqtbl1q`).

use super::arch;
use arch::*;

macro_rules! __impl {
	($name: ident, $feature: tt) => {
		#[derive(Clone, Copy)]
		#[repr(transparent)]
		pub struct $name {
			__private: (),
		}

		impl $name {
			/// # Safety
			/// Not checked
			#[inline(always)]
			pub const unsafe fn new_unchecked() -> Self {
				Self { __private: () }
			}

			#[inline(always)]
			pub fn try_new() -> Option<Self> {
				if feature_detected!($feature) {
					Some(Self { __private: () })
				} else {
					None
				}
			}

			#[inline(always)]
			pub fn is_available() -> bool {
				feature_detected!($feature)
			}
		}

		impl ::core::fmt::Debug for $name {
			#[inline]
			fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> core::fmt::Result {
				f.write_str(stringify!($name))
			}
		}
	};
}

__impl!(Neon, "neon");

impl Neon {
	delegate!({
		fn vdupq_n_u8(value: u8) -> uint8x16_t;
		fn vdupq_n_s8(value: i8) -> int8x16_t;
		fn vaddq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t;
		fn vaddq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t;
		fn vsubq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t;
		fn veorq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t;
		fn vandq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t;
		fn vorrq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t;
		fn vextq_u8<const N: i32>(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t;
		fn vgetq_lane_u8<const IMM5: i32>(v: uint8x16_t) -> u8;
		fn vsetq_lane_u8<const IMM5: i32>(a: u8, b: uint8x16_t) -> uint8x16_t;
	});

	/// AArch64-compatible `vqtbl1q_u8` for a 16-byte table.
	///
	/// AArch32 NEON only exposes 64-bit `vtbl`/`vtbx`. For a full 16×16 lookup
	/// we fall back to a small scalar loop (correct; reconstruct only uses an
	/// all-15 index, so the hot path is still the `vext`/`vadd` tree).
	#[inline(always)]
	pub fn vqtbl1q_u8(self, t: uint8x16_t, idx: uint8x16_t) -> uint8x16_t {
		// Safety: NEON token proves the feature; types are plain 128-bit registers.
		let table: [u8; 16] = unsafe { core::mem::transmute(t) };
		let indices: [u8; 16] = unsafe { core::mem::transmute(idx) };
		let mut out = [0u8; 16];
		for i in 0..16 {
			let j = indices[i];
			// AArch64 vqtbl1 zeros lanes with index >= 16.
			out[i] = if (j as usize) < 16 {
				table[j as usize]
			} else {
				0
			};
		}
		unsafe { core::mem::transmute(out) }
	}

	/// Broadcast byte lane `IMM5` of `v` to all 16 lanes (handy AArch32 stand-in
	/// for `vqtbl1q` with a constant index).
	#[inline(always)]
	pub fn vdupq_laneq_u8<const IMM5: i32>(self, v: uint8x16_t) -> uint8x16_t {
		self.vdupq_n_u8(self.vgetq_lane_u8::<IMM5>(v))
	}
}
