//! Low-level 32-bit ARM NEON API (pulp `nightly` feature).
//!
//! Rust's 32-bit ARM NEON intrinsics are still unstable; this module is only
//! available with `--features nightly` on a nightly toolchain. Mirrors the
//! `pulp::aarch64::Neon` token shape (`simd.neon.<intrinsic>(...)`) used by
//! exrs ZIP/RLE reconstruct.

use super::*;

simd_type!({
	/// NEON token for 32-bit ARM (`target_feature = "neon"`).
	#[allow(missing_docs)]
	pub struct Neon {
		pub neon: f!("neon"),
	}
});
