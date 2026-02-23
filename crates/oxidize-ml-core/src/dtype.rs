use std::fmt;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use serde::{Deserialize, Serialize};

/// Trait bound for numeric types usable in tensors.
/// Supports `f32` and `f64`.
pub trait Float:
    Copy
    + Clone
    + Default
    + PartialOrd
    + fmt::Debug
    + fmt::Display
    + Send
    + Sync
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + Sum
    + Serialize
    + for<'de> Deserialize<'de>
    + 'static
{
    const ZERO: Self;
    const ONE: Self;
    const TWO: Self;
    const HALF: Self;
    const NEG_ONE: Self;
    const EPSILON: Self;
    const PI: Self;
    const E: Self;
    const INFINITY: Self;
    const NEG_INFINITY: Self;

    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
    fn from_usize(v: usize) -> Self;

    fn abs(self) -> Self;
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn log2(self) -> Self;
    fn log10(self) -> Self;
    fn powf(self, n: Self) -> Self;
    fn powi(self, n: i32) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tanh(self) -> Self;
    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn round(self) -> Self;
    fn max(self, other: Self) -> Self;
    fn min(self, other: Self) -> Self;
    fn is_nan(self) -> bool;
    fn is_finite(self) -> bool;
    fn signum(self) -> Self;
    fn recip(self) -> Self;
}

impl Float for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;
    const HALF: Self = 0.5;
    const NEG_ONE: Self = -1.0;
    const EPSILON: Self = f32::EPSILON;
    const PI: Self = std::f32::consts::PI;
    const E: Self = std::f32::consts::E;
    const INFINITY: Self = f32::INFINITY;
    const NEG_INFINITY: Self = f32::NEG_INFINITY;

    #[inline] fn from_f64(v: f64) -> Self { v as f32 }
    #[inline] fn to_f64(self) -> f64 { self as f64 }
    #[inline] fn from_usize(v: usize) -> Self { v as f32 }
    #[inline] fn abs(self) -> Self { f32::abs(self) }
    #[inline] fn sqrt(self) -> Self { f32::sqrt(self) }
    #[inline] fn exp(self) -> Self { f32::exp(self) }
    #[inline] fn ln(self) -> Self { f32::ln(self) }
    #[inline] fn log2(self) -> Self { f32::log2(self) }
    #[inline] fn log10(self) -> Self { f32::log10(self) }
    #[inline] fn powf(self, n: Self) -> Self { f32::powf(self, n) }
    #[inline] fn powi(self, n: i32) -> Self { f32::powi(self, n) }
    #[inline] fn sin(self) -> Self { f32::sin(self) }
    #[inline] fn cos(self) -> Self { f32::cos(self) }
    #[inline] fn tanh(self) -> Self { f32::tanh(self) }
    #[inline] fn floor(self) -> Self { f32::floor(self) }
    #[inline] fn ceil(self) -> Self { f32::ceil(self) }
    #[inline] fn round(self) -> Self { f32::round(self) }
    #[inline] fn max(self, other: Self) -> Self { f32::max(self, other) }
    #[inline] fn min(self, other: Self) -> Self { f32::min(self, other) }
    #[inline] fn is_nan(self) -> bool { f32::is_nan(self) }
    #[inline] fn is_finite(self) -> bool { f32::is_finite(self) }
    #[inline] fn signum(self) -> Self { f32::signum(self) }
    #[inline] fn recip(self) -> Self { f32::recip(self) }
}

impl Float for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;
    const HALF: Self = 0.5;
    const NEG_ONE: Self = -1.0;
    const EPSILON: Self = f64::EPSILON;
    const PI: Self = std::f64::consts::PI;
    const E: Self = std::f64::consts::E;
    const INFINITY: Self = f64::INFINITY;
    const NEG_INFINITY: Self = f64::NEG_INFINITY;

    #[inline] fn from_f64(v: f64) -> Self { v }
    #[inline] fn to_f64(self) -> f64 { self }
    #[inline] fn from_usize(v: usize) -> Self { v as f64 }
    #[inline] fn abs(self) -> Self { f64::abs(self) }
    #[inline] fn sqrt(self) -> Self { f64::sqrt(self) }
    #[inline] fn exp(self) -> Self { f64::exp(self) }
    #[inline] fn ln(self) -> Self { f64::ln(self) }
    #[inline] fn log2(self) -> Self { f64::log2(self) }
    #[inline] fn log10(self) -> Self { f64::log10(self) }
    #[inline] fn powf(self, n: Self) -> Self { f64::powf(self, n) }
    #[inline] fn powi(self, n: i32) -> Self { f64::powi(self, n) }
    #[inline] fn sin(self) -> Self { f64::sin(self) }
    #[inline] fn cos(self) -> Self { f64::cos(self) }
    #[inline] fn tanh(self) -> Self { f64::tanh(self) }
    #[inline] fn floor(self) -> Self { f64::floor(self) }
    #[inline] fn ceil(self) -> Self { f64::ceil(self) }
    #[inline] fn round(self) -> Self { f64::round(self) }
    #[inline] fn max(self, other: Self) -> Self { f64::max(self, other) }
    #[inline] fn min(self, other: Self) -> Self { f64::min(self, other) }
    #[inline] fn is_nan(self) -> bool { f64::is_nan(self) }
    #[inline] fn is_finite(self) -> bool { f64::is_finite(self) }
    #[inline] fn signum(self) -> Self { f64::signum(self) }
    #[inline] fn recip(self) -> Self { f64::recip(self) }
}
