//! BLAS FFI bindings for hardware-accelerated linear algebra.
//!
//! On macOS, links to Apple's Accelerate framework which routes matrix
//! operations through the AMX (Apple Matrix Extensions) coprocessor
//! on M-series chips. This provides 100-1000x speedup over naive loops.
//!
//! On WASM, falls back to a pure-Rust naive implementation.

#[cfg(all(target_os = "macos", not(target_arch = "wasm32")))]
extern crate accelerate_src;

// ─── Native BLAS (non-WASM) ────────────────────────────────────────────────

#[cfg(not(target_arch = "wasm32"))]
pub const CBLAS_ROW_MAJOR: i32 = 101;
#[cfg(not(target_arch = "wasm32"))]
pub const CBLAS_NO_TRANS: i32 = 111;
#[cfg(not(target_arch = "wasm32"))]
pub const CBLAS_TRANS: i32 = 112;

#[cfg(not(target_arch = "wasm32"))]
extern "C" {
    pub fn cblas_dgemm(
        order: i32, trans_a: i32, trans_b: i32,
        m: i32, n: i32, k: i32,
        alpha: f64, a: *const f64, lda: i32,
        b: *const f64, ldb: i32,
        beta: f64, c: *mut f64, ldc: i32,
    );

    pub fn cblas_sgemm(
        order: i32, trans_a: i32, trans_b: i32,
        m: i32, n: i32, k: i32,
        alpha: f32, a: *const f32, lda: i32,
        b: *const f32, ldb: i32,
        beta: f32, c: *mut f32, ldc: i32,
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[inline]
pub fn blas_matmul_f64(
    a: &[f64], b: &[f64], c: &mut [f64],
    m: usize, n: usize, k: usize,
) {
    unsafe {
        cblas_dgemm(
            CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
            m as i32, n as i32, k as i32,
            1.0, a.as_ptr(), k as i32,
            b.as_ptr(), n as i32,
            0.0, c.as_mut_ptr(), n as i32,
        );
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[inline]
pub fn blas_matmul_f32(
    a: &[f32], b: &[f32], c: &mut [f32],
    m: usize, n: usize, k: usize,
) {
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
            m as i32, n as i32, k as i32,
            1.0f32, a.as_ptr(), k as i32,
            b.as_ptr(), n as i32,
            0.0f32, c.as_mut_ptr(), n as i32,
        );
    }
}

// ─── Pure-Rust Fallback (WASM) ─────────────────────────────────────────────

#[cfg(target_arch = "wasm32")]
#[inline]
pub fn blas_matmul_f64(
    a: &[f64], b: &[f64], c: &mut [f64],
    m: usize, n: usize, k: usize,
) {
    naive_matmul(a, b, c, m, n, k);
}

#[cfg(target_arch = "wasm32")]
#[inline]
pub fn blas_matmul_f32(
    a: &[f32], b: &[f32], c: &mut [f32],
    m: usize, n: usize, k: usize,
) {
    naive_matmul(a, b, c, m, n, k);
}

/// Naive O(n³) matrix multiply for WASM targets.
/// Not fast, but correct and dependency-free.
#[cfg(target_arch = "wasm32")]
fn naive_matmul<T: Copy + std::ops::Mul<Output = T> + std::ops::AddAssign + Default>(
    a: &[T], b: &[T], c: &mut [T],
    m: usize, n: usize, k: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = T::default();
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}
