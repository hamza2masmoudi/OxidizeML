use oxidize_ml_core::{Float, Tensor, TensorError};
use oxidize_ml_core::error::TensorResult;

/// Singular Value Decomposition (SVD) using one-sided Jacobi rotations.
///
/// Decomposes A = U Σ Vᵀ where:
/// - U: [m, k] left singular vectors
/// - Σ: [k] singular values (descending)
/// - V: [n, k] right singular vectors
/// k = min(m, n)
pub fn svd<T: Float>(a: &Tensor<T>) -> TensorResult<(Tensor<T>, Tensor<T>, Tensor<T>)> {
    let m = a.shape().dim(0)?;
    let n = a.shape().dim(1)?;
    let k = m.min(n);

    // Work in f64 for numerical stability
    let mut ata = vec![0.0f64; n * n]; // AᵀA
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for r in 0..m {
                sum += a.get(&[r, i])?.to_f64() * a.get(&[r, j])?.to_f64();
            }
            ata[i * n + j] = sum;
        }
    }

    // Eigendecomposition of AᵀA via Jacobi eigenvalue algorithm
    let mut eigvecs = vec![0.0f64; n * n];
    for i in 0..n { eigvecs[i * n + i] = 1.0; } // Identity

    for _ in 0..100 {
        let mut max_off = 0.0;
        let mut pi = 0;
        let mut pj = 1;

        // Find largest off-diagonal element
        for i in 0..n {
            for j in i + 1..n {
                if ata[i * n + j].abs() > max_off {
                    max_off = ata[i * n + j].abs();
                    pi = i;
                    pj = j;
                }
            }
        }

        if max_off < 1e-12 { break; }

        // Compute Jacobi rotation
        let aij = ata[pi * n + pj];
        let aii = ata[pi * n + pi];
        let ajj = ata[pj * n + pj];

        let theta = if (aii - ajj).abs() < 1e-15 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * aij / (aii - ajj)).atan()
        };
        let c = theta.cos();
        let s = theta.sin();

        // Apply rotation to AᵀA: rows and columns pi, pj
        let mut new_ata = ata.clone();
        for l in 0..n {
            if l == pi || l == pj { continue; }
            new_ata[pi * n + l] = c * ata[pi * n + l] + s * ata[pj * n + l];
            new_ata[l * n + pi] = new_ata[pi * n + l];
            new_ata[pj * n + l] = -s * ata[pi * n + l] + c * ata[pj * n + l];
            new_ata[l * n + pj] = new_ata[pj * n + l];
        }
        new_ata[pi * n + pi] = c * c * aii + 2.0 * c * s * aij + s * s * ajj;
        new_ata[pj * n + pj] = s * s * aii - 2.0 * c * s * aij + c * c * ajj;
        new_ata[pi * n + pj] = 0.0;
        new_ata[pj * n + pi] = 0.0;
        ata = new_ata;

        // Accumulate eigenvectors
        for l in 0..n {
            let vli = eigvecs[l * n + pi];
            let vlj = eigvecs[l * n + pj];
            eigvecs[l * n + pi] = c * vli + s * vlj;
            eigvecs[l * n + pj] = -s * vli + c * vlj;
        }
    }

    // Collect eigenvalues and sort descending
    let mut eig_pairs: Vec<(f64, usize)> = (0..n).map(|i| (ata[i * n + i].max(0.0), i)).collect();
    eig_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    // Singular values = sqrt(eigenvalues)
    let sigma_data: Vec<T> = eig_pairs[..k].iter()
        .map(|(ev, _)| T::from_f64(ev.sqrt()))
        .collect();
    let sigma = Tensor::new(sigma_data, vec![k])?;

    // V = eigenvectors (columns, reordered)
    let mut v_data = Vec::with_capacity(n * k);
    for i in 0..n {
        for &(_, idx) in &eig_pairs[..k] {
            v_data.push(T::from_f64(eigvecs[i * n + idx]));
        }
    }
    let v = Tensor::new(v_data, vec![n, k])?;

    // U = A @ V @ Σ⁻¹
    let av = a.matmul(&v)?;
    let mut u_data = Vec::with_capacity(m * k);
    for i in 0..m {
        for j in 0..k {
            let s_val = sigma.data()[j].to_f64();
            if s_val.abs() > 1e-10 {
                u_data.push(T::from_f64(av.get(&[i, j])?.to_f64() / s_val));
            } else {
                u_data.push(T::ZERO);
            }
        }
    }
    let u = Tensor::new(u_data, vec![m, k])?;

    Ok((u, sigma, v))
}

/// Compute matrix norm (Frobenius by default).
pub fn frobenius_norm<T: Float>(a: &Tensor<T>) -> f64 {
    a.data().iter().map(|&v| v.to_f64() * v.to_f64()).sum::<f64>().sqrt()
}

/// Compute condition number using SVD: σ_max / σ_min.
pub fn condition_number<T: Float>(a: &Tensor<T>) -> TensorResult<f64> {
    let (_, sigma, _) = svd(a)?;
    let s = sigma.data();
    let max = s.first().map(|v| v.to_f64()).unwrap_or(0.0);
    let min = s.last().map(|v| v.to_f64()).unwrap_or(1e-15);
    Ok(max / min.max(1e-15))
}

/// Matrix rank via SVD (count singular values above threshold).
pub fn matrix_rank<T: Float>(a: &Tensor<T>, tol: f64) -> TensorResult<usize> {
    let (_, sigma, _) = svd(a)?;
    Ok(sigma.data().iter().filter(|&&v| v.to_f64() > tol).count())
}

/// Pseudo-inverse via SVD: A⁺ = V Σ⁺ Uᵀ
pub fn pinv<T: Float>(a: &Tensor<T>) -> TensorResult<Tensor<T>> {
    let (u, sigma, v) = svd(a)?;
    let k = sigma.numel();
    let m = u.shape().dim(0)?;
    let n = v.shape().dim(0)?;

    // Σ⁺ = diag(1/σ_i for σ_i > tol)
    let tol = 1e-10;
    let mut sigma_inv = vec![T::ZERO; k];
    for i in 0..k {
        let s = sigma.data()[i].to_f64();
        if s > tol {
            sigma_inv[i] = T::from_f64(1.0 / s);
        }
    }

    // V @ diag(sigma_inv) @ U^T
    let ut = u.t()?;
    // First: diag(sigma_inv) @ U^T → scale rows of U^T
    let mut scaled = Vec::with_capacity(k * m);
    for i in 0..k {
        for j in 0..m {
            scaled.push(sigma_inv[i] * ut.get(&[i, j])?);
        }
    }
    let scaled_ut = Tensor::new(scaled, vec![k, m])?;

    // V @ scaled_ut
    v.matmul(&scaled_ut)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_svd_basic() {
        let a: Tensor<f64> = Tensor::from_vec2d(&[
            vec![3.0, 0.0],
            vec![0.0, 4.0],
        ]).unwrap();

        let (u, sigma, v) = svd(&a).unwrap();
        // Singular values should be 4 and 3
        let s = sigma.data();
        assert!((s[0] - 4.0).abs() < 0.1, "σ₁ = {}", s[0]);
        assert!((s[1] - 3.0).abs() < 0.1, "σ₂ = {}", s[1]);
    }

    #[test]
    fn test_pinv() {
        let a: Tensor<f64> = Tensor::from_vec2d(&[
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ]).unwrap();

        let a_pinv = pinv(&a).unwrap();
        assert_eq!(a_pinv.shape_vec(), vec![2, 3]);

        // A⁺ A should be approximately I (2x2)
        let ata = a_pinv.matmul(&a).unwrap();
        assert!((ata.get(&[0, 0]).unwrap() - 1.0).abs() < 0.1);
        assert!((ata.get(&[1, 1]).unwrap() - 1.0).abs() < 0.1);
    }
}
