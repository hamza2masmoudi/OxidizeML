use oxidize_ml_core::{Float, Tensor, TensorError};
use oxidize_ml_core::error::TensorResult;
use rand::distributions::{Distribution, Standard};

/// LU decomposition result: A = P * L * U
pub struct LuDecomposition<T: Float> {
    pub l: Tensor<T>,
    pub u: Tensor<T>,
    pub pivot: Vec<usize>,
}

/// QR decomposition result: A = Q * R
pub struct QrDecomposition<T: Float> {
    pub q: Tensor<T>,
    pub r: Tensor<T>,
}

/// Cholesky decomposition result: A = L * Lᵀ
pub struct CholeskyDecomposition<T: Float> {
    pub l: Tensor<T>,
}

/// SVD result: A = U * Σ * Vᵀ
pub struct SvdDecomposition<T: Float> {
    pub u: Tensor<T>,
    pub s: Tensor<T>, // singular values as 1-D
    pub vt: Tensor<T>,
}

/// LU decomposition with partial pivoting.
pub fn lu<T: Float>(a: &Tensor<T>) -> TensorResult<LuDecomposition<T>>
where
    Standard: Distribution<T>,
{
    if a.ndim() != 2 {
        return Err(TensorError::InvalidOperation("LU requires a 2D tensor".into()));
    }
    let n = a.shape().dim(0)?;
    let m = a.shape().dim(1)?;
    if n != m {
        return Err(TensorError::InvalidOperation("LU requires a square matrix".into()));
    }

    // Copy A into working matrix
    let mut u_data = a.data().to_vec();
    let mut l_data = vec![T::ZERO; n * n];
    let mut pivot: Vec<usize> = (0..n).collect();

    for k in 0..n {
        // Find pivot
        let mut max_val = u_data[k * n + k].abs();
        let mut max_row = k;
        for i in (k + 1)..n {
            let v = u_data[i * n + k].abs();
            if v > max_val {
                max_val = v;
                max_row = i;
            }
        }

        if max_val < T::EPSILON {
            return Err(TensorError::SingularMatrix);
        }

        // Swap rows
        if max_row != k {
            pivot.swap(k, max_row);
            for j in 0..n {
                let tmp = u_data[k * n + j];
                u_data[k * n + j] = u_data[max_row * n + j];
                u_data[max_row * n + j] = tmp;
            }
            // Swap L rows for already computed columns
            for j in 0..k {
                let tmp = l_data[k * n + j];
                l_data[k * n + j] = l_data[max_row * n + j];
                l_data[max_row * n + j] = tmp;
            }
        }

        l_data[k * n + k] = T::ONE;

        for i in (k + 1)..n {
            let factor = u_data[i * n + k] / u_data[k * n + k];
            l_data[i * n + k] = factor;
            for j in k..n {
                u_data[i * n + j] = u_data[i * n + j] - factor * u_data[k * n + j];
            }
        }
    }

    Ok(LuDecomposition {
        l: Tensor::new(l_data, vec![n, n])?,
        u: Tensor::new(u_data, vec![n, n])?,
        pivot,
    })
}

/// QR decomposition via Householder reflections.
pub fn qr<T: Float>(a: &Tensor<T>) -> TensorResult<QrDecomposition<T>>
where
    Standard: Distribution<T>,
{
    if a.ndim() != 2 {
        return Err(TensorError::InvalidOperation("QR requires a 2D tensor".into()));
    }
    let m = a.shape().dim(0)?;
    let n = a.shape().dim(1)?;
    let k = m.min(n);

    let mut r_data = a.data().to_vec();
    // Q starts as identity
    let mut q_data = vec![T::ZERO; m * m];
    for i in 0..m {
        q_data[i * m + i] = T::ONE;
    }

    for j in 0..k {
        // Extract column j below diagonal
        let mut x = vec![T::ZERO; m - j];
        for i in j..m {
            x[i - j] = r_data[i * n + j];
        }

        // Compute Householder vector
        let mut norm_x = T::ZERO;
        for &v in &x {
            norm_x = norm_x + v * v;
        }
        norm_x = norm_x.sqrt();

        if norm_x < T::EPSILON {
            continue;
        }

        let sign = if x[0] >= T::ZERO { T::ONE } else { T::NEG_ONE };
        x[0] = x[0] + sign * norm_x;

        // Normalize
        let mut norm_v = T::ZERO;
        for &v in &x {
            norm_v = norm_v + v * v;
        }
        norm_v = norm_v.sqrt();
        if norm_v < T::EPSILON {
            continue;
        }
        for v in x.iter_mut() {
            *v = *v / norm_v;
        }

        // Apply H = I - 2*v*vᵀ to R (columns j..n)
        for col in j..n {
            let mut dot = T::ZERO;
            for i in j..m {
                dot = dot + x[i - j] * r_data[i * n + col];
            }
            for i in j..m {
                r_data[i * n + col] = r_data[i * n + col] - T::TWO * x[i - j] * dot;
            }
        }

        // Apply H to Q from the right: Q = Q * H
        for row in 0..m {
            let mut dot = T::ZERO;
            for i in j..m {
                dot = dot + q_data[row * m + i] * x[i - j];
            }
            for i in j..m {
                q_data[row * m + i] = q_data[row * m + i] - T::TWO * dot * x[i - j];
            }
        }
    }

    // Truncate Q to m×k and R to k×n
    let mut q_trunc = vec![T::ZERO; m * k];
    for i in 0..m {
        for j in 0..k {
            q_trunc[i * k + j] = q_data[i * m + j];
        }
    }
    let mut r_trunc = vec![T::ZERO; k * n];
    for i in 0..k {
        for j in 0..n {
            r_trunc[i * n + j] = r_data[i * n + j];
        }
    }

    Ok(QrDecomposition {
        q: Tensor::new(q_trunc, vec![m, k])?,
        r: Tensor::new(r_trunc, vec![k, n])?,
    })
}

/// Cholesky decomposition for symmetric positive-definite matrices: A = L * Lᵀ
pub fn cholesky<T: Float>(a: &Tensor<T>) -> TensorResult<CholeskyDecomposition<T>>
where
    Standard: Distribution<T>,
{
    if a.ndim() != 2 {
        return Err(TensorError::InvalidOperation("Cholesky requires a 2D tensor".into()));
    }
    let n = a.shape().dim(0)?;
    if n != a.shape().dim(1)? {
        return Err(TensorError::InvalidOperation("Cholesky requires a square matrix".into()));
    }

    let mut l_data = vec![T::ZERO; n * n];

    for i in 0..n {
        for j in 0..=i {
            let mut sum = T::ZERO;
            for k in 0..j {
                sum = sum + l_data[i * n + k] * l_data[j * n + k];
            }

            if i == j {
                let val = a.get(&[i, i])? - sum;
                if val <= T::ZERO {
                    return Err(TensorError::InvalidOperation(
                        "Matrix is not positive definite".into(),
                    ));
                }
                l_data[i * n + j] = val.sqrt();
            } else {
                l_data[i * n + j] = (a.get(&[i, j])? - sum) / l_data[j * n + j];
            }
        }
    }

    Ok(CholeskyDecomposition {
        l: Tensor::new(l_data, vec![n, n])?,
    })
}

/// Compute the determinant of a square matrix using LU decomposition.
pub fn det<T: Float>(a: &Tensor<T>) -> TensorResult<T>
where
    Standard: Distribution<T>,
{
    let decomp = lu(a)?;
    let n = a.shape().dim(0)?;
    let mut d = T::ONE;
    for i in 0..n {
        d = d * decomp.u.get(&[i, i])?;
    }

    // Count transpositions in the permutation
    let mut swaps = 0usize;
    let mut visited = vec![false; n];
    for i in 0..n {
        if !visited[i] {
            visited[i] = true;
            let mut j = decomp.pivot[i];
            let mut cycle_len = 1;
            while j != i {
                visited[j] = true;
                j = decomp.pivot[j];
                cycle_len += 1;
            }
            // A cycle of length k requires k-1 transpositions
            swaps += cycle_len - 1;
        }
    }

    if swaps % 2 == 1 {
        d = -d;
    }
    Ok(d)
}

/// Matrix inverse using LU decomposition.
pub fn inv<T: Float>(a: &Tensor<T>) -> TensorResult<Tensor<T>>
where
    Standard: Distribution<T>,
{
    let n = a.shape().dim(0)?;
    let identity = Tensor::<T>::eye(n);
    let mut result_data = vec![T::ZERO; n * n];

    let decomp = lu(a)?;

    for col in 0..n {
        // Solve L*U*x = P*e_col
        let mut b = vec![T::ZERO; n];
        b[decomp.pivot[col]] = T::ONE;

        // Build the permuted column correctly
        let mut pb = vec![T::ZERO; n];
        for i in 0..n {
            pb[i] = if decomp.pivot[i] == col { T::ONE } else { T::ZERO };
        }

        // Forward substitution: L * y = pb
        let mut y = vec![T::ZERO; n];
        for i in 0..n {
            let mut sum = T::ZERO;
            for j in 0..i {
                sum = sum + decomp.l.get(&[i, j])? * y[j];
            }
            y[i] = pb[i] - sum;
        }

        // Back substitution: U * x = y
        let mut x = vec![T::ZERO; n];
        for i in (0..n).rev() {
            let mut sum = T::ZERO;
            for j in (i + 1)..n {
                sum = sum + decomp.u.get(&[i, j])? * x[j];
            }
            let diag = decomp.u.get(&[i, i])?;
            if diag.abs() < T::EPSILON {
                return Err(TensorError::SingularMatrix);
            }
            x[i] = (y[i] - sum) / diag;
        }

        for i in 0..n {
            result_data[i * n + col] = x[i];
        }
    }

    Tensor::new(result_data, vec![n, n])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lu() {
        let a: Tensor<f64> = Tensor::new(
            vec![2.0, 1.0, 1.0, 4.0, 3.0, 3.0, 8.0, 7.0, 9.0],
            vec![3, 3],
        ).unwrap();
        let decomp = lu(&a).unwrap();
        // Verify L * U ≈ P * A
        let lu_product = decomp.l.matmul(&decomp.u).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let orig_row = decomp.pivot[i];
                let diff = (lu_product.get(&[i, j]).unwrap() - a.get(&[orig_row, j]).unwrap()).abs();
                assert!(diff < 1e-10, "LU product mismatch at ({}, {})", i, j);
            }
        }
    }

    #[test]
    fn test_qr() {
        let a: Tensor<f64> = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
            vec![3, 3],
        ).unwrap();
        let decomp = qr(&a).unwrap();
        // Q * R ≈ A
        let qr_product = decomp.q.matmul(&decomp.r).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let diff = (qr_product.get(&[i, j]).unwrap() - a.get(&[i, j]).unwrap()).abs();
                assert!(diff < 1e-10, "QR product mismatch at ({}, {}): diff={}", i, j, diff);
            }
        }
        // Q should be orthogonal: QᵀQ ≈ I
        let qt = decomp.q.t().unwrap();
        let qtq = qt.matmul(&decomp.q).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                let diff = (qtq.get(&[i, j]).unwrap() - expected).abs();
                assert!(diff < 1e-10, "QᵀQ not identity at ({}, {})", i, j);
            }
        }
    }

    #[test]
    fn test_cholesky() {
        // Symmetric positive definite matrix
        let a: Tensor<f64> = Tensor::new(
            vec![4.0, 2.0, 2.0, 3.0],
            vec![2, 2],
        ).unwrap();
        let decomp = cholesky(&a).unwrap();
        // L * Lᵀ ≈ A
        let lt = decomp.l.t().unwrap();
        let llt = decomp.l.matmul(&lt).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                let diff = (llt.get(&[i, j]).unwrap() - a.get(&[i, j]).unwrap()).abs();
                assert!(diff < 1e-10, "Cholesky LLᵀ mismatch at ({}, {})", i, j);
            }
        }
    }

    #[test]
    fn test_det() {
        let a: Tensor<f64> = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
        ).unwrap();
        let d = det(&a).unwrap();
        // det = 1*4 - 2*3 = -2
        assert!((d - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_inv() {
        let a: Tensor<f64> = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
        ).unwrap();
        let a_inv = inv(&a).unwrap();
        let product = a.matmul(&a_inv).unwrap();
        // Should be identity
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                let diff = (product.get(&[i, j]).unwrap() - expected).abs();
                assert!(diff < 1e-10, "A*A⁻¹ not identity at ({}, {})", i, j);
            }
        }
    }
}
