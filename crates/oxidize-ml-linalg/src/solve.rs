use oxidize_ml_core::{Float, Tensor, TensorError};
use oxidize_ml_core::error::TensorResult;
use rand::distributions::{Distribution, Standard};

use crate::decomposition::{lu, qr};

/// Solve the linear system Ax = b using LU decomposition.
pub fn solve<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> TensorResult<Tensor<T>>
where
    Standard: Distribution<T>,
{
    if a.ndim() != 2 {
        return Err(TensorError::InvalidOperation("solve: A must be 2D".into()));
    }
    let n = a.shape().dim(0)?;
    if n != a.shape().dim(1)? {
        return Err(TensorError::InvalidOperation("solve: A must be square".into()));
    }

    let decomp = lu(a)?;

    // Handle b as either 1D vector or 2D matrix
    let (b_data, ncols) = if b.ndim() == 1 {
        if b.numel() != n {
            return Err(TensorError::DimensionMismatch(format!(
                "solve: b has {} elements but A is {}x{}",
                b.numel(), n, n
            )));
        }
        (b.data().to_vec(), 1usize)
    } else if b.ndim() == 2 {
        if b.shape().dim(0)? != n {
            return Err(TensorError::DimensionMismatch(format!(
                "solve: b has {} rows but A is {}x{}",
                b.shape().dim(0)?, n, n
            )));
        }
        (b.data().to_vec(), b.shape().dim(1)?)
    } else {
        return Err(TensorError::InvalidOperation("solve: b must be 1D or 2D".into()));
    };

    let mut result = vec![T::ZERO; n * ncols];

    for col in 0..ncols {
        // Permute b
        let mut pb = vec![T::ZERO; n];
        for i in 0..n {
            pb[i] = b_data[decomp.pivot[i] * ncols + col];
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
            result[i * ncols + col] = x[i];
        }
    }

    if ncols == 1 {
        Tensor::new(result, vec![n])
    } else {
        Tensor::new(result, vec![n, ncols])
    }
}

/// Least-squares solution: minimize ||Ax - b||² using QR decomposition.
/// Works for overdetermined systems (m > n).
pub fn lstsq<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> TensorResult<Tensor<T>>
where
    Standard: Distribution<T>,
{
    if a.ndim() != 2 {
        return Err(TensorError::InvalidOperation("lstsq: A must be 2D".into()));
    }
    let m = a.shape().dim(0)?;
    let n = a.shape().dim(1)?;

    if b.ndim() != 1 || b.numel() != m {
        return Err(TensorError::DimensionMismatch(format!(
            "lstsq: b must be 1D with {} elements",
            m
        )));
    }

    let decomp = qr(a)?;

    // Compute Qᵀ * b
    let qt = decomp.q.t()?;
    let b_2d = b.reshape(vec![m, 1])?;
    let qtb = qt.matmul(&b_2d)?;

    // Back-substitution on R * x = Qᵀb (first n rows)
    let mut x = vec![T::ZERO; n];
    for i in (0..n).rev() {
        let mut sum = T::ZERO;
        for j in (i + 1)..n {
            sum = sum + decomp.r.get(&[i, j])? * x[j];
        }
        let diag = decomp.r.get(&[i, i])?;
        if diag.abs() < T::EPSILON {
            return Err(TensorError::SingularMatrix);
        }
        x[i] = (qtb.get(&[i, 0])? - sum) / diag;
    }

    Tensor::new(x, vec![n])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve() {
        // 2x + y = 5
        // x + 3y = 7
        // Solution: x=1.6, y=1.8
        let a: Tensor<f64> = Tensor::new(vec![2.0, 1.0, 1.0, 3.0], vec![2, 2]).unwrap();
        let b: Tensor<f64> = Tensor::from_slice(&[5.0, 7.0]);
        let x = solve(&a, &b).unwrap();
        assert!((x.data()[0] - 1.6).abs() < 1e-10);
        assert!((x.data()[1] - 1.8).abs() < 1e-10);
    }

    #[test]
    fn test_lstsq() {
        // Overdetermined system: fit y = 2x + 1
        // x features: [1, 1], [1, 2], [1, 3]  (column of 1s for intercept)
        // y values: [3, 5, 7]
        let a: Tensor<f64> = Tensor::new(
            vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0],
            vec![3, 2],
        ).unwrap();
        let b: Tensor<f64> = Tensor::from_slice(&[3.0, 5.0, 7.0]);
        let x = lstsq(&a, &b).unwrap();
        // Should get approximately [1, 2] (intercept=1, slope=2)
        assert!((x.data()[0] - 1.0).abs() < 1e-10);
        assert!((x.data()[1] - 2.0).abs() < 1e-10);
    }
}
