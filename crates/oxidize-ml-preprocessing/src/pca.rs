use oxidize_ml_core::{Float, Tensor, TensorError};
use oxidize_ml_core::error::TensorResult;

/// Principal Component Analysis (PCA).
///
/// Reduces dimensionality by projecting data onto the top-k
/// principal components (directions of maximum variance).
///
/// Uses the power iteration method for eigendecomposition of the
/// covariance matrix (pure Rust, no external BLAS).
pub struct PCA<T: Float> {
    pub n_components: usize,
    pub components: Option<Tensor<T>>,   // [n_components, n_features]
    pub explained_variance: Option<Vec<f64>>,
    pub mean: Option<Tensor<T>>,
}

impl<T: Float> PCA<T> {
    pub fn new(n_components: usize) -> Self {
        PCA {
            n_components,
            components: None,
            explained_variance: None,
            mean: None,
        }
    }

    pub fn fit(&mut self, x: &Tensor<T>) -> TensorResult<()> {
        let n = x.shape().dim(0)?;
        let p = x.shape().dim(1)?;
        let k = self.n_components.min(p);
        let n_f = T::from_usize(n);

        // 1. Center the data (subtract mean)
        let mean = x.mean_axis(0)?;
        self.mean = Some(mean.clone());

        // Subtract mean from each row
        let mut centered_data = Vec::with_capacity(n * p);
        for i in 0..n {
            for j in 0..p {
                centered_data.push(x.get(&[i, j])? - mean.data()[j]);
            }
        }
        let centered = Tensor::new(centered_data, vec![n, p])?;

        // 2. Compute covariance matrix: C = (1/n) * Xᵀ X
        let ct = centered.t()?;
        let cov = ct.matmul(&centered)?;
        let mut cov_data: Vec<T> = cov.data().iter().map(|&v| v / n_f).collect();

        // 3. Power iteration for top-k eigenvectors
        let mut components_data = Vec::with_capacity(k * p);
        let mut eigenvalues = Vec::with_capacity(k);

        for _comp in 0..k {
            // Random initial vector
            let mut v: Vec<T> = (0..p).map(|i| T::from_f64((i as f64 + 1.0).sin())).collect();

            // Normalize
            let mut norm: T = v.iter().map(|&x| x * x).sum::<T>().sqrt();
            for x in v.iter_mut() { *x = *x / norm; }

            // Power iteration
            for _ in 0..200 {
                // w = C * v
                let mut w = vec![T::ZERO; p];
                for i in 0..p {
                    for j in 0..p {
                        w[i] = w[i] + T::from_f64(cov_data[i * p + j].to_f64()) * v[j];
                    }
                }

                // Eigenvalue estimate
                let eigenvalue: T = w.iter().zip(v.iter()).map(|(&wi, &vi)| wi * vi).sum();

                // Normalize
                norm = w.iter().map(|&x| x * x).sum::<T>().sqrt();
                if norm < T::EPSILON { break; }
                for j in 0..p { v[j] = w[j] / norm; }

                eigenvalues.push(eigenvalue.to_f64());
            }

            // Store this component
            components_data.extend_from_slice(&v);

            // Deflate: C = C - λ * v * vᵀ
            let lambda = eigenvalues.last().copied().unwrap_or(0.0);
            eigenvalues.truncate(eigenvalues.len().min(_comp + 1));
            if eigenvalues.len() <= _comp {
                eigenvalues.push(lambda);
            } else {
                eigenvalues[_comp] = lambda;
            }

            for i in 0..p {
                for j in 0..p {
                    cov_data[i * p + j] = cov_data[i * p + j] - T::from_f64(lambda) * v[i] * v[j];
                }
            }
        }

        self.components = Some(Tensor::new(components_data, vec![k, p])?);
        self.explained_variance = Some(eigenvalues[..k].to_vec());
        Ok(())
    }

    /// Transform data by projecting onto principal components.
    pub fn transform(&self, x: &Tensor<T>) -> TensorResult<Tensor<T>> {
        let mean = self.mean.as_ref().ok_or_else(|| {
            TensorError::InvalidOperation("PCA not fitted".into())
        })?;
        let components = self.components.as_ref().unwrap();

        let n = x.shape().dim(0)?;
        let p = x.shape().dim(1)?;

        // Center
        let mut centered_data = Vec::with_capacity(n * p);
        for i in 0..n {
            for j in 0..p {
                centered_data.push(x.get(&[i, j])? - mean.data()[j]);
            }
        }
        let centered = Tensor::new(centered_data, vec![n, p])?;

        // Project: X_new = X_centered @ components.T
        let ct = components.t()?;
        centered.matmul(&ct)
    }

    /// Fit and transform in one step.
    pub fn fit_transform(&mut self, x: &Tensor<T>) -> TensorResult<Tensor<T>> {
        self.fit(x)?;
        self.transform(x)
    }

    /// Explained variance ratio for each component.
    pub fn explained_variance_ratio(&self) -> Option<Vec<f64>> {
        self.explained_variance.as_ref().map(|ev| {
            let total: f64 = ev.iter().sum();
            if total > 0.0 {
                ev.iter().map(|&v| v / total).collect()
            } else {
                vec![0.0; ev.len()]
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pca() {
        // Data with two correlated features
        let x: Tensor<f64> = Tensor::from_vec2d(&[
            vec![2.5, 2.4],
            vec![0.5, 0.7],
            vec![2.2, 2.9],
            vec![1.9, 2.2],
            vec![3.1, 3.0],
            vec![2.3, 2.7],
            vec![2.0, 1.6],
            vec![1.0, 1.1],
            vec![1.5, 1.6],
            vec![1.1, 0.9],
        ]).unwrap();

        let mut pca = PCA::new(1);
        let x_reduced = pca.fit_transform(&x).unwrap();
        assert_eq!(x_reduced.shape_vec(), vec![10, 1]);

        let ev = pca.explained_variance.as_ref().unwrap();
        assert!(ev[0] > 0.0, "First eigenvalue should be positive");
    }
}
