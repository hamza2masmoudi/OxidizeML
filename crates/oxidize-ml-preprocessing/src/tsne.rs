use oxidize_ml_core::{Float, Tensor, TensorError};
use oxidize_ml_core::error::TensorResult;
use std::marker::PhantomData;

/// t-SNE (t-distributed Stochastic Neighbor Embedding).
///
/// Reduces high-dimensional data to 2D or 3D for visualization.
/// Uses gradient descent to minimize KL divergence between
/// high-dimensional and low-dimensional probability distributions.
pub struct TSNE<T: Float> {
    pub n_components: usize,
    pub perplexity: f64,
    pub learning_rate: f64,
    pub n_iter: usize,
    _marker: PhantomData<T>,
}

impl<T: Float> TSNE<T> {
    pub fn new(n_components: usize) -> Self {
        TSNE {
            n_components,
            perplexity: 30.0,
            learning_rate: 200.0,
            n_iter: 1000,
            _marker: PhantomData,
        }
    }

    pub fn with_perplexity(mut self, perplexity: f64) -> Self {
        self.perplexity = perplexity;
        self
    }

    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn with_n_iter(mut self, n: usize) -> Self {
        self.n_iter = n;
        self
    }

    /// Compute pairwise distances.
    fn pairwise_distances(x: &Tensor<T>) -> Vec<Vec<f64>> {
        let n = x.shape().dim(0).unwrap();
        let p = x.shape().dim(1).unwrap();
        let mut dist = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in i + 1..n {
                let mut d = 0.0;
                for k in 0..p {
                    let diff = x.get(&[i, k]).unwrap().to_f64() - x.get(&[j, k]).unwrap().to_f64();
                    d += diff * diff;
                }
                dist[i][j] = d;
                dist[j][i] = d;
            }
        }
        dist
    }

    /// Compute conditional probabilities p(j|i) using binary search for sigma.
    fn compute_pij(dist: &[Vec<f64>], perplexity: f64) -> Vec<Vec<f64>> {
        let n = dist.len();
        let target_entropy = perplexity.ln();
        let mut pij = vec![vec![0.0f64; n]; n];

        for i in 0..n {
            // Binary search for sigma_i
            let mut sigma_lo = 1e-10;
            let mut sigma_hi = 1e10;
            let mut sigma = 1.0;

            for _ in 0..50 {
                // Compute p(j|i) with this sigma
                let mut sum_exp = 0.0;
                for j in 0..n {
                    if j != i {
                        sum_exp += (-dist[i][j] / (2.0 * sigma * sigma)).exp();
                    }
                }

                if sum_exp < 1e-15 { sigma_lo = sigma; sigma *= 2.0; continue; }

                // Compute entropy
                let mut entropy = 0.0;
                for j in 0..n {
                    if j != i {
                        let p = (-dist[i][j] / (2.0 * sigma * sigma)).exp() / sum_exp;
                        if p > 1e-15 { entropy -= p * p.ln(); }
                    }
                }

                if (entropy - target_entropy).abs() < 1e-5 { break; }

                if entropy > target_entropy {
                    sigma_hi = sigma;
                } else {
                    sigma_lo = sigma;
                }
                sigma = (sigma_lo + sigma_hi) / 2.0;
            }

            // Set p(j|i) with final sigma
            let mut sum_exp = 0.0;
            for j in 0..n {
                if j != i {
                    sum_exp += (-dist[i][j] / (2.0 * sigma * sigma)).exp();
                }
            }
            if sum_exp < 1e-15 { sum_exp = 1e-15; }
            for j in 0..n {
                if j != i {
                    pij[i][j] = (-dist[i][j] / (2.0 * sigma * sigma)).exp() / sum_exp;
                }
            }
        }

        // Symmetrize: P = (P + Pᵀ) / (2n)
        let n_f = n as f64;
        for i in 0..n {
            for j in i + 1..n {
                let sym = (pij[i][j] + pij[j][i]) / (2.0 * n_f);
                pij[i][j] = sym.max(1e-12);
                pij[j][i] = sym.max(1e-12);
            }
        }
        pij
    }

    /// Fit and transform the data to low-dimensional embedding.
    pub fn fit_transform(&self, x: &Tensor<T>) -> TensorResult<Tensor<T>> {
        let n = x.shape().dim(0)?;
        let d = self.n_components;

        // Compute high-dim probabilities
        let dist = Self::pairwise_distances(x);
        let pij = Self::compute_pij(&dist, self.perplexity);

        // Initialize Y randomly (small values)
        let mut y: Vec<Vec<f64>> = (0..n).map(|i| {
            (0..d).map(|j| {
                let seed = (i * d + j) as f64;
                ((seed * 0.7 + 0.3).sin()) * 0.01
            }).collect()
        }).collect();

        // Gradient descent with momentum
        let mut gains = vec![vec![1.0f64; d]; n];
        let mut y_velocity = vec![vec![0.0f64; d]; n];
        let momentum_init = 0.5;
        let momentum_final = 0.8;

        for iter in 0..self.n_iter {
            let momentum = if iter < 250 { momentum_init } else { momentum_final };

            // Compute low-dim affinities (Student-t, 1 DOF)
            let mut qij = vec![vec![0.0f64; n]; n];
            let mut q_sum = 0.0;
            for i in 0..n {
                for j in i + 1..n {
                    let mut sq_dist = 0.0;
                    for k in 0..d {
                        let diff = y[i][k] - y[j][k];
                        sq_dist += diff * diff;
                    }
                    let q = 1.0 / (1.0 + sq_dist);
                    qij[i][j] = q;
                    qij[j][i] = q;
                    q_sum += 2.0 * q;
                }
            }
            if q_sum < 1e-15 { q_sum = 1e-15; }

            // Normalize Q
            for i in 0..n {
                for j in 0..n {
                    qij[i][j] = (qij[i][j] / q_sum).max(1e-12);
                }
            }

            // Compute gradients
            for i in 0..n {
                for k in 0..d {
                    let mut grad = 0.0;
                    for j in 0..n {
                        if j != i {
                            let pq_diff = pij[i][j] - qij[i][j];
                            let mut sq_dist = 0.0;
                            for kk in 0..d {
                                let diff = y[i][kk] - y[j][kk];
                                sq_dist += diff * diff;
                            }
                            let inv = 1.0 / (1.0 + sq_dist);
                            grad += 4.0 * pq_diff * (y[i][k] - y[j][k]) * inv;
                        }
                    }

                    // Update with momentum and adaptive gains
                    if (grad > 0.0) != (y_velocity[i][k] > 0.0) {
                        gains[i][k] += 0.2;
                    } else {
                        gains[i][k] = (gains[i][k] * 0.8).max(0.01);
                    }

                    y_velocity[i][k] = momentum * y_velocity[i][k]
                        - self.learning_rate * gains[i][k] * grad;
                    y[i][k] += y_velocity[i][k];
                }
            }

            // Center Y
            for k in 0..d {
                let mean: f64 = y.iter().map(|yi| yi[k]).sum::<f64>() / n as f64;
                for i in 0..n {
                    y[i][k] -= mean;
                }
            }
        }

        // Convert to tensor
        let mut data = Vec::with_capacity(n * d);
        for i in 0..n {
            for k in 0..d {
                data.push(T::from_f64(y[i][k]));
            }
        }
        Tensor::new(data, vec![n, d])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tsne() {
        // Simple 2-cluster data
        let x: Tensor<f64> = Tensor::from_vec2d(&[
            vec![0.0, 0.0, 0.0], vec![0.1, 0.1, 0.1], vec![0.2, 0.2, 0.2],
            vec![5.0, 5.0, 5.0], vec![5.1, 5.1, 5.1], vec![5.2, 5.2, 5.2],
        ]).unwrap();

        let tsne = TSNE::<f64>::new(2).with_perplexity(2.0).with_n_iter(100);
        let y = tsne.fit_transform(&x).unwrap();
        assert_eq!(y.shape_vec(), vec![6, 2]);

        // Cluster separation in 2D should be visible
        // (points 0-2 should be close to each other, 3-5 close to each other)
        let d01_sq = (y.get(&[0, 0]).unwrap() - y.get(&[1, 0]).unwrap()).powi(2)
            + (y.get(&[0, 1]).unwrap() - y.get(&[1, 1]).unwrap()).powi(2);
        let d03_sq = (y.get(&[0, 0]).unwrap() - y.get(&[3, 0]).unwrap()).powi(2)
            + (y.get(&[0, 1]).unwrap() - y.get(&[3, 1]).unwrap()).powi(2);
        // intra-cluster distance should be smaller than inter-cluster
        assert!(d01_sq < d03_sq, "t-SNE should separate clusters: intra={} inter={}", d01_sq, d03_sq);
    }
}
