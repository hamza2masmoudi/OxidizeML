use oxidize_ml_core::{Float, Tensor, TensorError};
use oxidize_ml_core::error::TensorResult;
use rand::distributions::{Distribution, Standard};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// K-Means clustering with k-means++ initialization.
pub struct KMeans<T: Float> {
    pub n_clusters: usize,
    pub max_iter: usize,
    pub tol: T,
    pub seed: Option<u64>,
    pub centroids: Option<Tensor<T>>,
    pub labels: Option<Tensor<T>>,
    pub inertia: Option<T>,
}

impl<T: Float> KMeans<T>
where
    Standard: Distribution<T>,
{
    pub fn new(n_clusters: usize, max_iter: usize) -> Self {
        KMeans {
            n_clusters,
            max_iter,
            tol: T::from_f64(1e-4),
            seed: Some(42),
            centroids: None,
            labels: None,
            inertia: None,
        }
    }

    /// Fit the model to data.
    pub fn fit(&mut self, x: &Tensor<T>) -> TensorResult<()> {
        let n = x.shape().dim(0)?;
        let d = x.shape().dim(1)?;

        // K-means++ initialization
        let mut centroids = self.init_centroids_pp(x, n, d)?;
        let mut labels = vec![0usize; n];

        for _iter in 0..self.max_iter {
            // Assignment step
            for i in 0..n {
                let mut best_dist = T::INFINITY;
                let mut best_k = 0;
                for k in 0..self.n_clusters {
                    let mut dist = T::ZERO;
                    for j in 0..d {
                        let diff = x.get(&[i, j])? - centroids[k * d + j];
                        dist = dist + diff * diff;
                    }
                    if dist < best_dist {
                        best_dist = dist;
                        best_k = k;
                    }
                }
                labels[i] = best_k;
            }

            // Update step
            let mut new_centroids = vec![T::ZERO; self.n_clusters * d];
            let mut counts = vec![0usize; self.n_clusters];
            for i in 0..n {
                let k = labels[i];
                counts[k] += 1;
                for j in 0..d {
                    new_centroids[k * d + j] = new_centroids[k * d + j] + x.get(&[i, j])?;
                }
            }
            for k in 0..self.n_clusters {
                if counts[k] > 0 {
                    for j in 0..d {
                        new_centroids[k * d + j] = new_centroids[k * d + j] / T::from_usize(counts[k]);
                    }
                }
            }

            // Check convergence
            let mut max_shift = T::ZERO;
            for i in 0..new_centroids.len() {
                let shift = (new_centroids[i] - centroids[i]).abs();
                if shift > max_shift {
                    max_shift = shift;
                }
            }

            centroids = new_centroids;
            if max_shift < self.tol {
                break;
            }
        }

        // Compute inertia
        let mut inertia = T::ZERO;
        for i in 0..n {
            let k = labels[i];
            for j in 0..d {
                let diff = x.get(&[i, j])? - centroids[k * d + j];
                inertia = inertia + diff * diff;
            }
        }

        self.centroids = Some(Tensor::new(centroids, vec![self.n_clusters, d])?);
        let label_data: Vec<T> = labels.iter().map(|&l| T::from_usize(l)).collect();
        self.labels = Some(Tensor::new(label_data, vec![n])?);
        self.inertia = Some(inertia);

        Ok(())
    }

    fn init_centroids_pp(&self, x: &Tensor<T>, n: usize, d: usize) -> TensorResult<Vec<T>> {
        let mut rng = match self.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        let mut centroids = Vec::with_capacity(self.n_clusters * d);

        // Pick first centroid randomly
        let first = (rand::Rng::gen::<f64>(&mut rng) * n as f64) as usize;
        let first = first.min(n - 1);
        for j in 0..d {
            centroids.push(x.get(&[first, j])?);
        }

        // Pick remaining centroids proportional to distance²
        for _k in 1..self.n_clusters {
            let mut distances = vec![T::INFINITY; n];
            let n_existing = centroids.len() / d;

            for i in 0..n {
                for c in 0..n_existing {
                    let mut dist = T::ZERO;
                    for j in 0..d {
                        let diff = x.get(&[i, j])? - centroids[c * d + j];
                        dist = dist + diff * diff;
                    }
                    if dist < distances[i] {
                        distances[i] = dist;
                    }
                }
            }

            let total: T = distances.iter().copied().sum();
            let threshold = T::from_f64(rand::Rng::gen::<f64>(&mut rng)) * total;
            let mut cumulative = T::ZERO;
            let mut selected = 0;
            for (i, &d) in distances.iter().enumerate() {
                cumulative = cumulative + d;
                if cumulative >= threshold {
                    selected = i;
                    break;
                }
            }

            for j in 0..d {
                centroids.push(x.get(&[selected, j])?);
            }
        }

        Ok(centroids)
    }

    /// Predict cluster labels for new data.
    pub fn predict(&self, x: &Tensor<T>) -> TensorResult<Tensor<T>> {
        let centroids = self.centroids.as_ref().ok_or_else(|| {
            TensorError::InvalidOperation("Model not fitted".into())
        })?;
        let n = x.shape().dim(0)?;
        let d = x.shape().dim(1)?;
        let mut labels = Vec::with_capacity(n);

        for i in 0..n {
            let mut best_dist = T::INFINITY;
            let mut best_k = 0;
            for k in 0..self.n_clusters {
                let mut dist = T::ZERO;
                for j in 0..d {
                    let diff = x.get(&[i, j])? - centroids.get(&[k, j])?;
                    dist = dist + diff * diff;
                }
                if dist < best_dist {
                    best_dist = dist;
                    best_k = k;
                }
            }
            labels.push(T::from_usize(best_k));
        }

        Tensor::new(labels, vec![n])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans() {
        // Two clear clusters
        let x: Tensor<f64> = Tensor::from_vec2d(&[
            vec![0.0, 0.0], vec![0.5, 0.5], vec![1.0, 0.0],
            vec![10.0, 10.0], vec![10.5, 10.5], vec![11.0, 10.0],
        ]).unwrap();

        let mut km = KMeans::new(2, 100);
        km.fit(&x).unwrap();

        let labels = km.labels.as_ref().unwrap();
        // First 3 should be same cluster, last 3 should be same cluster
        let a = labels.data()[0].to_f64().round() as usize;
        let b = labels.data()[3].to_f64().round() as usize;
        assert_ne!(a, b);
        assert_eq!(labels.data()[0].to_f64().round() as usize, labels.data()[1].to_f64().round() as usize);
        assert_eq!(labels.data()[3].to_f64().round() as usize, labels.data()[4].to_f64().round() as usize);
    }
}
