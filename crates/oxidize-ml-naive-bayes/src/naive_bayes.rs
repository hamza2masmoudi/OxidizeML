use oxidize_ml_core::{Float, Tensor};
use oxidize_ml_core::error::TensorResult;

/// Gaussian Naive Bayes classifier.
pub struct GaussianNB<T: Float> {
    pub class_priors: Vec<T>,
    pub class_means: Vec<Vec<T>>,
    pub class_vars: Vec<Vec<T>>,
    pub n_classes: usize,
    pub n_features: usize,
}

impl<T: Float> GaussianNB<T> {
    pub fn new() -> Self {
        GaussianNB {
            class_priors: Vec::new(),
            class_means: Vec::new(),
            class_vars: Vec::new(),
            n_classes: 0,
            n_features: 0,
        }
    }

    pub fn fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> TensorResult<()> {
        let n = x.shape().dim(0)?;
        let p = x.shape().dim(1)?;
        self.n_features = p;

        let max_label = y.data().iter().map(|v| v.to_f64().round() as usize).max().unwrap_or(0);
        self.n_classes = max_label + 1;

        self.class_priors = vec![T::ZERO; self.n_classes];
        self.class_means = vec![vec![T::ZERO; p]; self.n_classes];
        self.class_vars = vec![vec![T::ZERO; p]; self.n_classes];
        let mut class_counts = vec![0usize; self.n_classes];

        // Compute means
        for i in 0..n {
            let cls = y.data()[i].to_f64().round() as usize;
            class_counts[cls] += 1;
            for j in 0..p {
                self.class_means[cls][j] = self.class_means[cls][j] + x.get(&[i, j])?;
            }
        }
        for c in 0..self.n_classes {
            if class_counts[c] > 0 {
                let cnt = T::from_usize(class_counts[c]);
                for j in 0..p {
                    self.class_means[c][j] = self.class_means[c][j] / cnt;
                }
            }
            self.class_priors[c] = T::from_usize(class_counts[c]) / T::from_usize(n);
        }

        // Compute variances
        for i in 0..n {
            let cls = y.data()[i].to_f64().round() as usize;
            for j in 0..p {
                let diff = x.get(&[i, j])? - self.class_means[cls][j];
                self.class_vars[cls][j] = self.class_vars[cls][j] + diff * diff;
            }
        }
        for c in 0..self.n_classes {
            if class_counts[c] > 0 {
                let cnt = T::from_usize(class_counts[c]);
                for j in 0..p {
                    self.class_vars[c][j] = self.class_vars[c][j] / cnt;
                    // Add small epsilon to prevent division by zero
                    if self.class_vars[c][j] < T::from_f64(1e-9) {
                        self.class_vars[c][j] = T::from_f64(1e-9);
                    }
                }
            }
        }

        Ok(())
    }

    /// Compute log-probability of x given class c using Gaussian PDF.
    fn log_likelihood(&self, x: &Tensor<T>, row: usize, class: usize) -> TensorResult<T> {
        let mut log_prob = T::ZERO;
        let two_pi = T::TWO * T::PI;

        for j in 0..self.n_features {
            let mean = self.class_means[class][j];
            let var = self.class_vars[class][j];
            let xij = x.get(&[row, j])?;
            let diff = xij - mean;

            // log N(x|μ,σ²) = -0.5 * (log(2π) + log(σ²) + (x-μ)²/σ²)
            log_prob = log_prob - T::HALF * (two_pi.ln() + var.ln() + diff * diff / var);
        }

        Ok(log_prob)
    }

    pub fn predict(&self, x: &Tensor<T>) -> TensorResult<Tensor<T>> {
        let n = x.shape().dim(0)?;
        let mut predictions = Vec::with_capacity(n);

        for i in 0..n {
            let mut best_log_prob = T::NEG_INFINITY;
            let mut best_class = 0;

            for c in 0..self.n_classes {
                let log_prior = self.class_priors[c].ln();
                let log_likelihood = self.log_likelihood(x, i, c)?;
                let log_posterior = log_prior + log_likelihood;

                if log_posterior > best_log_prob {
                    best_log_prob = log_posterior;
                    best_class = c;
                }
            }

            predictions.push(T::from_usize(best_class));
        }

        Tensor::new(predictions, vec![n])
    }
}

impl<T: Float> Default for GaussianNB<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_nb() {
        let x: Tensor<f64> = Tensor::from_vec2d(&[
            vec![0.0, 0.0], vec![0.5, 0.5], vec![1.0, 0.0],
            vec![5.0, 5.0], vec![5.5, 5.5], vec![6.0, 5.0],
        ]).unwrap();
        let y: Tensor<f64> = Tensor::from_slice(&[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut nb = GaussianNB::new();
        nb.fit(&x, &y).unwrap();
        let pred = nb.predict(&x).unwrap();

        for i in 0..6 {
            assert!((pred.data()[i] - y.data()[i]).abs() < 0.5);
        }
    }
}
