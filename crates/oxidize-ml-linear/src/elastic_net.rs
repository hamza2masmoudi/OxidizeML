use oxidize_ml_core::{Float, Tensor, TensorError};
use oxidize_ml_core::error::TensorResult;

/// ElasticNet regression combines L1 (Lasso) and L2 (Ridge) penalties.
///
/// Minimizes: (1/2n)||y - Xw||² + α·l1_ratio·||w||₁ + α·(1-l1_ratio)/2·||w||²₂
///
/// When l1_ratio = 1.0, equivalent to Lasso.
/// When l1_ratio = 0.0, equivalent to Ridge.
pub struct ElasticNet<T: Float> {
    pub alpha: T,
    pub l1_ratio: T,
    pub weights: Option<Tensor<T>>,
    pub bias: Option<T>,
    pub max_iter: usize,
    pub tol: T,
}

impl<T: Float> ElasticNet<T> {
    pub fn new(alpha: T, l1_ratio: T, max_iter: usize) -> Self {
        ElasticNet {
            alpha,
            l1_ratio,
            weights: None,
            bias: None,
            max_iter,
            tol: T::from_f64(1e-6),
        }
    }

    pub fn fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> TensorResult<()> {
        let n = x.shape().dim(0)?;
        let p = x.shape().dim(1)?;
        let n_t = T::from_usize(n);
        let l1_penalty = self.alpha * self.l1_ratio;
        let l2_penalty = self.alpha * (T::ONE - self.l1_ratio);

        let mut w = vec![T::ZERO; p];
        let mut b = T::ZERO;

        for _iter in 0..self.max_iter {
            let old_w = w.clone();

            // Update intercept
            let mut residual_sum = T::ZERO;
            for i in 0..n {
                let mut pred = b;
                for j in 0..p {
                    pred = pred + x.get(&[i, j])? * w[j];
                }
                residual_sum = residual_sum + (y.data()[i] - pred);
            }
            b = b + residual_sum / n_t;

            // Coordinate descent for each feature
            for j in 0..p {
                let mut rho = T::ZERO;
                for i in 0..n {
                    let mut pred = b;
                    for k in 0..p {
                        if k != j {
                            pred = pred + x.get(&[i, k])? * w[k];
                        }
                    }
                    let residual = y.data()[i] - pred;
                    rho = rho + x.get(&[i, j])? * residual;
                }
                rho = rho / n_t;

                let mut xj_sq = T::ZERO;
                for i in 0..n {
                    let xij = x.get(&[i, j])?;
                    xj_sq = xj_sq + xij * xij;
                }
                xj_sq = xj_sq / n_t;

                // Soft thresholding with L1 + L2 penalty
                let denom = xj_sq + l2_penalty;
                w[j] = if rho > l1_penalty {
                    (rho - l1_penalty) / denom
                } else if rho < -l1_penalty {
                    (rho + l1_penalty) / denom
                } else {
                    T::ZERO
                };
            }

            // Check convergence
            let mut max_change = T::ZERO;
            for j in 0..p {
                let change = (w[j] - old_w[j]).abs();
                if change > max_change { max_change = change; }
            }
            if max_change < self.tol { break; }
        }

        self.weights = Some(Tensor::new(w, vec![p])?);
        self.bias = Some(b);
        Ok(())
    }

    pub fn predict(&self, x: &Tensor<T>) -> TensorResult<Tensor<T>> {
        let w = self.weights.as_ref().ok_or_else(|| {
            TensorError::InvalidOperation("Model not fitted".into())
        })?;
        let n = x.shape().dim(0)?;
        let p = x.shape().dim(1)?;
        let w_col = w.reshape(vec![p, 1])?;
        let mut pred = x.matmul(&w_col)?;
        if let Some(b) = self.bias {
            pred = pred.add_scalar(b);
        }
        pred.reshape(vec![n])
    }
}

/// Perceptron — a simple linear binary classifier.
///
/// Uses the perceptron update rule: if misclassified, w += η·y·x
pub struct Perceptron<T: Float> {
    pub weights: Option<Tensor<T>>,
    pub bias: Option<T>,
    pub learning_rate: T,
    pub max_iter: usize,
}

impl<T: Float> Perceptron<T> {
    pub fn new(learning_rate: T, max_iter: usize) -> Self {
        Perceptron { weights: None, bias: None, learning_rate, max_iter }
    }

    pub fn fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> TensorResult<()> {
        let n = x.shape().dim(0)?;
        let p = x.shape().dim(1)?;

        let mut w = vec![T::ZERO; p];
        let mut b = T::ZERO;

        for _epoch in 0..self.max_iter {
            let mut errors = 0;
            for i in 0..n {
                // Compute prediction: sign(w·x + b)
                let mut score = b;
                for j in 0..p {
                    score = score + w[j] * x.get(&[i, j])?;
                }
                let pred = if score >= T::ZERO { T::ONE } else { T::ZERO };
                let yi = y.data()[i];

                if (pred - yi).abs() > T::HALF {
                    // Misclassified: update
                    let label = if yi > T::HALF { T::ONE } else { T::NEG_ONE };
                    for j in 0..p {
                        w[j] = w[j] + self.learning_rate * label * x.get(&[i, j])?;
                    }
                    b = b + self.learning_rate * label;
                    errors += 1;
                }
            }
            if errors == 0 { break; }
        }

        self.weights = Some(Tensor::new(w, vec![p])?);
        self.bias = Some(b);
        Ok(())
    }

    pub fn predict(&self, x: &Tensor<T>) -> TensorResult<Tensor<T>> {
        let w = self.weights.as_ref().ok_or_else(|| {
            TensorError::InvalidOperation("Model not fitted".into())
        })?;
        let n = x.shape().dim(0)?;
        let p = x.shape().dim(1)?;
        let mut preds = Vec::with_capacity(n);

        for i in 0..n {
            let mut score = self.bias.unwrap_or(T::ZERO);
            for j in 0..p {
                score = score + w.data()[j] * x.get(&[i, j])?;
            }
            preds.push(if score >= T::ZERO { T::ONE } else { T::ZERO });
        }
        Tensor::new(preds, vec![n])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elastic_net() {
        let x: Tensor<f64> = Tensor::from_vec2d(&[
            vec![1.0, 0.0], vec![2.0, 0.0], vec![3.0, 0.0],
            vec![4.0, 0.0], vec![5.0, 0.0],
        ]).unwrap();
        let y: Tensor<f64> = Tensor::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);

        let mut model = ElasticNet::new(0.01, 0.5, 1000);
        model.fit(&x, &y).unwrap();

        let pred = model.predict(&x).unwrap();
        for i in 0..5 {
            assert!((pred.data()[i] - y.data()[i]).abs() < 1.0,
                "ElasticNet pred {} vs expected {}", pred.data()[i], y.data()[i]);
        }
    }

    #[test]
    fn test_perceptron() {
        let x: Tensor<f64> = Tensor::from_vec2d(&[
            vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0],
        ]).unwrap();
        // OR gate
        let y: Tensor<f64> = Tensor::from_slice(&[0.0, 1.0, 1.0, 1.0]);

        let mut model = Perceptron::new(0.1, 100);
        model.fit(&x, &y).unwrap();

        let pred = model.predict(&x).unwrap();
        assert_eq!(pred.data()[0].round() as i32, 0); // 0 OR 0 = 0
        assert_eq!(pred.data()[3].round() as i32, 1); // 1 OR 1 = 1
    }
}
