use oxidize_ml_core::{Float, Tensor, TensorError};
use oxidize_ml_core::error::TensorResult;
use rand::distributions::{Distribution, Standard};

/// Logistic Regression — binary classification via gradient descent.
pub struct LogisticRegression<T: Float> {
    pub weights: Option<Tensor<T>>,
    pub bias: Option<T>,
    pub learning_rate: T,
    pub max_iter: usize,
    pub tol: T,
}

impl<T: Float> LogisticRegression<T>
where
    Standard: Distribution<T>,
{
    pub fn new(learning_rate: T, max_iter: usize) -> Self {
        LogisticRegression {
            weights: None,
            bias: None,
            learning_rate,
            max_iter,
            tol: T::from_f64(1e-6),
        }
    }

    fn sigmoid_val(x: T) -> T {
        T::ONE / (T::ONE + (-x).exp())
    }

    pub fn fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> TensorResult<()> {
        let n = x.shape().dim(0)?;
        let p = x.shape().dim(1)?;
        let n_t = T::from_usize(n);

        let mut w = vec![T::ZERO; p];
        let mut b = T::ZERO;

        for _iter in 0..self.max_iter {
            let mut dw = vec![T::ZERO; p];
            let mut db = T::ZERO;
            let mut total_loss = T::ZERO;

            for i in 0..n {
                // Compute z = w·x + b
                let mut z = b;
                for j in 0..p {
                    z = z + w[j] * x.get(&[i, j])?;
                }
                let a = Self::sigmoid_val(z); // prediction
                let yi = y.data()[i];
                let error = a - yi;

                for j in 0..p {
                    dw[j] = dw[j] + error * x.get(&[i, j])?;
                }
                db = db + error;

                // BCE loss for monitoring
                let eps = T::from_f64(1e-15);
                let loss = -(yi * (a + eps).ln() + (T::ONE - yi) * (T::ONE - a + eps).ln());
                total_loss = total_loss + loss;
            }

            // Update weights
            let mut max_grad = T::ZERO;
            for j in 0..p {
                let grad = dw[j] / n_t;
                w[j] = w[j] - self.learning_rate * grad;
                if grad.abs() > max_grad {
                    max_grad = grad.abs();
                }
            }
            b = b - self.learning_rate * (db / n_t);

            if max_grad < self.tol {
                break;
            }
        }

        self.weights = Some(Tensor::new(w, vec![p])?);
        self.bias = Some(b);
        Ok(())
    }

    /// Predict probabilities.
    pub fn predict_proba(&self, x: &Tensor<T>) -> TensorResult<Tensor<T>> {
        let w = self.weights.as_ref().ok_or_else(|| {
            TensorError::InvalidOperation("Model not fitted".into())
        })?;
        let n = x.shape().dim(0)?;
        let p = x.shape().dim(1)?;
        let b = self.bias.unwrap_or(T::ZERO);

        let mut proba = Vec::with_capacity(n);
        for i in 0..n {
            let mut z = b;
            for j in 0..p {
                z = z + w.data()[j] * x.get(&[i, j])?;
            }
            proba.push(Self::sigmoid_val(z));
        }

        Tensor::new(proba, vec![n])
    }

    /// Predict class labels (threshold = 0.5).
    pub fn predict(&self, x: &Tensor<T>) -> TensorResult<Tensor<T>> {
        let proba = self.predict_proba(x)?;
        let labels: Vec<T> = proba
            .data()
            .iter()
            .map(|&p| if p >= T::HALF { T::ONE } else { T::ZERO })
            .collect();
        let n = labels.len();
        Tensor::new(labels, vec![n])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logistic_regression() {
        // Linearly separable data
        let x: Tensor<f64> = Tensor::from_vec2d(&[
            vec![0.0, 0.0],
            vec![0.5, 0.5],
            vec![1.0, 1.0],
            vec![5.0, 5.0],
            vec![5.5, 5.5],
            vec![6.0, 6.0],
        ]).unwrap();
        let y: Tensor<f64> = Tensor::from_slice(&[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut model = LogisticRegression::new(0.1, 1000);
        model.fit(&x, &y).unwrap();

        let pred = model.predict(&x).unwrap();
        // Should classify correctly
        for i in 0..3 {
            assert!(pred.data()[i] < 0.5, "Expected 0 at {}", i);
        }
        for i in 3..6 {
            assert!(pred.data()[i] > 0.5, "Expected 1 at {}", i);
        }
    }
}
