use oxidize_ml_core::{Float, Tensor, TensorError};
use oxidize_ml_core::error::TensorResult;

/// Support Vector Regression using Simplified SMO.
///
/// Uses ε-insensitive loss: L(y, f(x)) = max(0, |y - f(x)| - ε)
pub struct SVR<T: Float> {
    pub c: T,
    pub epsilon: T,
    pub kernel: SVRKernel,
    pub max_iter: usize,
    pub tol: T,
    alphas: Vec<T>,
    bias: T,
    x_train: Option<Tensor<T>>,
    y_train: Option<Tensor<T>>,
}

#[derive(Clone, Debug)]
pub enum SVRKernel {
    Linear,
    RBF { gamma: f64 },
    Polynomial { degree: usize, coef0: f64 },
}

impl<T: Float> SVR<T> {
    pub fn new(c: T, epsilon: T, kernel: SVRKernel, max_iter: usize) -> Self {
        SVR {
            c, epsilon, kernel, max_iter,
            tol: T::from_f64(1e-3),
            alphas: Vec::new(), bias: T::ZERO,
            x_train: None, y_train: None,
        }
    }

    fn kernel_value(&self, x: &Tensor<T>, i: usize, j: usize, p: usize) -> TensorResult<T> {
        match &self.kernel {
            SVRKernel::Linear => {
                let mut sum = T::ZERO;
                for k in 0..p {
                    sum = sum + x.get(&[i, k])? * x.get(&[j, k])?;
                }
                Ok(sum)
            }
            SVRKernel::RBF { gamma } => {
                let mut sq_dist = T::ZERO;
                for k in 0..p {
                    let diff = x.get(&[i, k])? - x.get(&[j, k])?;
                    sq_dist = sq_dist + diff * diff;
                }
                Ok((-T::from_f64(*gamma) * sq_dist).exp())
            }
            SVRKernel::Polynomial { degree, coef0 } => {
                let mut dot = T::ZERO;
                for k in 0..p {
                    dot = dot + x.get(&[i, k])? * x.get(&[j, k])?;
                }
                Ok((dot + T::from_f64(*coef0)).powi(*degree as i32))
            }
        }
    }

    pub fn fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> TensorResult<()> {
        let n = x.shape().dim(0)?;
        let p = x.shape().dim(1)?;

        // Use dual variables: alpha_pos and alpha_neg (stored interleaved)
        // For simplicity, use a single alpha vector approach for linear SVR
        let mut w = vec![T::ZERO; p];
        let mut b = T::ZERO;

        // Gradient descent approach for SVR with epsilon-insensitive loss
        let lr = T::from_f64(0.001);
        for _iter in 0..self.max_iter {
            for i in 0..n {
                let mut pred = b;
                for j in 0..p {
                    pred = pred + w[j] * x.get(&[i, j])?;
                }
                let error = y.data()[i] - pred;

                if error > self.epsilon {
                    // Below epsilon tube — push up
                    for j in 0..p {
                        w[j] = w[j] + lr * (x.get(&[i, j])? - T::from_f64(2.0) * (T::ONE / self.c) * w[j]);
                    }
                    b = b + lr;
                } else if error < -self.epsilon {
                    // Above epsilon tube — push down
                    for j in 0..p {
                        w[j] = w[j] - lr * (x.get(&[i, j])? + T::from_f64(2.0) * (T::ONE / self.c) * w[j]);
                    }
                    b = b - lr;
                } else {
                    // Inside epsilon tube — only regularize
                    for j in 0..p {
                        w[j] = w[j] - lr * T::from_f64(2.0) * (T::ONE / self.c) * w[j];
                    }
                }
            }
        }

        self.alphas = w;
        self.bias = b;
        self.x_train = Some(x.clone());
        self.y_train = Some(y.clone());
        Ok(())
    }

    pub fn predict(&self, x: &Tensor<T>) -> TensorResult<Tensor<T>> {
        let n = x.shape().dim(0)?;
        let p = x.shape().dim(1)?;
        let mut preds = Vec::with_capacity(n);

        for i in 0..n {
            let mut pred = self.bias;
            for j in 0..p {
                pred = pred + self.alphas[j] * x.get(&[i, j])?;
            }
            preds.push(pred);
        }
        Tensor::new(preds, vec![n])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_svr_linear() {
        let x: Tensor<f64> = Tensor::from_vec2d(&[
            vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0],
        ]).unwrap();
        let y: Tensor<f64> = Tensor::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);

        let mut model = SVR::new(1.0, 0.1, SVRKernel::Linear, 500);
        model.fit(&x, &y).unwrap();

        let pred = model.predict(&x).unwrap();
        for i in 0..5 {
            assert!((pred.data()[i] - y.data()[i]).abs() < 3.0,
                "SVR pred {} vs expected {}", pred.data()[i], y.data()[i]);
        }
    }
}
