use oxidize_ml_core::{Float, Tensor, TensorError};
use oxidize_ml_core::error::TensorResult;
use oxidize_ml_linalg::{inv, solve};
use rand::distributions::{Distribution, Standard};

/// Ordinary Least Squares linear regression.
///
/// Fits `y = Xw + b` using the normal equation: `w = (XᵀX)⁻¹Xᵀy`.
pub struct LinearRegression<T: Float> {
    pub weights: Option<Tensor<T>>,
    pub bias: Option<T>,
    pub fit_intercept: bool,
}

impl<T: Float> LinearRegression<T>
where
    Standard: Distribution<T>,
{
    pub fn new(fit_intercept: bool) -> Self {
        LinearRegression {
            weights: None,
            bias: None,
            fit_intercept,
        }
    }

    pub fn fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> TensorResult<()> {
        let n = x.shape().dim(0)?;
        let p = x.shape().dim(1)?;

        // Optionally prepend column of ones for intercept
        let x_aug = if self.fit_intercept {
            let ones_col = Tensor::ones(vec![n, 1]);
            Tensor::concatenate(&[&ones_col, x], 1)?
        } else {
            x.clone()
        };

        // Normal equation: w = (XᵀX)⁻¹ Xᵀy
        let xt = x_aug.t()?;
        let xtx = xt.matmul(&x_aug)?;
        let xty = xt.matmul(&y.reshape(vec![n, 1])?)?;
        let xtx_inv = inv(&xtx)?;
        let w = xtx_inv.matmul(&xty)?;

        if self.fit_intercept {
            self.bias = Some(w.get(&[0, 0])?);
            let mut weights = Vec::with_capacity(p);
            for i in 0..p {
                weights.push(w.get(&[i + 1, 0])?);
            }
            self.weights = Some(Tensor::new(weights, vec![p])?);
        } else {
            let mut weights = Vec::with_capacity(p);
            for i in 0..p {
                weights.push(w.get(&[i, 0])?);
            }
            self.weights = Some(Tensor::new(weights, vec![p])?);
            self.bias = None;
        }

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

        // Flatten to 1D
        pred.reshape(vec![n])
    }
}

/// Ridge regression (L2-regularized).
///
/// Fits using: `w = (XᵀX + αI)⁻¹Xᵀy`.
pub struct Ridge<T: Float> {
    pub alpha: T,
    pub weights: Option<Tensor<T>>,
    pub bias: Option<T>,
    pub fit_intercept: bool,
}

impl<T: Float> Ridge<T>
where
    Standard: Distribution<T>,
{
    pub fn new(alpha: T, fit_intercept: bool) -> Self {
        Ridge {
            alpha,
            weights: None,
            bias: None,
            fit_intercept,
        }
    }

    pub fn fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> TensorResult<()> {
        let n = x.shape().dim(0)?;
        let p = x.shape().dim(1)?;

        let x_aug = if self.fit_intercept {
            let ones_col = Tensor::ones(vec![n, 1]);
            Tensor::concatenate(&[&ones_col, x], 1)?
        } else {
            x.clone()
        };

        let dim = x_aug.shape().dim(1)?;
        let xt = x_aug.t()?;
        let xtx = xt.matmul(&x_aug)?;
        let reg = Tensor::<T>::eye(dim).mul_scalar(self.alpha);
        let xtx_reg = xtx.add(&reg)?;
        let xty = xt.matmul(&y.reshape(vec![n, 1])?)?;
        let xtx_inv = inv(&xtx_reg)?;
        let w = xtx_inv.matmul(&xty)?;

        if self.fit_intercept {
            self.bias = Some(w.get(&[0, 0])?);
            let mut weights = Vec::with_capacity(p);
            for i in 0..p {
                weights.push(w.get(&[i + 1, 0])?);
            }
            self.weights = Some(Tensor::new(weights, vec![p])?);
        } else {
            let mut weights = Vec::with_capacity(p);
            for i in 0..p {
                weights.push(w.get(&[i, 0])?);
            }
            self.weights = Some(Tensor::new(weights, vec![p])?);
            self.bias = None;
        }

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

/// Lasso regression (L1-regularized) via coordinate descent.
pub struct Lasso<T: Float> {
    pub alpha: T,
    pub weights: Option<Tensor<T>>,
    pub bias: Option<T>,
    pub max_iter: usize,
    pub tol: T,
}

impl<T: Float> Lasso<T>
where
    Standard: Distribution<T>,
{
    pub fn new(alpha: T, max_iter: usize) -> Self {
        Lasso {
            alpha,
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

                // Soft thresholding
                let mut xj_sq = T::ZERO;
                for i in 0..n {
                    let xij = x.get(&[i, j])?;
                    xj_sq = xj_sq + xij * xij;
                }
                xj_sq = xj_sq / n_t;

                w[j] = if rho > self.alpha {
                    (rho - self.alpha) / xj_sq
                } else if rho < -self.alpha {
                    (rho + self.alpha) / xj_sq
                } else {
                    T::ZERO
                };
            }

            // Check convergence
            let mut max_change = T::ZERO;
            for j in 0..p {
                let change = (w[j] - old_w[j]).abs();
                if change > max_change {
                    max_change = change;
                }
            }
            if max_change < self.tol {
                break;
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_regression() {
        // y = 2*x1 + 3*x2 + 1
        let x: Tensor<f64> = Tensor::from_vec2d(&[
            vec![1.0, 2.0],
            vec![2.0, 1.0],
            vec![3.0, 4.0],
            vec![4.0, 3.0],
            vec![5.0, 5.0],
        ]).unwrap();
        let y: Tensor<f64> = Tensor::from_slice(&[
            2.0*1.0 + 3.0*2.0 + 1.0,  // 9.0
            2.0*2.0 + 3.0*1.0 + 1.0,  // 8.0
            2.0*3.0 + 3.0*4.0 + 1.0,  // 19.0
            2.0*4.0 + 3.0*3.0 + 1.0,  // 18.0
            2.0*5.0 + 3.0*5.0 + 1.0,  // 26.0
        ]);

        let mut model = LinearRegression::new(true);
        model.fit(&x, &y).unwrap();

        let pred = model.predict(&x).unwrap();
        for i in 0..5 {
            assert!((pred.data()[i] - y.data()[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_ridge() {
        let x: Tensor<f64> = Tensor::from_vec2d(&[
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![3.0, 3.0],
            vec![4.0, 4.0],
        ]).unwrap();
        let y: Tensor<f64> = Tensor::from_slice(&[6.0, 11.0, 16.0, 21.0]);

        let mut model = Ridge::new(0.01, true);
        model.fit(&x, &y).unwrap();
        let pred = model.predict(&x).unwrap();
        // With small alpha, should be close to OLS
        for i in 0..4 {
            assert!((pred.data()[i] - y.data()[i]).abs() < 0.5);
        }
    }

    #[test]
    fn test_lasso() {
        let x: Tensor<f64> = Tensor::from_vec2d(&[
            vec![1.0, 0.0],
            vec![2.0, 0.0],
            vec![3.0, 0.0],
        ]).unwrap();
        let y: Tensor<f64> = Tensor::from_slice(&[2.0, 4.0, 6.0]); // y = 2*x1

        let mut model = Lasso::new(0.001, 1000);
        model.fit(&x, &y).unwrap();

        // Irrelevant feature (x2) should be close to zero
        let w = model.weights.as_ref().unwrap();
        assert!(w.data()[1].abs() < 0.1);
    }
}
