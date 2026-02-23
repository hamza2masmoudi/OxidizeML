use oxidize_ml_core::{Float, Tensor, TensorError};
use oxidize_ml_core::error::TensorResult;
use rand::distributions::{Distribution, Standard};

/// Kernel type for SVM.
#[derive(Debug, Clone)]
pub enum Kernel<T: Float> {
    Linear,
    RBF { gamma: T },
    Polynomial { degree: usize, coef0: T },
}

/// Support Vector Classifier using simplified SMO.
pub struct SVC<T: Float> {
    pub c: T,
    pub kernel: Kernel<T>,
    pub max_iter: usize,
    pub tol: T,
    // Trained parameters
    alphas: Option<Vec<T>>,
    bias: T,
    x_train: Option<Tensor<T>>,
    y_train: Option<Vec<T>>,
}

impl<T: Float> SVC<T>
where
    Standard: Distribution<T>,
{
    pub fn new(c: T, kernel: Kernel<T>, max_iter: usize) -> Self {
        SVC {
            c,
            kernel,
            max_iter,
            tol: T::from_f64(1e-3),
            alphas: None,
            bias: T::ZERO,
            x_train: None,
            y_train: None,
        }
    }

    fn kernel_eval(&self, x: &Tensor<T>, i: usize, j: usize, d: usize) -> TensorResult<T> {
        match &self.kernel {
            Kernel::Linear => {
                let mut dot = T::ZERO;
                for k in 0..d {
                    dot = dot + x.get(&[i, k])? * x.get(&[j, k])?;
                }
                Ok(dot)
            }
            Kernel::RBF { gamma } => {
                let mut sq_dist = T::ZERO;
                for k in 0..d {
                    let diff = x.get(&[i, k])? - x.get(&[j, k])?;
                    sq_dist = sq_dist + diff * diff;
                }
                Ok((-*gamma * sq_dist).exp())
            }
            Kernel::Polynomial { degree, coef0 } => {
                let mut dot = T::ZERO;
                for k in 0..d {
                    dot = dot + x.get(&[i, k])? * x.get(&[j, k])?;
                }
                Ok((dot + *coef0).powi(*degree as i32))
            }
        }
    }

    fn kernel_eval_xy(&self, x1: &Tensor<T>, i: usize, x2: &Tensor<T>, j: usize, d: usize) -> TensorResult<T> {
        match &self.kernel {
            Kernel::Linear => {
                let mut dot = T::ZERO;
                for k in 0..d {
                    dot = dot + x1.get(&[i, k])? * x2.get(&[j, k])?;
                }
                Ok(dot)
            }
            Kernel::RBF { gamma } => {
                let mut sq_dist = T::ZERO;
                for k in 0..d {
                    let diff = x1.get(&[i, k])? - x2.get(&[j, k])?;
                    sq_dist = sq_dist + diff * diff;
                }
                Ok((-*gamma * sq_dist).exp())
            }
            Kernel::Polynomial { degree, coef0 } => {
                let mut dot = T::ZERO;
                for k in 0..d {
                    dot = dot + x1.get(&[i, k])? * x2.get(&[j, k])?;
                }
                Ok((dot + *coef0).powi(*degree as i32))
            }
        }
    }

    /// Fit using simplified SMO algorithm.
    pub fn fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> TensorResult<()> {
        let n = x.shape().dim(0)?;
        let d = x.shape().dim(1)?;

        // Convert labels to +1/-1
        let labels: Vec<T> = y.data().iter().map(|&v| {
            if v > T::ZERO { T::ONE } else { T::NEG_ONE }
        }).collect();

        let mut alphas = vec![T::ZERO; n];
        let mut b = T::ZERO;

        for _pass in 0..self.max_iter {
            let mut num_changed = 0;

            for i in 0..n {
                let mut ei = -labels[i];
                for j in 0..n {
                    ei = ei + alphas[j] * labels[j] * self.kernel_eval(x, j, i, d)?;
                }
                ei = ei + b - labels[i] + labels[i]; // E_i = f(x_i) - y_i
                // Recompute properly
                let mut fi = b;
                for j in 0..n {
                    fi = fi + alphas[j] * labels[j] * self.kernel_eval(x, j, i, d)?;
                }
                ei = fi - labels[i];

                let yi = labels[i];
                if (yi * ei < -self.tol && alphas[i] < self.c)
                    || (yi * ei > self.tol && alphas[i] > T::ZERO)
                {
                    // Select j randomly (simplified SMO picks any j != i)
                    let j = if i == 0 { 1 } else { 0 };
                    let yj = labels[j];

                    let mut fj = b;
                    for k in 0..n {
                        fj = fj + alphas[k] * labels[k] * self.kernel_eval(x, k, j, d)?;
                    }
                    let ej = fj - yj;

                    let ai_old = alphas[i];
                    let aj_old = alphas[j];

                    // Compute bounds
                    let (lo, hi) = if yi != yj {
                        let lo = T::ZERO.max(alphas[j] - alphas[i]);
                        let hi = self.c.min(self.c + alphas[j] - alphas[i]);
                        (lo, hi)
                    } else {
                        let lo = T::ZERO.max(alphas[i] + alphas[j] - self.c);
                        let hi = self.c.min(alphas[i] + alphas[j]);
                        (lo, hi)
                    };

                    if (lo - hi).abs() < T::EPSILON {
                        continue;
                    }

                    let kii = self.kernel_eval(x, i, i, d)?;
                    let kjj = self.kernel_eval(x, j, j, d)?;
                    let kij = self.kernel_eval(x, i, j, d)?;
                    let eta = T::TWO * kij - kii - kjj;

                    if eta >= T::ZERO {
                        continue;
                    }

                    alphas[j] = alphas[j] - yj * (ei - ej) / eta;
                    alphas[j] = alphas[j].max(lo).min(hi);

                    if (alphas[j] - aj_old).abs() < T::from_f64(1e-5) {
                        continue;
                    }

                    alphas[i] = alphas[i] + yi * yj * (aj_old - alphas[j]);

                    let b1 = b - ei - yi * (alphas[i] - ai_old) * kii - yj * (alphas[j] - aj_old) * kij;
                    let b2 = b - ej - yi * (alphas[i] - ai_old) * kij - yj * (alphas[j] - aj_old) * kjj;

                    b = if alphas[i] > T::ZERO && alphas[i] < self.c {
                        b1
                    } else if alphas[j] > T::ZERO && alphas[j] < self.c {
                        b2
                    } else {
                        (b1 + b2) / T::TWO
                    };

                    num_changed += 1;
                }
            }

            if num_changed == 0 {
                break;
            }
        }

        self.alphas = Some(alphas);
        self.bias = b;
        self.x_train = Some(x.clone());
        self.y_train = Some(labels);

        Ok(())
    }

    pub fn predict(&self, x: &Tensor<T>) -> TensorResult<Tensor<T>> {
        let x_train = self.x_train.as_ref().ok_or_else(|| {
            TensorError::InvalidOperation("Model not fitted".into())
        })?;
        let alphas = self.alphas.as_ref().unwrap();
        let labels = self.y_train.as_ref().unwrap();
        let n_test = x.shape().dim(0)?;
        let n_train = x_train.shape().dim(0)?;
        let d = x.shape().dim(1)?;

        let mut predictions = Vec::with_capacity(n_test);
        for i in 0..n_test {
            let mut f = self.bias;
            for j in 0..n_train {
                if alphas[j].abs() > T::EPSILON {
                    f = f + alphas[j] * labels[j] * self.kernel_eval_xy(x_train, j, x, i, d)?;
                }
            }
            predictions.push(if f >= T::ZERO { T::ONE } else { T::ZERO });
        }

        Tensor::new(predictions, vec![n_test])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_svc_linear() {
        let x: Tensor<f64> = Tensor::from_vec2d(&[
            vec![0.0, 0.0], vec![0.5, 0.5], vec![1.0, 1.0],
            vec![5.0, 5.0], vec![5.5, 5.5], vec![6.0, 6.0],
        ]).unwrap();
        let y: Tensor<f64> = Tensor::from_slice(&[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut svc = SVC::new(1.0, Kernel::Linear, 100);
        svc.fit(&x, &y).unwrap();
        let pred = svc.predict(&x).unwrap();

        // Should classify most correctly for linearly separable data
        let correct: usize = pred.data().iter().zip(y.data().iter())
            .filter(|(&p, &t)| (p - t).abs() < 0.5)
            .count();
        assert!(correct >= 4, "SVM classified {} out of 6", correct);
    }
}
