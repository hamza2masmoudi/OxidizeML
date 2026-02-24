use oximl_core::TensorResult;
use oximl_autodiff::Variable;

/// Stochastic Gradient Descent
pub struct SGD {
    pub lr: f64,
}

impl SGD {
    pub fn new(lr: f64) -> Self {
        SGD { lr }
    }

    /// Update standard stochastic gradient descent weights.
    pub fn step(&mut self, params: &mut [Variable]) -> TensorResult<()> {
        for param in params.iter_mut() {
            // Apply gradient step if gradients were mathematically derived during `.backward()`
            if let Some(grad) = param.grad() {
                // delta = -lr * grad
                let step_grad = grad.scalar_mul(-self.lr)?;
                // param -= delta (using addition of negatives)
                param.data = (&param.data + &step_grad)?;
            }
        }
        Ok(())
    }
}
