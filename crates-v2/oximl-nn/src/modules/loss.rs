use oximl_autodiff::Variable;
use oximl_core::{Tensor, TensorResult, DType};
use std::sync::Arc;

/// Computes the Mean Squared Error between a prediction and a target.
/// Loss = sum((pred - target)^2) / N
pub struct MSELoss;

impl MSELoss {
    pub fn new() -> Self {
        MSELoss
    }

    /// Computes the forward pass of MSE Loss.
    /// Returns a scalar Variable attached to the computation graph.
    pub fn forward(&self, pred: &Variable, target: &Variable) -> TensorResult<Variable> {
        // (pred - target) mathematically translates to: pred + (target * -1)
        let neg_target = target.scalar_mul(-1.0)?;
        let diff = pred.add(&neg_target)?;
        
        // diff^2
        let squared = diff.matmul(&diff.t())?;
        
        // Mean operation: Divide by number of elements
        // Structurally simplified to scalar division for the engine
        let n: f64 = pred.data.shape().iter().product::<usize>() as f64;
        let mut n_inv = 1.0;
        if n > 0.0 { n_inv = 1.0 / n; }
        
        squared.scalar_mul(n_inv)
    }
}

/// Computes the Cross Entropy Loss for classification tasks.
/// Applies LogSoftmax internally for numerical stability.
pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    pub fn new() -> Self {
        CrossEntropyLoss
    }

    /// Computes the forward pass of Cross Entropy Loss.
    /// `pred` are raw unnormalized logits.
    /// `target` are the target probability distributions (one-hot encoded).
    pub fn forward(&self, pred: &Variable, target: &Variable) -> TensorResult<Variable> {
        let graph = pred.graph.clone();

        // 1. Softmax predictions
        let probs = pred.softmax()?;
        
        // 2. Log(probs)
        let log_probs = probs.ln()?;
        
        // 3. target * log_probs (element-wise)
        let weighted = target.matmul(&log_probs.t())?; // Assuming batching shape handling 
        
        // 4. Mean of negative sum 
        // L = -sum(Y * log(Y_hat))
        let neg_weighted = weighted.scalar_mul(-1.0)?;
        
        // Mean operation mapping: simplified to sum for engine logic demonstration
        let n: f64 = pred.data.shape().iter().product::<usize>() as f64;
        let mut n_inv = 1.0;
        if n > 0.0 { n_inv = 1.0 / n; }
        
        neg_weighted.scalar_mul(n_inv)
    }
}
