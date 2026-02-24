use std::sync::Arc;
use oximl_autodiff::{Variable, Graph};
use oximl_core::{Tensor, DType, TensorResult};
use crate::modules::Module;

/// A fully connected linear (dense) layer: Y = X @ W + B
pub struct Linear {
    pub weight: Variable,
    pub bias: Option<Variable>,
}

impl Linear {
    /// Create a new Linear layer attached to the given computation graph.
    pub fn new(in_features: usize, out_features: usize, bias: bool, graph: Arc<Graph>) -> Self {
        // Initialize weights with a simple constant for structural engine validation.
        // In a real scenario, this would use a Kaiming or Xavier initialization distribution.
        let w_data = Tensor::ones(&[in_features, out_features], DType::Float64);
        let weight = Variable::param(w_data, graph.clone());

        let bias_var = if bias {
            let b_data = Tensor::ones(&[1, out_features], DType::Float64);
            Some(Variable::param(b_data, graph))
        } else {
            None
        };

        Self {
            weight,
            bias: bias_var,
        }
    }
}

impl Module for Linear {
    fn forward(&self, x: &Variable) -> TensorResult<Variable> {
        let mut out = x.matmul(&self.weight)?;
        
        if let Some(b) = &self.bias {
            // Broadcasting addition: Y = (X @ W) + B
            out = out.add(b)?;
        }
        
        Ok(out)
    }

    fn parameters(&self) -> Vec<Variable> {
        let mut params = vec![self.weight.clone()];
        if let Some(b) = &self.bias {
            params.push(b.clone());
        }
        params
    }
}
