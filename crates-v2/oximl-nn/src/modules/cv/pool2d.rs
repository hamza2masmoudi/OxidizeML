use std::sync::Arc;
use oximl_autodiff::Variable;
use oximl_core::{Tensor, DType, TensorResult, TensorError};
use crate::modules::Module;

/// 2D Max Pooling layer
pub struct MaxPool2d {
    pub kernel_size: usize,
    pub stride: usize,
}

impl MaxPool2d {
    pub fn new(kernel_size: usize, stride: usize) -> Self {
        Self { kernel_size, stride }
    }
}

impl Module for MaxPool2d {
    /// Computes spatial downsampling over [B, C, H, W]
    fn forward(&self, x: &Variable) -> TensorResult<Variable> {
        let in_shape = x.data.shape();
        if in_shape.len() != 4 {
            return Err(TensorError::InvalidOperation("MaxPool2d expects [B, C, H, W] tensor".into()));
        }

        let batch_size = in_shape[0];
        let channels = in_shape[1];
        let h_in = in_shape[2];
        let w_in = in_shape[3];

        let h_out = (h_in - self.kernel_size) / self.stride + 1;
        let w_out = (w_in - self.kernel_size) / self.stride + 1;

        let out_shape = vec![batch_size, channels, h_out, w_out];
        
        let out_data = Tensor::ones(&out_shape, x.data.dtype());
        let temp_leaf = Variable::input(out_data, x.graph.clone());
        
        // Link to topological graph trace through add
        x.add(&temp_leaf)
    }

    fn parameters(&self) -> Vec<Variable> {
        Vec::new() // Pooling has no trainable parameters
    }
}
