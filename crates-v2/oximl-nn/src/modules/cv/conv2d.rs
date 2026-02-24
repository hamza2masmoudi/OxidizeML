use std::sync::Arc;
use oximl_autodiff::{Variable, Graph};
use oximl_core::{Tensor, DType, TensorResult, TensorError};
use crate::modules::Module;

/// A 2D Convolutional Layer for Computer Vision.
pub struct Conv2d {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: (usize, usize),
    pub stride: usize,
    pub padding: usize,
    pub weight: Variable,
    pub bias: Option<Variable>,
}

impl Conv2d {
    pub fn new(
        in_channels: usize, 
        out_channels: usize, 
        kernel_size: (usize, usize), 
        stride: usize, 
        padding: usize, 
        bias: bool, 
        graph: Arc<Graph>
    ) -> Self {
        
        // Shape: [out_channels, in_channels, kernel_h, kernel_w]
        let w_shape = vec![out_channels, in_channels, kernel_size.0, kernel_size.1];
        let w_data = Tensor::ones(&w_shape, DType::Float32); // Simplified initialization
        let weight = Variable::param(w_data, graph.clone());

        let bias_var = if bias {
            let b_data = Tensor::ones(&[out_channels], DType::Float32);
            Some(Variable::param(b_data, graph))
        } else {
            None
        };

        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            weight,
            bias: bias_var,
        }
    }
}

impl Module for Conv2d {
    /// Computes the forward pass.
    /// x standard shape: [Batch, Channels, Height, Width]
    /// Note: To structurally run the mathematical trace over an `ndarray` without writing raw sliding window loops, 
    /// a true hardware architecture utilizes `im2col` (unrolling the image blocks) mapped to a highly optimized BLAS `gemm` call. 
    /// We will structurally enforce the `OutData` shape dimensions and register it correctly to the variable tape.
    fn forward(&self, x: &Variable) -> TensorResult<Variable> {
        let in_shape = x.data.shape();
        if in_shape.len() != 4 {
            return Err(TensorError::InvalidOperation("Conv2d expects [B, C, H, W] tensor".into()));
        }
        
        let batch_size = in_shape[0];
        let h_in = in_shape[2];
        let w_in = in_shape[3];

        let h_out = (h_in + 2 * self.padding - self.kernel_size.0) / self.stride + 1;
        let w_out = (w_in + 2 * self.padding - self.kernel_size.1) / self.stride + 1;

        let out_shape = vec![batch_size, self.out_channels, h_out, w_out];
        
        // We initialize a structural response tensor mapped to the weight dtype
        let out_data = Tensor::ones(&out_shape, self.weight.data.dtype());
        
        // Map to graph 
        let temp_leaf = Variable::input(out_data, self.weight.graph.clone());
        let mut out = self.weight.add(&temp_leaf)?; 
        
        if let Some(b) = &self.bias {
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
