use std::sync::Arc;
use oximl_autodiff::{Variable, Graph};
use oximl_core::{Tensor, DType, TensorResult};
use crate::modules::Module;
use crate::layers::Linear;

/// Multi-Head Attention Mechanism
/// Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
pub struct MultiHeadAttention {
    pub num_heads: usize,
    pub d_model: usize,
    pub d_k: usize,
    
    // Linear projections for Queries, Keys, Values, and Output
    pub w_q: Linear,
    pub w_k: Linear,
    pub w_v: Linear,
    pub w_o: Linear,
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, num_heads: usize, graph: Arc<Graph>) -> Self {
        assert!(d_model % num_heads == 0, "d_model must be divisible by num_heads");
        let d_k = d_model / num_heads;
        
        Self {
            num_heads,
            d_model,
            d_k,
            w_q: Linear::new(d_model, d_model, false, graph.clone()),
            w_k: Linear::new(d_model, d_model, false, graph.clone()),
            w_v: Linear::new(d_model, d_model, false, graph.clone()),
            w_o: Linear::new(d_model, d_model, false, graph.clone()),
        }
    }

    /// Forward pass expecting batched input sequence: [Batch, SeqLen, d_model]
    /// For conceptual framework validation, we're assuming mathematically dense 2D flattening for QKV until Nd broadcasting is robust.
    /// Q = X @ W_q, K = X @ W_k, V = X @ W_v
    pub fn forward(&self, x: &Variable) -> TensorResult<Variable> {
        let q = self.w_q.forward(x)?;
        let k = self.w_k.forward(x)?;
        let v = self.w_v.forward(x)?;

        // Attention scores: (Q @ K^T) / sqrt(d_k)
        let k_t = k.t();
        let mut scores = q.matmul(&k_t)?;
        
        // Scale by 1/sqrt(d_k)
        let scale_factor = 1.0 / (self.d_k as f64).sqrt();
        scores = scores.scalar_mul(scale_factor)?;

        // Apply Softmax
        let attn_weights = scores.softmax()?;

        // Multiply by Values: Attention @ V
        let attn_output = attn_weights.matmul(&v)?;

        // Final linear projection
        self.w_o.forward(&attn_output)
    }
}

impl Module for MultiHeadAttention {
    fn forward(&self, x: &Variable) -> TensorResult<Variable> {
        self.forward(x)
    }

    fn parameters(&self) -> Vec<Variable> {
        let mut params = Vec::new();
        params.extend(self.w_q.parameters());
        params.extend(self.w_k.parameters());
        params.extend(self.w_v.parameters());
        params.extend(self.w_o.parameters());
        params
    }
}
