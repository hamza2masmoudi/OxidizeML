use std::sync::Arc;
use oximl_autodiff::{Variable, Graph};
use oximl_core::TensorResult;
use crate::modules::Module;
use crate::layers::Linear;
use super::attention::MultiHeadAttention;

/// A standard Transformer structural block.
/// Combines Multi-Head Attention with a Feed Forward Network and Residual Connections.
pub struct TransformerBlock {
    pub attention: MultiHeadAttention,
    pub ff1: Linear,
    pub ff2: Linear,
    // Note: LayerNorm requires mean/std ops which we can stub or approximate
}

impl TransformerBlock {
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize, graph: Arc<Graph>) -> Self {
        Self {
            attention: MultiHeadAttention::new(d_model, num_heads, graph.clone()),
            ff1: Linear::new(d_model, d_ff, true, graph.clone()),
            ff2: Linear::new(d_ff, d_model, true, graph),
        }
    }

    /// Forward pass of the block.
    /// x shape: [Batch, SeqLen, d_model]
    pub fn forward(&self, x: &Variable) -> TensorResult<Variable> {
        // 1. Multi-Head Attention
        let attn_out = self.attention.forward(x)?;
        
        // 2. Residual Connection (Add & Norm)
        // Stubbing LayerNorm for engine validation, keeping residual add.
        let out1 = x.add(&attn_out)?;
        
        // 3. Feed Forward (Linear -> ReLU -> Linear)
        let ff1_out = self.ff1.forward(&out1)?;
        let relu_out = ff1_out.relu()?;
        let ff2_out = self.ff2.forward(&relu_out)?;
        
        // 4. Residual Connection (Add & Norm)
        let out2 = out1.add(&ff2_out)?;
        
        Ok(out2)
    }
}

impl Module for TransformerBlock {
    fn forward(&self, x: &Variable) -> TensorResult<Variable> {
        self.forward(x)
    }

    fn parameters(&self) -> Vec<Variable> {
        let mut params = Vec::new();
        params.extend(self.attention.parameters());
        params.extend(self.ff1.parameters());
        params.extend(self.ff2.parameters());
        params
    }
}
