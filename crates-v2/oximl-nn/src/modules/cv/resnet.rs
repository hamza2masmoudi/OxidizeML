use std::sync::Arc;
use oximl_autodiff::{Variable, Graph};
use oximl_core::TensorResult;
use crate::modules::Module;
use super::conv2d::Conv2d;

/// Structural ResNet block implementing residual spatial pathways.
pub struct ResNetBlock {
    pub conv1: Conv2d,
    pub conv2: Conv2d,
    // Note: BatchNorm requires detailed moving-average spatial statistics mapping. We rely on the raw conv path here.
}

impl ResNetBlock {
    pub fn new(channels: usize, stride: usize, graph: Arc<Graph>) -> Self {
        Self {
            conv1: Conv2d::new(channels, channels, (3, 3), stride, 1, false, graph.clone()),
            conv2: Conv2d::new(channels, channels, (3, 3), 1, 1, false, graph),
        }
    }

    pub fn forward(&self, x: &Variable) -> TensorResult<Variable> {
        // Residual path mapping:  F(x) + x
        
        let mut out = self.conv1.forward(x)?;
        out = out.relu()?;
        out = self.conv2.forward(&out)?;
        
        // Add skip connection
        out = out.add(x)?;
        out.relu()
    }
}

impl Module for ResNetBlock {
    fn forward(&self, x: &Variable) -> TensorResult<Variable> {
        self.forward(x)
    }

    fn parameters(&self) -> Vec<Variable> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.extend(self.conv2.parameters());
        params
    }
}
