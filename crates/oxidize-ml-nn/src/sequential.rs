use crate::layers::Layer;
use oxidize_ml_autodiff::Variable;

/// Sequential model — chains layers in order.
pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
}

impl Sequential {
    pub fn new() -> Self {
        Sequential { layers: Vec::new() }
    }

    /// Add a layer to the model.
    pub fn add(mut self, layer: Box<dyn Layer>) -> Self {
        self.layers.push(layer);
        self
    }

    /// Forward pass through all layers.
    pub fn forward(&self, input: &Variable) -> Variable {
        let mut x = input.clone();
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        x
    }

    /// Collect all trainable parameters from all layers.
    pub fn parameters(&self) -> Vec<Variable> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params
    }
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}
