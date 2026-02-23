use oxidize_ml_core::Tensor;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs;
use std::path::Path;

/// Serializable model wrapper for saving/loading model weights.
#[derive(Serialize, Deserialize)]
pub struct ModelWeights {
    pub tensors: Vec<(String, Vec<f64>, Vec<usize>)>, // (name, data, shape)
}

impl ModelWeights {
    pub fn new() -> Self {
        ModelWeights {
            tensors: Vec::new(),
        }
    }

    pub fn add(&mut self, name: &str, tensor: &Tensor<f64>) {
        self.tensors.push((
            name.to_string(),
            tensor.data().to_vec(),
            tensor.shape_vec(),
        ));
    }

    pub fn get(&self, name: &str) -> Option<Tensor<f64>> {
        self.tensors.iter().find(|(n, _, _)| n == name).map(|(_, data, shape)| {
            Tensor::new(data.clone(), shape.clone()).expect("load tensor")
        })
    }
}

impl Default for ModelWeights {
    fn default() -> Self {
        Self::new()
    }
}

/// Save model weights to a JSON file.
pub fn save_model(weights: &ModelWeights, path: &str) -> Result<(), Box<dyn Error>> {
    let json = serde_json::to_string_pretty(weights)?;
    fs::write(Path::new(path), json)?;
    Ok(())
}

/// Load model weights from a JSON file.
pub fn load_model(path: &str) -> Result<ModelWeights, Box<dyn Error>> {
    let json = fs::read_to_string(Path::new(path))?;
    let weights: ModelWeights = serde_json::from_str(&json)?;
    Ok(weights)
}
