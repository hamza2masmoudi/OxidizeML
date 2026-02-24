use oximl_autodiff::Variable;
use oximl_core::TensorResult;

/// The base trait for all neural network modules.
/// It defines the standard forward pass and parameter extraction.
pub trait Module {
    /// Perform the mathematical forward pass of this module.
    fn forward(&self, x: &Variable) -> TensorResult<Variable>;

    /// Return all trainable parameters registered within this module and its submodules.
    /// This allows optimizers to easily collect and update weights.
    fn parameters(&self) -> Vec<Variable>;

    /// Extract the raw inner Tensors and save them to a binary `.oximl` file on disk, mimicking a state_dict.
    fn save<P: AsRef<std::path::Path>>(&self, path: P) -> TensorResult<()> {
        let params = self.parameters();
        let tensors: Vec<oximl_core::Tensor> = params.into_iter().map(|v| v.data.clone()).collect();
        let file = std::fs::File::create(path).map_err(|e| oximl_core::TensorError::InvalidOperation(format!("Save error: {}", e)))?;
        bincode::serialize_into(file, &tensors).map_err(|e| oximl_core::TensorError::InvalidOperation(format!("Bincode error: {}", e)))?;
        Ok(())
    }

    /// Load serialized weights from a binary `.oximl` file sequentially into this module's parameters.
    fn load<P: AsRef<std::path::Path>>(&self, path: P) -> TensorResult<()> {
        let file = std::fs::File::open(path).map_err(|e| oximl_core::TensorError::InvalidOperation(format!("Load error: {}", e)))?;
        let tensors: Vec<oximl_core::Tensor> = bincode::deserialize_from(file)
            .map_err(|e| oximl_core::TensorError::InvalidOperation(format!("Bincode parse error: {}", e)))?;
        
        let mut params = self.parameters();
        if params.len() != tensors.len() {
            return Err(oximl_core::TensorError::InvalidOperation(
                format!("Model architecture mismatch. Expected {} parameter blocks, file contains {}", params.len(), tensors.len())
            ));
        }
        
        // Directly overwrite the memory pointers mimicking PyTorch `.load_state_dict(strict=True)`
        for (param, loaded_tensor) in params.iter_mut().zip(tensors.into_iter()) {
            param.data = loaded_tensor;
        }
        
        Ok(())
    }
}

pub mod loss;
pub use loss::{MSELoss, CrossEntropyLoss};

pub mod nlp;
pub mod cv;
