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
}

pub mod loss;
pub use loss::{MSELoss, CrossEntropyLoss};

pub mod nlp;
pub mod cv;
