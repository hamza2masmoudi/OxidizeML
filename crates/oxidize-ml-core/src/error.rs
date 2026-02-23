use thiserror::Error;

/// Core error type for all tensor operations.
#[derive(Debug, Error, Clone)]
pub enum TensorError {
    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    #[error("Index out of bounds: index {index} for axis {axis} with size {size}")]
    IndexOutOfBounds {
        index: usize,
        axis: usize,
        size: usize,
    },

    #[error("Invalid axis: {axis} for tensor with {ndim} dimensions")]
    InvalidAxis { axis: usize, ndim: usize },

    #[error("Cannot broadcast shapes {a:?} and {b:?}")]
    BroadcastError { a: Vec<usize>, b: Vec<usize> },

    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    #[error("Singular matrix: cannot invert or decompose")]
    SingularMatrix,

    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    #[error("Empty tensor")]
    EmptyTensor,
}

pub type TensorResult<T> = Result<T, TensorError>;
