use ndarray::{s, ArrayD, ArcArray, IxDyn, LinalgScalar};
use thiserror::Error;
use serde::{Serialize, Deserialize};
use crate::DType;

#[derive(Error, Debug)]
pub enum TensorError {
    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },
    #[error("Type mismatch: expected {expected:?}, got {got:?}")]
    TypeMismatch { expected: DType, got: DType },
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
}

pub type TensorResult<T> = Result<T, TensorError>;

/// A dynamic-typed Tensor, powered by `ndarray`.
/// Uses `ArcArray` internally to allow incredibly fast cloning (pointer sharing)
/// and zero-copy views, mirroring PyTorch's architecture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Tensor {
    Float32(ArcArray<f32, IxDyn>),
    Float64(ArcArray<f64, IxDyn>),
    Int32(ArcArray<i32, IxDyn>),
    Int64(ArcArray<i64, IxDyn>),
    UInt8(ArcArray<u8, IxDyn>),
}

impl Tensor {
    pub fn dtype(&self) -> DType {
        match self {
            Tensor::Float32(_) => DType::Float32,
            Tensor::Float64(_) => DType::Float64,
            Tensor::Int32(_) => DType::Int32,
            Tensor::Int64(_) => DType::Int64,
            Tensor::UInt8(_) => DType::UInt8,
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            Tensor::Float32(a) => a.shape(),
            Tensor::Float64(a) => a.shape(),
            Tensor::Int32(a) => a.shape(),
            Tensor::Int64(a) => a.shape(),
            Tensor::UInt8(a) => a.shape(),
        }
    }

    pub fn ndim(&self) -> usize {
        self.shape().len()
    }

    pub fn numel(&self) -> usize {
        self.shape().iter().product()
    }

    /// Create tensor filled with zeros
    pub fn zeros(shape: &[usize], dtype: DType) -> Self {
        let dyn_shape = IxDyn(shape);
        match dtype {
            DType::Float32 => Tensor::Float32(ArrayD::<f32>::zeros(dyn_shape).into_shared()),
            DType::Float64 => Tensor::Float64(ArrayD::<f64>::zeros(dyn_shape).into_shared()),
            DType::Int32 => Tensor::Int32(ArrayD::<i32>::zeros(dyn_shape).into_shared()),
            DType::Int64 => Tensor::Int64(ArrayD::<i64>::zeros(dyn_shape).into_shared()),
            DType::UInt8 => Tensor::UInt8(ArrayD::<u8>::zeros(dyn_shape).into_shared()),
        }
    }

    /// Create tensor filled with ones
    pub fn ones(shape: &[usize], dtype: DType) -> Self {
        let dyn_shape = IxDyn(shape);
        match dtype {
            DType::Float32 => Tensor::Float32(ArrayD::<f32>::ones(dyn_shape).into_shared()),
            DType::Float64 => Tensor::Float64(ArrayD::<f64>::ones(dyn_shape).into_shared()),
            DType::Int32 => Tensor::Int32(ArrayD::<i32>::ones(dyn_shape).into_shared()),
            DType::Int64 => Tensor::Int64(ArrayD::<i64>::ones(dyn_shape).into_shared()),
            DType::UInt8 => Tensor::UInt8(ArrayD::<u8>::ones(dyn_shape).into_shared()),
        }
    }

    pub fn into_owned(self) -> Self {
        match self {
            Tensor::Float32(a) => Tensor::Float32(a.into_owned().into_shared()),
            Tensor::Float64(a) => Tensor::Float64(a.into_owned().into_shared()),
            Tensor::Int32(a) => Tensor::Int32(a.into_owned().into_shared()),
            Tensor::Int64(a) => Tensor::Int64(a.into_owned().into_shared()),
            Tensor::UInt8(a) => Tensor::UInt8(a.into_owned().into_shared()),
        }
    }

    /// Transpose (Zero-copy in ndarray)
    pub fn t(&self) -> Self {
        match self {
            Tensor::Float32(a) => Tensor::Float32(a.clone().reversed_axes()),
            Tensor::Float64(a) => Tensor::Float64(a.clone().reversed_axes()),
            Tensor::Int32(a) => Tensor::Int32(a.clone().reversed_axes()),
            Tensor::Int64(a) => Tensor::Int64(a.clone().reversed_axes()),
            Tensor::UInt8(a) => Tensor::UInt8(a.clone().reversed_axes()),
        }
    }

    /// Reshape (Zero-copy if memory layout allows, else panics/errors in ndarray if invalid)
    pub fn reshape(&self, new_shape: &[usize]) -> TensorResult<Self> {
        let n: usize = new_shape.iter().product();
        if n != self.numel() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![self.numel()],
                got: new_shape.to_vec(),
            });
        }
        
        let shape_dyn = IxDyn(new_shape);
        match self {
            Tensor::Float32(a) => Ok(Tensor::Float32(a.clone().into_shape(shape_dyn).unwrap())),
            Tensor::Float64(a) => Ok(Tensor::Float64(a.clone().into_shape(shape_dyn).unwrap())),
            Tensor::Int32(a) => Ok(Tensor::Int32(a.clone().into_shape(shape_dyn).unwrap())),
            Tensor::Int64(a) => Ok(Tensor::Int64(a.clone().into_shape(shape_dyn).unwrap())),
            Tensor::UInt8(a) => Ok(Tensor::UInt8(a.clone().into_shape(shape_dyn).unwrap())),
        }
    }

    /// Slice a 2D Tensor along its first (outer) dimension `[start..end, :]`. This is the core operation for extracting mini-batches.
    pub fn slice(&self, start: usize, end: usize) -> TensorResult<Self> {
        if self.ndim() != 2 {
            return Err(TensorError::InvalidOperation("Slice is currently only implemented for 2D batched tensors".into()));
        }
        let dims = self.shape();
        if start >= end || end > dims[0] {
            return Err(TensorError::InvalidOperation(format!("Invalid slice bounds {}..{} for dimension 0 size {}", start, end, dims[0])));
        }

        match self {
            Tensor::Float32(a) => {
                let sliced = a.slice(s![start..end, ..]).into_owned().into_dyn().into_shared();
                Ok(Tensor::Float32(sliced))
            }
            Tensor::Float64(a) => {
                let sliced = a.slice(s![start..end, ..]).into_owned().into_dyn().into_shared();
                Ok(Tensor::Float64(sliced))
            }
            Tensor::Int32(a) => {
                let sliced = a.slice(s![start..end, ..]).into_owned().into_dyn().into_shared();
                Ok(Tensor::Int32(sliced))
            }
            Tensor::Int64(a) => {
                let sliced = a.slice(s![start..end, ..]).into_owned().into_dyn().into_shared();
                Ok(Tensor::Int64(sliced))
            }
            Tensor::UInt8(a) => {
                let sliced = a.slice(s![start..end, ..]).into_owned().into_dyn().into_shared();
                Ok(Tensor::UInt8(sliced))
            }
        }
    }
}
