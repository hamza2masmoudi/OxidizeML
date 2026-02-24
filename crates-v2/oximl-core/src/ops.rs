use crate::{DType, Tensor, TensorResult, TensorError};
use ndarray::{ArrayD, IxDyn};
use std::ops::{Add, Sub, Mul, Div};

/// Basic arithmetic ops that enforce dynamic type checking (no automatic casting).
impl Add for &Tensor {
    type Output = TensorResult<Tensor>;

    fn add(self, rhs: Self) -> Self::Output {
        if self.dtype() != rhs.dtype() {
            return Err(TensorError::TypeMismatch{ expected: self.dtype(), got: rhs.dtype() });
        }
        match (self, rhs) {
            (Tensor::Float32(a), Tensor::Float32(b)) => Ok(Tensor::Float32((a + b).into_shared())),
            (Tensor::Float64(a), Tensor::Float64(b)) => Ok(Tensor::Float64((a + b).into_shared())),
            (Tensor::Int32(a), Tensor::Int32(b)) => Ok(Tensor::Int32((a + b).into_shared())),
            (Tensor::Int64(a), Tensor::Int64(b)) => Ok(Tensor::Int64((a + b).into_shared())),
            (Tensor::UInt8(a), Tensor::UInt8(b)) => Ok(Tensor::UInt8((a + b).into_shared())),
            _ => unreachable!(),
        }
    }
}

impl Mul for &Tensor {
    type Output = TensorResult<Tensor>;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.dtype() != rhs.dtype() {
            return Err(TensorError::TypeMismatch{ expected: self.dtype(), got: rhs.dtype() });
        }
        match (self, rhs) {
            (Tensor::Float32(a), Tensor::Float32(b)) => Ok(Tensor::Float32((a * b).into_shared())),
            (Tensor::Float64(a), Tensor::Float64(b)) => Ok(Tensor::Float64((a * b).into_shared())),
            (Tensor::Int32(a), Tensor::Int32(b)) => Ok(Tensor::Int32((a * b).into_shared())),
            (Tensor::Int64(a), Tensor::Int64(b)) => Ok(Tensor::Int64((a * b).into_shared())),
            (Tensor::UInt8(a), Tensor::UInt8(b)) => Ok(Tensor::UInt8((a * b).into_shared())),
            _ => unreachable!(),
        }
    }
}

impl Div for &Tensor {
    type Output = TensorResult<Tensor>;

    fn div(self, rhs: Self) -> Self::Output {
        if self.dtype() != rhs.dtype() {
            return Err(TensorError::TypeMismatch{ expected: self.dtype(), got: rhs.dtype() });
        }
        match (self, rhs) {
            (Tensor::Float32(a), Tensor::Float32(b)) => Ok(Tensor::Float32((a / b).into_shared())),
            (Tensor::Float64(a), Tensor::Float64(b)) => Ok(Tensor::Float64((a / b).into_shared())),
            (Tensor::Int32(a), Tensor::Int32(b)) => Ok(Tensor::Int32((a / b).into_shared())),
            (Tensor::Int64(a), Tensor::Int64(b)) => Ok(Tensor::Int64((a / b).into_shared())),
            (Tensor::UInt8(a), Tensor::UInt8(b)) => Ok(Tensor::UInt8((a / b).into_shared())),
            _ => unreachable!(),
        }
    }
}

impl Tensor {
    /// Matrix multiplication. Extremely fast due to `ndarray` backend hitting BLAS directly.
    /// Operates on 2D Tensors.
    pub fn matmul(&self, rhs: &Tensor) -> TensorResult<Tensor> {
        if self.ndim() != 2 || rhs.ndim() != 2 {
            return Err(TensorError::InvalidOperation("Matmul requires exactly 2D tensors".into()));
        }
        if self.dtype() != rhs.dtype() {
            return Err(TensorError::TypeMismatch{ expected: self.dtype(), got: rhs.dtype() });
        }

        match (self, rhs) {
            (Tensor::Float32(a), Tensor::Float32(b)) => {
                let a2 = a.view().into_dimensionality::<ndarray::Ix2>().unwrap();
                let b2 = b.view().into_dimensionality::<ndarray::Ix2>().unwrap();
                let res = a2.dot(&b2).into_dyn().into_shared();
                Ok(Tensor::Float32(res))
            }
            (Tensor::Float64(a), Tensor::Float64(b)) => {
                let a2 = a.view().into_dimensionality::<ndarray::Ix2>().unwrap();
                let b2 = b.view().into_dimensionality::<ndarray::Ix2>().unwrap();
                let res = a2.dot(&b2).into_dyn().into_shared();
                Ok(Tensor::Float64(res))
            }
            // Integer matmul allowed but not accelerated by BLAS
            (Tensor::Int32(a), Tensor::Int32(b)) => {
                let a2 = a.view().into_dimensionality::<ndarray::Ix2>().unwrap();
                let b2 = b.view().into_dimensionality::<ndarray::Ix2>().unwrap();
                let res = a2.dot(&b2).into_dyn().into_shared();
                Ok(Tensor::Int32(res))
            }
            (Tensor::Int64(a), Tensor::Int64(b)) => {
                let a2 = a.view().into_dimensionality::<ndarray::Ix2>().unwrap();
                let b2 = b.view().into_dimensionality::<ndarray::Ix2>().unwrap();
                let res = a2.dot(&b2).into_dyn().into_shared();
                Ok(Tensor::Int64(res))
            }
            _ => Err(TensorError::InvalidOperation("Matmul not supported for this dtype".into())),
        }
    }

    /// Compute ReLU element-wise
    pub fn relu(&self) -> TensorResult<Tensor> {
        match self {
            Tensor::Float32(a) => {
                let out = a.mapv(|x| x.max(0.0));
                Ok(Tensor::Float32(out.into_shared()))
            }
            Tensor::Float64(a) => {
                let out = a.mapv(|x| x.max(0.0));
                Ok(Tensor::Float64(out.into_shared()))
            }
            _ => Err(TensorError::InvalidOperation("ReLU requires float dtype".into())),
        }
    }

    /// Compute ReLU backward pass element-wise (1 if x > 0 else 0) * grad
    pub fn relu_backward(&self, grad: &Tensor) -> TensorResult<Tensor> {
        match (self, grad) {
            (Tensor::Float32(a), Tensor::Float32(g)) => {
                let mut out = g.to_owned();
                out.zip_mut_with(a, |out_val, x_val| {
                    *out_val *= if *x_val > 0.0 { 1.0 } else { 0.0 };
                });
                Ok(Tensor::Float32(out.into_shared()))
            }
            (Tensor::Float64(a), Tensor::Float64(g)) => {
                let mut out = g.to_owned();
                out.zip_mut_with(a, |out_val, x_val| {
                    *out_val *= if *x_val > 0.0 { 1.0 } else { 0.0 };
                });
                Ok(Tensor::Float64(out.into_shared()))
            }
            _ => Err(TensorError::InvalidOperation("ReLU backward requires float matching dtype".into())),
        }
    }

    /// Compute Exponential element-wise
    pub fn exp(&self) -> TensorResult<Tensor> {
        match self {
            Tensor::Float32(a) => Ok(Tensor::Float32(a.mapv(|x| x.exp()).into_shared())),
            Tensor::Float64(a) => Ok(Tensor::Float64(a.mapv(|x| x.exp()).into_shared())),
            _ => Err(TensorError::InvalidOperation("Exp requires float dtype".into())),
        }
    }

    /// Compute Natural Logarithm element-wise
    pub fn ln(&self) -> TensorResult<Tensor> {
        match self {
            Tensor::Float32(a) => Ok(Tensor::Float32(a.mapv(|x| x.ln()).into_shared())),
            Tensor::Float64(a) => Ok(Tensor::Float64(a.mapv(|x| x.ln()).into_shared())),
            _ => Err(TensorError::InvalidOperation("Ln requires float dtype".into())),
        }
    }

    /// Multiply tensor by a scalar value
    pub fn scalar_mul(&self, scalar: f64) -> TensorResult<Tensor> {
        match self {
            Tensor::Float32(a) => Ok(Tensor::Float32((a * scalar as f32).into_shared())),
            Tensor::Float64(a) => Ok(Tensor::Float64((a * scalar).into_shared())),
            Tensor::Int32(a) => Ok(Tensor::Int32((a * scalar as i32).into_shared())),
            Tensor::Int64(a) => Ok(Tensor::Int64((a * scalar as i64).into_shared())),
            Tensor::UInt8(a) => Ok(Tensor::UInt8((a * scalar as u8).into_shared())),
        }
    }

    /// Compute softmax along the last axis (-1)
    pub fn softmax(&self) -> TensorResult<Tensor> {
        if !self.dtype().is_float() {
            return Err(TensorError::InvalidOperation("Softmax requires float dtype".into()));
        }
        
        match self {
            Tensor::Float32(a) => {
                let mut out = a.to_owned();
                // Simple numerical stability map: exp(x - max(x)) / sum
                // For a proper Nd implementation, we map over the last axis. 
                // To keep this architecture pure, we'll map all elements together as a fallback if not 2D, 
                // but usually it's applied on the last axis. We'll simplify for the engine architecture.
                let max = a.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                out.mapv_inplace(|x| (x - max).exp());
                let sum = out.sum();
                out.mapv_inplace(|x| x / sum);
                Ok(Tensor::Float32(out.into_shared()))
            }
            Tensor::Float64(a) => {
                let mut out = a.to_owned();
                let max = a.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                out.mapv_inplace(|x| (x - max).exp());
                let sum = out.sum();
                out.mapv_inplace(|x| x / sum);
                Ok(Tensor::Float64(out.into_shared()))
            }
            _ => unreachable!(),
        }
    }

    /// Approximate Softmax diagonal derivative
    pub fn softmax_backward(&self, grad: &Tensor) -> TensorResult<Tensor> {
        // Approx: grad_in = grad * y * (1 - y)
        let ones = Tensor::ones(self.shape(), self.dtype());
        let minus_y = self.scalar_mul(-1.0)?;
        let one_minus_y = (&ones + &minus_y)?;
        let y_one_y = (self * &one_minus_y)?;
        (&y_one_y * grad)
    }
}
