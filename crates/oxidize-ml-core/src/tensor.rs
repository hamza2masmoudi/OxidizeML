use crate::dtype::Float;
use crate::error::{TensorError, TensorResult};
use crate::shape::Shape;

use rand::distributions::{Distribution, Standard};
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops;

/// N-dimensional tensor — the fundamental data structure of OxidizeML.
///
/// Stores data in a flat contiguous `Vec<T>` with row-major (C-order) layout.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "T: Float")]
pub struct Tensor<T: Float> {
    data: Vec<T>,
    shape: Shape,
}

// ─── Construction ───────────────────────────────────────────────────────────

impl<T: Float> Tensor<T> {
    /// Create a tensor from raw data and shape.
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> TensorResult<Self> {
        let s = Shape::new(shape);
        if data.len() != s.numel() {
            return Err(TensorError::ShapeMismatch {
                expected: s.to_vec(),
                got: vec![data.len()],
            });
        }
        Ok(Tensor { data, shape: s })
    }

    /// Create a tensor filled with zeros.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let s = Shape::new(shape);
        Tensor {
            data: vec![T::ZERO; s.numel()],
            shape: s,
        }
    }

    /// Create a tensor filled with ones.
    pub fn ones(shape: Vec<usize>) -> Self {
        let s = Shape::new(shape);
        Tensor {
            data: vec![T::ONE; s.numel()],
            shape: s,
        }
    }

    /// Create a tensor filled with a constant value.
    pub fn full(shape: Vec<usize>, value: T) -> Self {
        let s = Shape::new(shape);
        Tensor {
            data: vec![value; s.numel()],
            shape: s,
        }
    }

    /// Create a scalar tensor (0-d).
    pub fn scalar(value: T) -> Self {
        Tensor {
            data: vec![value],
            shape: Shape::scalar(),
        }
    }

    /// Create a 1-D tensor from a slice.
    pub fn from_slice(data: &[T]) -> Self {
        Tensor {
            data: data.to_vec(),
            shape: Shape::new(vec![data.len()]),
        }
    }

    /// Create a 2-D tensor from a nested slice.
    pub fn from_vec2d(data: &[Vec<T>]) -> TensorResult<Self> {
        if data.is_empty() {
            return Ok(Tensor::zeros(vec![0, 0]));
        }
        let rows = data.len();
        let cols = data[0].len();
        for row in data {
            if row.len() != cols {
                return Err(TensorError::InvalidOperation(
                    "All rows must have the same number of columns".to_string(),
                ));
            }
        }
        let flat: Vec<T> = data.iter().flat_map(|r| r.iter().copied()).collect();
        Tensor::new(flat, vec![rows, cols])
    }

    /// Identity matrix of size n×n.
    pub fn eye(n: usize) -> Self {
        let mut data = vec![T::ZERO; n * n];
        for i in 0..n {
            data[i * n + i] = T::ONE;
        }
        Tensor {
            data,
            shape: Shape::new(vec![n, n]),
        }
    }

    /// Linearly spaced values from `start` to `end` (inclusive), `n` points.
    pub fn linspace(start: T, end: T, n: usize) -> Self {
        if n == 0 {
            return Tensor {
                data: vec![],
                shape: Shape::new(vec![0]),
            };
        }
        if n == 1 {
            return Tensor::from_slice(&[start]);
        }
        let step = (end - start) / T::from_usize(n - 1);
        let data: Vec<T> = (0..n).map(|i| start + step * T::from_usize(i)).collect();
        Tensor {
            data,
            shape: Shape::new(vec![n]),
        }
    }

    /// Range of values from `start` to `end` (exclusive) with step.
    pub fn arange(start: T, end: T, step: T) -> Self {
        let mut data = Vec::new();
        let mut val = start;
        if step.to_f64() > 0.0 {
            while val.to_f64() < end.to_f64() {
                data.push(val);
                val = val + step;
            }
        } else if step.to_f64() < 0.0 {
            while val.to_f64() > end.to_f64() {
                data.push(val);
                val = val + step;
            }
        }
        let len = data.len();
        Tensor {
            data,
            shape: Shape::new(vec![len]),
        }
    }

    /// Random tensor with uniform distribution in [0, 1).
    pub fn rand(shape: Vec<usize>, seed: Option<u64>) -> Self {
        let s = Shape::new(shape);
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        let data: Vec<T> = (0..s.numel())
            .map(|_| T::from_f64(rand::Rng::gen::<f64>(&mut rng)))
            .collect();
        Tensor { data, shape: s }
    }

    /// Random tensor with standard normal distribution (approximate via Box-Muller).
    pub fn randn(shape: Vec<usize>, seed: Option<u64>) -> Self {
        let s = Shape::new(shape);
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        let n = s.numel();
        let mut data = Vec::with_capacity(n);

        // Box-Muller transform
        let mut i = 0;
        while i < n {
            let u1: f64 = rand::Rng::gen::<f64>(&mut rng).max(1e-10);
            let u2: f64 = rand::Rng::gen::<f64>(&mut rng);
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            data.push(T::from_f64(r * theta.cos()));
            if i + 1 < n {
                data.push(T::from_f64(r * theta.sin()));
            }
            i += 2;
        }
        data.truncate(n);
        Tensor { data, shape: s }
    }

    // ─── Accessors ──────────────────────────────────────────────────────────

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn shape_vec(&self) -> Vec<usize> {
        self.shape.to_vec()
    }

    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn data(&self) -> &[T] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    pub fn into_data(self) -> Vec<T> {
        self.data
    }

    pub fn is_scalar(&self) -> bool {
        self.shape.ndim() == 0
    }

    /// Get a single element (scalar value).
    pub fn item(&self) -> TensorResult<T> {
        if self.data.len() != 1 {
            return Err(TensorError::InvalidOperation(format!(
                "item() requires exactly 1 element, got {}",
                self.data.len()
            )));
        }
        Ok(self.data[0])
    }

    /// Multi-dimensional indexing: compute flat offset from indices.
    pub fn get(&self, indices: &[usize]) -> TensorResult<T> {
        let strides = self.shape.strides();
        if indices.len() != self.ndim() {
            return Err(TensorError::DimensionMismatch(format!(
                "Expected {} indices, got {}",
                self.ndim(),
                indices.len()
            )));
        }
        let mut offset = 0;
        for (i, &idx) in indices.iter().enumerate() {
            let dim_size = self.shape.dim(i)?;
            if idx >= dim_size {
                return Err(TensorError::IndexOutOfBounds {
                    index: idx,
                    axis: i,
                    size: dim_size,
                });
            }
            offset += idx * strides[i];
        }
        Ok(self.data[offset])
    }

    /// Set a single element.
    pub fn set(&mut self, indices: &[usize], value: T) -> TensorResult<()> {
        let strides = self.shape.strides();
        if indices.len() != self.ndim() {
            return Err(TensorError::DimensionMismatch(format!(
                "Expected {} indices, got {}",
                self.ndim(),
                indices.len()
            )));
        }
        let mut offset = 0;
        for (i, &idx) in indices.iter().enumerate() {
            let dim_size = self.shape.dim(i)?;
            if idx >= dim_size {
                return Err(TensorError::IndexOutOfBounds {
                    index: idx,
                    axis: i,
                    size: dim_size,
                });
            }
            offset += idx * strides[i];
        }
        self.data[offset] = value;
        Ok(())
    }

    /// Extract a row from a 2D tensor.
    pub fn row(&self, i: usize) -> TensorResult<Tensor<T>> {
        if self.ndim() != 2 {
            return Err(TensorError::InvalidOperation(
                "row() requires a 2D tensor".to_string(),
            ));
        }
        let cols = self.shape.dim(1)?;
        let start = i * cols;
        let end = start + cols;
        if end > self.data.len() {
            return Err(TensorError::IndexOutOfBounds {
                index: i,
                axis: 0,
                size: self.shape.dim(0)?,
            });
        }
        Ok(Tensor {
            data: self.data[start..end].to_vec(),
            shape: Shape::new(vec![cols]),
        })
    }

    /// Extract a column from a 2D tensor.
    pub fn col(&self, j: usize) -> TensorResult<Tensor<T>> {
        if self.ndim() != 2 {
            return Err(TensorError::InvalidOperation(
                "col() requires a 2D tensor".to_string(),
            ));
        }
        let rows = self.shape.dim(0)?;
        let cols = self.shape.dim(1)?;
        if j >= cols {
            return Err(TensorError::IndexOutOfBounds {
                index: j,
                axis: 1,
                size: cols,
            });
        }
        let data: Vec<T> = (0..rows).map(|i| self.data[i * cols + j]).collect();
        Ok(Tensor {
            data,
            shape: Shape::new(vec![rows]),
        })
    }

    // ─── Shape Manipulation ─────────────────────────────────────────────────

    /// Reshape the tensor (data remains the same, only shape changes).
    pub fn reshape(&self, new_shape: Vec<usize>) -> TensorResult<Tensor<T>> {
        let ns = Shape::new(new_shape);
        if self.numel() != ns.numel() {
            return Err(TensorError::ShapeMismatch {
                expected: ns.to_vec(),
                got: self.shape_vec(),
            });
        }
        Ok(Tensor {
            data: self.data.clone(),
            shape: ns,
        })
    }

    /// Flatten to 1-D.
    pub fn flatten(&self) -> Tensor<T> {
        Tensor {
            data: self.data.clone(),
            shape: Shape::new(vec![self.numel()]),
        }
    }

    /// Transpose the last two dimensions.
    pub fn t(&self) -> TensorResult<Tensor<T>> {
        if self.ndim() < 2 {
            return Err(TensorError::InvalidOperation(
                "Cannot transpose tensor with fewer than 2 dimensions".to_string(),
            ));
        }
        let dims = self.shape.dims();
        let rows = dims[dims.len() - 2];
        let cols = dims[dims.len() - 1];

        if self.ndim() == 2 {
            let mut data = vec![T::ZERO; self.numel()];
            for i in 0..rows {
                for j in 0..cols {
                    data[j * rows + i] = self.data[i * cols + j];
                }
            }
            return Ok(Tensor {
                data,
                shape: self.shape.transposed()?,
            });
        }

        // Batched transpose: transpose last two dims
        let batch_size: usize = dims[..dims.len() - 2].iter().product();
        let mat_size = rows * cols;
        let mut data = vec![T::ZERO; self.numel()];
        for b in 0..batch_size {
            let offset = b * mat_size;
            for i in 0..rows {
                for j in 0..cols {
                    data[offset + j * rows + i] = self.data[offset + i * cols + j];
                }
            }
        }
        Ok(Tensor {
            data,
            shape: self.shape.transposed()?,
        })
    }

    /// Add a dimension of size 1 at the given axis.
    pub fn unsqueeze(&self, axis: usize) -> TensorResult<Tensor<T>> {
        let mut dims = self.shape.to_vec();
        if axis > dims.len() {
            return Err(TensorError::InvalidAxis {
                axis,
                ndim: self.ndim(),
            });
        }
        dims.insert(axis, 1);
        Ok(Tensor {
            data: self.data.clone(),
            shape: Shape::new(dims),
        })
    }

    /// Remove all dimensions of size 1.
    pub fn squeeze(&self) -> Tensor<T> {
        let dims: Vec<usize> = self.shape.dims().iter().copied().filter(|&d| d != 1).collect();
        let dims = if dims.is_empty() { vec![1] } else { dims };
        Tensor {
            data: self.data.clone(),
            shape: Shape::new(dims),
        }
    }

    /// Concatenate tensors along axis.
    pub fn concatenate(tensors: &[&Tensor<T>], axis: usize) -> TensorResult<Tensor<T>> {
        if tensors.is_empty() {
            return Err(TensorError::EmptyTensor);
        }
        let ndim = tensors[0].ndim();
        if axis >= ndim {
            return Err(TensorError::InvalidAxis { axis, ndim });
        }

        // Verify all shapes match except along axis
        let ref_shape = tensors[0].shape_vec();
        for t in &tensors[1..] {
            if t.ndim() != ndim {
                return Err(TensorError::DimensionMismatch(
                    "All tensors must have the same number of dimensions".to_string(),
                ));
            }
            for (i, (&a, &b)) in ref_shape.iter().zip(t.shape_vec().iter()).enumerate() {
                if i != axis && a != b {
                    return Err(TensorError::ShapeMismatch {
                        expected: ref_shape.clone(),
                        got: t.shape_vec(),
                    });
                }
            }
        }

        // Simple case: axis 0
        if axis == 0 {
            let mut data = Vec::new();
            let mut total_rows = 0usize;
            for t in tensors {
                data.extend_from_slice(&t.data);
                total_rows += t.shape.dim(0)?;
            }
            let mut new_shape = ref_shape;
            new_shape[0] = total_rows;
            return Tensor::new(data, new_shape);
        }

        // General case
        let outer: usize = ref_shape[..axis].iter().product();
        let inner: usize = ref_shape[axis + 1..].iter().product();

        let mut new_axis_size = 0usize;
        for t in tensors {
            new_axis_size += t.shape.dim(axis)?;
        }

        let mut data = Vec::with_capacity(outer * new_axis_size * inner);
        for o in 0..outer {
            for t in tensors {
                let t_axis = t.shape.dim(axis)?;
                let t_stride: usize = t_axis * inner;
                let start = o * t_stride;
                data.extend_from_slice(&t.data[start..start + t_stride]);
            }
        }

        let mut new_shape = ref_shape;
        new_shape[axis] = new_axis_size;
        Tensor::new(data, new_shape)
    }

    // ─── Element-wise Unary Operations ──────────────────────────────────────

    pub fn apply<F: Fn(T) -> T>(&self, f: F) -> Tensor<T> {
        Tensor {
            data: self.data.iter().map(|&x| f(x)).collect(),
            shape: self.shape.clone(),
        }
    }

    pub fn apply_mut<F: Fn(T) -> T>(&mut self, f: F) {
        for x in self.data.iter_mut() {
            *x = f(*x);
        }
    }

    pub fn abs(&self) -> Tensor<T> { self.apply(T::abs) }
    pub fn exp(&self) -> Tensor<T> { self.apply(T::exp) }
    pub fn ln(&self) -> Tensor<T> { self.apply(T::ln) }
    pub fn sqrt(&self) -> Tensor<T> { self.apply(T::sqrt) }
    pub fn sin(&self) -> Tensor<T> { self.apply(T::sin) }
    pub fn cos(&self) -> Tensor<T> { self.apply(T::cos) }
    pub fn tanh_elem(&self) -> Tensor<T> { self.apply(T::tanh) }
    pub fn signum(&self) -> Tensor<T> { self.apply(T::signum) }
    pub fn recip(&self) -> Tensor<T> { self.apply(T::recip) }
    pub fn floor(&self) -> Tensor<T> { self.apply(T::floor) }
    pub fn ceil(&self) -> Tensor<T> { self.apply(T::ceil) }
    pub fn round(&self) -> Tensor<T> { self.apply(T::round) }

    pub fn powf(&self, n: T) -> Tensor<T> {
        self.apply(|x| x.powf(n))
    }

    pub fn powi(&self, n: i32) -> Tensor<T> {
        self.apply(|x| x.powi(n))
    }

    /// Clamp values to [min, max].
    pub fn clamp(&self, min: T, max: T) -> Tensor<T> {
        self.apply(|x| x.max(min).min(max))
    }

    /// ReLU activation: max(0, x).
    pub fn relu(&self) -> Tensor<T> {
        self.apply(|x| x.max(T::ZERO))
    }

    /// Sigmoid activation: 1 / (1 + exp(-x)).
    pub fn sigmoid(&self) -> Tensor<T> {
        self.apply(|x| T::ONE / (T::ONE + (-x).exp()))
    }

    // ─── Scalar Operations ──────────────────────────────────────────────────

    pub fn add_scalar(&self, s: T) -> Tensor<T> { self.apply(|x| x + s) }
    pub fn sub_scalar(&self, s: T) -> Tensor<T> { self.apply(|x| x - s) }
    pub fn mul_scalar(&self, s: T) -> Tensor<T> { self.apply(|x| x * s) }
    pub fn div_scalar(&self, s: T) -> Tensor<T> { self.apply(|x| x / s) }

    // ─── Element-wise Binary Operations (with broadcasting) ─────────────────

    fn broadcast_binary_op<F: Fn(T, T) -> T>(
        &self,
        other: &Tensor<T>,
        op: F,
    ) -> TensorResult<Tensor<T>> {
        // Fast path: same shape
        if self.shape == other.shape {
            let data: Vec<T> = self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| op(a, b))
                .collect();
            return Ok(Tensor {
                data,
                shape: self.shape.clone(),
            });
        }

        let out_shape = Shape::broadcast_shape(&self.shape, &other.shape)?;
        let out_numel = out_shape.numel();
        let out_strides = out_shape.strides();
        let a_strides = self.shape.strides();
        let b_strides = other.shape.strides();
        let a_dims = self.shape.dims();
        let b_dims = other.shape.dims();
        let out_dims = out_shape.dims();
        let ndim = out_dims.len();

        let mut data = Vec::with_capacity(out_numel);

        for flat_idx in 0..out_numel {
            // Convert flat index to multi-dim index
            let mut remaining = flat_idx;
            let mut a_offset = 0usize;
            let mut b_offset = 0usize;

            for d in 0..ndim {
                let idx = remaining / out_strides[d];
                remaining %= out_strides[d];

                let a_dim_offset = ndim as isize - a_dims.len() as isize;
                let a_d = d as isize - a_dim_offset;
                if a_d >= 0 {
                    let a_d = a_d as usize;
                    if a_dims[a_d] > 1 {
                        a_offset += idx * a_strides[a_d];
                    }
                }

                let b_dim_offset = ndim as isize - b_dims.len() as isize;
                let b_d = d as isize - b_dim_offset;
                if b_d >= 0 {
                    let b_d = b_d as usize;
                    if b_dims[b_d] > 1 {
                        b_offset += idx * b_strides[b_d];
                    }
                }
            }

            data.push(op(self.data[a_offset], other.data[b_offset]));
        }

        Ok(Tensor {
            data,
            shape: out_shape,
        })
    }

    pub fn add(&self, other: &Tensor<T>) -> TensorResult<Tensor<T>> {
        self.broadcast_binary_op(other, |a, b| a + b)
    }

    pub fn sub(&self, other: &Tensor<T>) -> TensorResult<Tensor<T>> {
        self.broadcast_binary_op(other, |a, b| a - b)
    }

    pub fn mul(&self, other: &Tensor<T>) -> TensorResult<Tensor<T>> {
        self.broadcast_binary_op(other, |a, b| a * b)
    }

    pub fn div(&self, other: &Tensor<T>) -> TensorResult<Tensor<T>> {
        self.broadcast_binary_op(other, |a, b| a / b)
    }

    // ─── Reduction Operations ───────────────────────────────────────────────

    /// Sum of all elements.
    pub fn sum_all(&self) -> T {
        self.data.iter().copied().sum()
    }

    /// Mean of all elements.
    pub fn mean_all(&self) -> T {
        self.sum_all() / T::from_usize(self.numel())
    }

    /// Max of all elements.
    pub fn max_all(&self) -> TensorResult<T> {
        self.data
            .iter()
            .copied()
            .reduce(T::max)
            .ok_or(TensorError::EmptyTensor)
    }

    /// Min of all elements.
    pub fn min_all(&self) -> TensorResult<T> {
        self.data
            .iter()
            .copied()
            .reduce(T::min)
            .ok_or(TensorError::EmptyTensor)
    }

    /// Argmax of all elements (flat index).
    pub fn argmax_all(&self) -> TensorResult<usize> {
        if self.data.is_empty() {
            return Err(TensorError::EmptyTensor);
        }
        let mut best = 0;
        for (i, &v) in self.data.iter().enumerate() {
            if v > self.data[best] {
                best = i;
            }
        }
        Ok(best)
    }

    /// Argmin of all elements (flat index).
    pub fn argmin_all(&self) -> TensorResult<usize> {
        if self.data.is_empty() {
            return Err(TensorError::EmptyTensor);
        }
        let mut best = 0;
        for (i, &v) in self.data.iter().enumerate() {
            if v < self.data[best] {
                best = i;
            }
        }
        Ok(best)
    }

    /// Sum along a specific axis, collapsing that dimension.
    pub fn sum_axis(&self, axis: usize) -> TensorResult<Tensor<T>> {
        let dims = self.shape.dims();
        if axis >= dims.len() {
            return Err(TensorError::InvalidAxis {
                axis,
                ndim: self.ndim(),
            });
        }

        let outer: usize = dims[..axis].iter().product();
        let axis_size = dims[axis];
        let inner: usize = dims[axis + 1..].iter().product();

        let mut new_dims: Vec<usize> = dims.to_vec();
        new_dims.remove(axis);
        if new_dims.is_empty() {
            new_dims.push(1);
        }

        let mut result = vec![T::ZERO; outer * inner];
        for o in 0..outer {
            for a in 0..axis_size {
                for i in 0..inner {
                    let src = o * axis_size * inner + a * inner + i;
                    let dst = o * inner + i;
                    result[dst] = result[dst] + self.data[src];
                }
            }
        }

        Tensor::new(result, new_dims)
    }

    /// Mean along a specific axis.
    pub fn mean_axis(&self, axis: usize) -> TensorResult<Tensor<T>> {
        let axis_size = self.shape.dim(axis)?;
        let s = self.sum_axis(axis)?;
        Ok(s.div_scalar(T::from_usize(axis_size)))
    }

    /// Argmax along axis — returns tensor of indices.
    pub fn argmax_axis(&self, axis: usize) -> TensorResult<Tensor<T>> {
        let dims = self.shape.dims();
        if axis >= dims.len() {
            return Err(TensorError::InvalidAxis {
                axis,
                ndim: self.ndim(),
            });
        }

        let outer: usize = dims[..axis].iter().product();
        let axis_size = dims[axis];
        let inner: usize = dims[axis + 1..].iter().product();

        let mut new_dims: Vec<usize> = dims.to_vec();
        new_dims.remove(axis);
        if new_dims.is_empty() {
            new_dims.push(1);
        }

        let mut result = vec![T::ZERO; outer * inner];
        for o in 0..outer {
            for i in 0..inner {
                let mut best_idx = 0usize;
                let mut best_val = self.data[o * axis_size * inner + i];
                for a in 1..axis_size {
                    let v = self.data[o * axis_size * inner + a * inner + i];
                    if v > best_val {
                        best_val = v;
                        best_idx = a;
                    }
                }
                result[o * inner + i] = T::from_usize(best_idx);
            }
        }

        Tensor::new(result, new_dims)
    }

    /// Variance along axis (population variance).
    pub fn var_axis(&self, axis: usize) -> TensorResult<Tensor<T>> {
        let mean = self.mean_axis(axis)?;
        // Expand mean back along the axis
        let dims = self.shape.dims();
        let outer: usize = dims[..axis].iter().product();
        let axis_size = dims[axis];
        let inner: usize = dims[axis + 1..].iter().product();

        let mut result = vec![T::ZERO; outer * inner];
        for o in 0..outer {
            for a in 0..axis_size {
                for i in 0..inner {
                    let src = o * axis_size * inner + a * inner + i;
                    let mu = mean.data[o * inner + i];
                    let diff = self.data[src] - mu;
                    result[o * inner + i] = result[o * inner + i] + diff * diff;
                }
            }
        }
        for v in result.iter_mut() {
            *v = *v / T::from_usize(axis_size);
        }

        let mut new_dims: Vec<usize> = dims.to_vec();
        new_dims.remove(axis);
        if new_dims.is_empty() {
            new_dims.push(1);
        }
        Tensor::new(result, new_dims)
    }

    /// Standard deviation along axis.
    pub fn std_axis(&self, axis: usize) -> TensorResult<Tensor<T>> {
        let var = self.var_axis(axis)?;
        Ok(var.sqrt())
    }

    // ─── Slicing ────────────────────────────────────────────────────────────

    /// Slice rows from a 2D tensor: returns rows[start..end].
    pub fn slice_rows(&self, start: usize, end: usize) -> TensorResult<Tensor<T>> {
        if self.ndim() != 2 {
            return Err(TensorError::InvalidOperation(
                "slice_rows requires a 2D tensor".to_string(),
            ));
        }
        let rows = self.shape.dim(0)?;
        let cols = self.shape.dim(1)?;
        if start >= rows || end > rows || start >= end {
            return Err(TensorError::IndexOutOfBounds {
                index: end,
                axis: 0,
                size: rows,
            });
        }
        let data = self.data[start * cols..end * cols].to_vec();
        Tensor::new(data, vec![end - start, cols])
    }

    /// Slice columns from a 2D tensor.
    pub fn slice_cols(&self, start: usize, end: usize) -> TensorResult<Tensor<T>> {
        if self.ndim() != 2 {
            return Err(TensorError::InvalidOperation(
                "slice_cols requires a 2D tensor".to_string(),
            ));
        }
        let rows = self.shape.dim(0)?;
        let cols = self.shape.dim(1)?;
        if start >= cols || end > cols || start >= end {
            return Err(TensorError::IndexOutOfBounds {
                index: end,
                axis: 1,
                size: cols,
            });
        }
        let new_cols = end - start;
        let mut data = Vec::with_capacity(rows * new_cols);
        for i in 0..rows {
            for j in start..end {
                data.push(self.data[i * cols + j]);
            }
        }
        Tensor::new(data, vec![rows, new_cols])
    }

    // ─── Comparisons ────────────────────────────────────────────────────────

    /// Element-wise comparison, returns tensor of 1.0 / 0.0.
    pub fn eq_elem(&self, other: &Tensor<T>) -> TensorResult<Tensor<T>> {
        self.broadcast_binary_op(other, |a, b| {
            if (a - b).abs() < T::EPSILON {
                T::ONE
            } else {
                T::ZERO
            }
        })
    }

    pub fn gt(&self, other: &Tensor<T>) -> TensorResult<Tensor<T>> {
        self.broadcast_binary_op(other, |a, b| if a > b { T::ONE } else { T::ZERO })
    }

    pub fn lt(&self, other: &Tensor<T>) -> TensorResult<Tensor<T>> {
        self.broadcast_binary_op(other, |a, b| if a < b { T::ONE } else { T::ZERO })
    }

    // ─── Dot Product / Matrix Multiply ──────────────────────────────────────

    /// Dot product of two 1-D tensors.
    pub fn dot(&self, other: &Tensor<T>) -> TensorResult<T> {
        if self.ndim() != 1 || other.ndim() != 1 {
            return Err(TensorError::InvalidOperation(
                "dot requires two 1D tensors".to_string(),
            ));
        }
        if self.numel() != other.numel() {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape_vec(),
                got: other.shape_vec(),
            });
        }
        Ok(self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a * b)
            .sum())
    }

    /// Matrix multiply: supports 2D×2D and batched.
    pub fn matmul(&self, other: &Tensor<T>) -> TensorResult<Tensor<T>> {
        if self.ndim() < 2 || other.ndim() < 2 {
            return Err(TensorError::InvalidOperation(
                "matmul requires tensors with at least 2 dimensions".to_string(),
            ));
        }

        let a_dims = self.shape.dims();
        let b_dims = other.shape.dims();
        let m = a_dims[a_dims.len() - 2];
        let k = a_dims[a_dims.len() - 1];
        let k2 = b_dims[b_dims.len() - 2];
        let n = b_dims[b_dims.len() - 1];

        if k != k2 {
            return Err(TensorError::DimensionMismatch(format!(
                "matmul: inner dimensions must match, got {} and {}",
                k, k2
            )));
        }

        if self.ndim() == 2 && other.ndim() == 2 {
            // Standard 2D matmul
            let mut data = vec![T::ZERO; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut sum = T::ZERO;
                    for p in 0..k {
                        sum = sum + self.data[i * k + p] * other.data[p * n + j];
                    }
                    data[i * n + j] = sum;
                }
            }
            return Tensor::new(data, vec![m, n]);
        }

        // Batched matmul
        let batch_a: usize = a_dims[..a_dims.len() - 2].iter().product();
        let batch_b: usize = b_dims[..b_dims.len() - 2].iter().product();
        let batch = batch_a.max(batch_b);

        let mut data = vec![T::ZERO; batch * m * n];
        let a_mat = m * k;
        let b_mat = k * n;
        let c_mat = m * n;

        for b_idx in 0..batch {
            let a_off = if batch_a == 1 { 0 } else { b_idx * a_mat };
            let b_off = if batch_b == 1 { 0 } else { b_idx * b_mat };
            let c_off = b_idx * c_mat;

            for i in 0..m {
                for j in 0..n {
                    let mut sum = T::ZERO;
                    for p in 0..k {
                        sum = sum
                            + self.data[a_off + i * k + p]
                                * other.data[b_off + p * n + j];
                    }
                    data[c_off + i * n + j] = sum;
                }
            }
        }

        let mut out_shape = if batch_a > batch_b {
            a_dims[..a_dims.len() - 2].to_vec()
        } else {
            b_dims[..b_dims.len() - 2].to_vec()
        };
        out_shape.push(m);
        out_shape.push(n);
        Tensor::new(data, out_shape)
    }

    // ─── Softmax ────────────────────────────────────────────────────────────

    /// Softmax along the last axis.
    pub fn softmax(&self) -> TensorResult<Tensor<T>> {
        if self.ndim() == 0 {
            return Ok(Tensor::scalar(T::ONE));
        }
        let last_axis = self.ndim() - 1;
        let dims = self.shape.dims();
        let outer: usize = dims[..last_axis].iter().product();
        let axis_size = dims[last_axis];

        let mut data = self.data.clone();
        for o in 0..outer {
            let start = o * axis_size;
            let end = start + axis_size;
            let slice = &data[start..end];

            // Numerical stability: subtract max
            let max_val = slice.iter().copied().reduce(T::max).unwrap_or(T::ZERO);
            let mut sum = T::ZERO;
            for i in start..end {
                data[i] = (data[i] - max_val).exp();
                sum = sum + data[i];
            }
            for i in start..end {
                data[i] = data[i] / sum;
            }
        }

        Ok(Tensor {
            data,
            shape: self.shape.clone(),
        })
    }

    /// Log-softmax along the last axis.
    pub fn log_softmax(&self) -> TensorResult<Tensor<T>> {
        let sm = self.softmax()?;
        Ok(sm.ln())
    }

    /// Softmax along an arbitrary axis.
    pub fn softmax_axis(&self, axis: usize) -> TensorResult<Tensor<T>> {
        let dims = self.shape.dims();
        if axis >= dims.len() {
            return Err(TensorError::InvalidAxis { axis, ndim: self.ndim() });
        }
        let outer: usize = dims[..axis].iter().product();
        let axis_size = dims[axis];
        let inner: usize = dims[axis + 1..].iter().product();
        let mut data = self.data.clone();

        for o in 0..outer {
            for i in 0..inner {
                // Find max for numerical stability
                let mut max_val = T::NEG_INFINITY;
                for a in 0..axis_size {
                    let idx = o * axis_size * inner + a * inner + i;
                    if data[idx] > max_val { max_val = data[idx]; }
                }
                // Exponentiate & sum
                let mut sum = T::ZERO;
                for a in 0..axis_size {
                    let idx = o * axis_size * inner + a * inner + i;
                    data[idx] = (data[idx] - max_val).exp();
                    sum = sum + data[idx];
                }
                // Normalize
                for a in 0..axis_size {
                    let idx = o * axis_size * inner + a * inner + i;
                    data[idx] = data[idx] / sum;
                }
            }
        }
        Ok(Tensor { data, shape: self.shape.clone() })
    }

    /// Element-wise negation.
    pub fn neg(&self) -> Tensor<T> {
        self.apply(|x| -x)
    }

    /// Stack a list of tensors along a new axis.
    pub fn stack(tensors: &[&Tensor<T>], axis: usize) -> TensorResult<Tensor<T>> {
        if tensors.is_empty() {
            return Err(TensorError::EmptyTensor);
        }
        let ref_shape = tensors[0].shape_vec();
        for t in &tensors[1..] {
            if t.shape_vec() != ref_shape {
                return Err(TensorError::ShapeMismatch { expected: ref_shape.clone(), got: t.shape_vec() });
            }
        }
        // Unsqueeze each tensor at the given axis, then concatenate
        let unsqueezed: Vec<Tensor<T>> = tensors.iter()
            .map(|t| t.unsqueeze(axis))
            .collect::<Result<Vec<_>, _>>()?;
        let refs: Vec<&Tensor<T>> = unsqueezed.iter().collect();
        Tensor::concatenate(&refs, axis)
    }

    /// Conditional selection: where mask is 1, return self; else return other.
    pub fn where_cond(&self, mask: &Tensor<T>, other: &Tensor<T>) -> TensorResult<Tensor<T>> {
        if self.shape != mask.shape || self.shape != other.shape {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape_vec(), got: mask.shape_vec(),
            });
        }
        let data: Vec<T> = mask.data.iter().zip(self.data.iter().zip(other.data.iter()))
            .map(|(&m, (&s, &o))| if m > T::ZERO { s } else { o })
            .collect();
        Ok(Tensor { data, shape: self.shape.clone() })
    }

    /// Top-k values and indices along the last axis of a 1D/2D tensor.
    pub fn topk(&self, k: usize) -> TensorResult<(Tensor<T>, Vec<Vec<usize>>)> {
        if self.ndim() == 1 {
            let mut indexed: Vec<(T, usize)> = self.data.iter().copied()
                .enumerate().map(|(i, v)| (v, i)).collect();
            indexed.sort_by(|a, b| b.0.to_f64().partial_cmp(&a.0.to_f64()).unwrap());
            let k = k.min(indexed.len());
            let values: Vec<T> = indexed[..k].iter().map(|&(v, _)| v).collect();
            let indices: Vec<usize> = indexed[..k].iter().map(|&(_, i)| i).collect();
            Ok((Tensor::new(values, vec![k])?, vec![indices]))
        } else if self.ndim() == 2 {
            let rows = self.shape.dim(0)?;
            let cols = self.shape.dim(1)?;
            let k = k.min(cols);
            let mut all_values = Vec::with_capacity(rows * k);
            let mut all_indices = Vec::with_capacity(rows);
            for r in 0..rows {
                let mut indexed: Vec<(T, usize)> = (0..cols)
                    .map(|c| (self.data[r * cols + c], c)).collect();
                indexed.sort_by(|a, b| b.0.to_f64().partial_cmp(&a.0.to_f64()).unwrap());
                all_values.extend(indexed[..k].iter().map(|&(v, _)| v));
                all_indices.push(indexed[..k].iter().map(|&(_, i)| i).collect());
            }
            Ok((Tensor::new(all_values, vec![rows, k])?, all_indices))
        } else {
            Err(TensorError::InvalidOperation("topk supports 1D/2D tensors".into()))
        }
    }

    /// L2 (Frobenius) norm of the entire tensor.
    pub fn norm(&self) -> T {
        let mut sum = T::ZERO;
        for &v in &self.data {
            sum = sum + v * v;
        }
        sum.sqrt()
    }

    /// L1 norm.
    pub fn norm_l1(&self) -> T {
        self.data.iter().map(|&v| v.abs()).sum()
    }

    /// Clip (clamp) values in-place.
    pub fn clip_mut(&mut self, min: T, max: T) {
        for v in self.data.iter_mut() {
            *v = v.max(min).min(max);
        }
    }

    /// One-hot encode a 1D integer tensor.
    pub fn one_hot(&self, n_classes: usize) -> TensorResult<Tensor<T>> {
        if self.ndim() != 1 {
            return Err(TensorError::InvalidOperation("one_hot requires 1D tensor".into()));
        }
        let n = self.numel();
        let mut data = vec![T::ZERO; n * n_classes];
        for i in 0..n {
            let cls = self.data[i].to_f64().round() as usize;
            if cls < n_classes {
                data[i * n_classes + cls] = T::ONE;
            }
        }
        Tensor::new(data, vec![n, n_classes])
    }

    /// Outer product of two 1D tensors: result[i,j] = a[i] * b[j].
    pub fn outer(&self, other: &Tensor<T>) -> TensorResult<Tensor<T>> {
        if self.ndim() != 1 || other.ndim() != 1 {
            return Err(TensorError::InvalidOperation("outer requires two 1D tensors".into()));
        }
        let m = self.numel();
        let n = other.numel();
        let mut data = Vec::with_capacity(m * n);
        for i in 0..m {
            for j in 0..n {
                data.push(self.data[i] * other.data[j]);
            }
        }
        Tensor::new(data, vec![m, n])
    }

    /// Product of all elements.
    pub fn prod_all(&self) -> T {
        self.data.iter().copied().fold(T::ONE, |acc, x| acc * x)
    }

    /// Max along axis, returning tensor with that dimension removed.
    pub fn max_axis(&self, axis: usize) -> TensorResult<Tensor<T>> {
        let dims = self.shape.dims();
        if axis >= dims.len() {
            return Err(TensorError::InvalidAxis { axis, ndim: self.ndim() });
        }
        let outer: usize = dims[..axis].iter().product();
        let axis_size = dims[axis];
        let inner: usize = dims[axis + 1..].iter().product();
        let mut result = vec![T::NEG_INFINITY; outer * inner];
        for o in 0..outer {
            for a in 0..axis_size {
                for i in 0..inner {
                    let src = o * axis_size * inner + a * inner + i;
                    let dst = o * inner + i;
                    if self.data[src] > result[dst] {
                        result[dst] = self.data[src];
                    }
                }
            }
        }
        let mut new_dims: Vec<usize> = dims.to_vec();
        new_dims.remove(axis);
        if new_dims.is_empty() { new_dims.push(1); }
        Tensor::new(result, new_dims)
    }

    /// Min along axis.
    pub fn min_axis(&self, axis: usize) -> TensorResult<Tensor<T>> {
        let dims = self.shape.dims();
        if axis >= dims.len() {
            return Err(TensorError::InvalidAxis { axis, ndim: self.ndim() });
        }
        let outer: usize = dims[..axis].iter().product();
        let axis_size = dims[axis];
        let inner: usize = dims[axis + 1..].iter().product();
        let mut result = vec![T::INFINITY; outer * inner];
        for o in 0..outer {
            for a in 0..axis_size {
                for i in 0..inner {
                    let src = o * axis_size * inner + a * inner + i;
                    let dst = o * inner + i;
                    if self.data[src] < result[dst] {
                        result[dst] = self.data[src];
                    }
                }
            }
        }
        let mut new_dims: Vec<usize> = dims.to_vec();
        new_dims.remove(axis);
        if new_dims.is_empty() { new_dims.push(1); }
        Tensor::new(result, new_dims)
    }

    /// Check if any element is NaN.
    pub fn has_nan(&self) -> bool {
        self.data.iter().any(|v| v.to_f64().is_nan())
    }

    /// Replace NaN with a given value.
    pub fn nan_to_num(&self, replacement: T) -> Tensor<T> {
        self.apply(|x| if x.to_f64().is_nan() { replacement } else { x })
    }

    /// Repeat tensor along axis.
    pub fn repeat_axis(&self, axis: usize, n: usize) -> TensorResult<Tensor<T>> {
        let refs: Vec<&Tensor<T>> = vec![self; n];
        Tensor::concatenate(&refs, axis)
    }

    /// Cumulative sum along axis 0 for 1D/2D tensors.
    pub fn cumsum(&self) -> Tensor<T> {
        let mut data = self.data.clone();
        if self.ndim() == 1 {
            for i in 1..data.len() {
                data[i] = data[i] + data[i - 1];
            }
        } else if self.ndim() == 2 {
            let cols = self.shape.dims()[1];
            for j in 0..cols {
                for i in 1..self.shape.dims()[0] {
                    data[i * cols + j] = data[i * cols + j] + data[(i - 1) * cols + j];
                }
            }
        }
        Tensor { data, shape: self.shape.clone() }
    }
}

// ─── Operator Overloads ─────────────────────────────────────────────────────

impl<T: Float> ops::Neg for &Tensor<T>
where
    Standard: Distribution<T>,
{
    type Output = Tensor<T>;
    fn neg(self) -> Tensor<T> {
        self.apply(|x| -x)
    }
}

impl<T: Float> ops::Add for &Tensor<T>
where
    Standard: Distribution<T>,
{
    type Output = TensorResult<Tensor<T>>;
    fn add(self, rhs: Self) -> TensorResult<Tensor<T>> {
        self.add(rhs)
    }
}

impl<T: Float> ops::Sub for &Tensor<T>
where
    Standard: Distribution<T>,
{
    type Output = TensorResult<Tensor<T>>;
    fn sub(self, rhs: Self) -> TensorResult<Tensor<T>> {
        Tensor::sub(self, rhs)
    }
}

impl<T: Float> ops::Mul for &Tensor<T>
where
    Standard: Distribution<T>,
{
    type Output = TensorResult<Tensor<T>>;
    fn mul(self, rhs: Self) -> TensorResult<Tensor<T>> {
        Tensor::mul(self, rhs)
    }
}

impl<T: Float> PartialEq for Tensor<T> {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.data == other.data
    }
}

// ─── Display ────────────────────────────────────────────────────────────────

impl<T: Float> fmt::Display for Tensor<T>
where
    Standard: Distribution<T>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_scalar() {
            return write!(f, "tensor({})", self.data[0]);
        }
        if self.ndim() == 1 {
            write!(f, "tensor([")?;
            for (i, v) in self.data.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                if i > 6 {
                    write!(f, "...")?;
                    break;
                }
                write!(f, "{:.4}", v)?;
            }
            return write!(f, "])");
        }
        if self.ndim() == 2 {
            let rows = self.shape.dim(0).unwrap();
            let cols = self.shape.dim(1).unwrap();
            writeln!(f, "tensor([")?;
            for i in 0..rows.min(8) {
                write!(f, "  [")?;
                for j in 0..cols.min(8) {
                    if j > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{:.4}", self.data[i * cols + j])?;
                }
                if cols > 8 {
                    write!(f, ", ...")?;
                }
                writeln!(f, "],")?;
            }
            if rows > 8 {
                writeln!(f, "  ...")?;
            }
            return write!(f, "], shape={})", self.shape);
        }
        write!(f, "tensor(shape={}, numel={})", self.shape, self.numel())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creation() {
        let t: Tensor<f64> = Tensor::zeros(vec![3, 4]);
        assert_eq!(t.shape_vec(), vec![3, 4]);
        assert_eq!(t.numel(), 12);
        assert_eq!(t.data()[0], 0.0);

        let t: Tensor<f64> = Tensor::ones(vec![2, 3]);
        assert_eq!(t.sum_all(), 6.0);

        let t: Tensor<f64> = Tensor::eye(3);
        assert_eq!(t.sum_all(), 3.0);
        assert_eq!(t.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(t.get(&[0, 1]).unwrap(), 0.0);
    }

    #[test]
    fn test_from_vec2d() {
        let t: Tensor<f64> = Tensor::from_vec2d(&[
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ])
        .unwrap();
        assert_eq!(t.shape_vec(), vec![2, 3]);
        assert_eq!(t.get(&[1, 2]).unwrap(), 6.0);
    }

    #[test]
    fn test_arithmetic() {
        let a: Tensor<f64> = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b: Tensor<f64> = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let c = a.add(&b).unwrap();
        assert_eq!(c.data(), &[6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_broadcasting() {
        let a: Tensor<f64> = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b: Tensor<f64> = Tensor::new(vec![10.0, 20.0, 30.0], vec![1, 3]).unwrap();
        let c = a.add(&b).unwrap();
        assert_eq!(c.shape_vec(), vec![2, 3]);
        assert_eq!(c.data(), &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn test_matmul() {
        let a: Tensor<f64> = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b: Tensor<f64> =
            Tensor::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape_vec(), vec![2, 2]);
        // [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        // [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        assert_eq!(c.data(), &[58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_transpose() {
        let a: Tensor<f64> = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let t = a.t().unwrap();
        assert_eq!(t.shape_vec(), vec![3, 2]);
        assert_eq!(t.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(t.get(&[1, 0]).unwrap(), 2.0);
        assert_eq!(t.get(&[2, 1]).unwrap(), 6.0);
    }

    #[test]
    fn test_reshape() {
        let a: Tensor<f64> = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = a.reshape(vec![3, 2]).unwrap();
        assert_eq!(b.shape_vec(), vec![3, 2]);
        assert_eq!(b.data(), a.data());
    }

    #[test]
    fn test_sum_axis() {
        let a: Tensor<f64> = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let s0 = a.sum_axis(0).unwrap();
        assert_eq!(s0.data(), &[5.0, 7.0, 9.0]);

        let s1 = a.sum_axis(1).unwrap();
        assert_eq!(s1.data(), &[6.0, 15.0]);
    }

    #[test]
    fn test_softmax() {
        let a: Tensor<f64> = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let sm = a.softmax().unwrap();
        let sum: f64 = sm.data().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid() {
        let a: Tensor<f64> = Tensor::from_slice(&[0.0]);
        let s = a.sigmoid();
        assert!((s.data()[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_linspace() {
        let t: Tensor<f64> = Tensor::linspace(0.0, 1.0, 5);
        assert_eq!(t.numel(), 5);
        assert!((t.data()[0] - 0.0).abs() < 1e-10);
        assert!((t.data()[4] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_concatenate() {
        let a: Tensor<f64> = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b: Tensor<f64> = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let c = Tensor::concatenate(&[&a, &b], 0).unwrap();
        assert_eq!(c.shape_vec(), vec![4, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_dot() {
        let a: Tensor<f64> = Tensor::from_slice(&[1.0, 2.0, 3.0]);
        let b: Tensor<f64> = Tensor::from_slice(&[4.0, 5.0, 6.0]);
        let d = a.dot(&b).unwrap();
        assert_eq!(d, 32.0);
    }

    #[test]
    fn test_rand() {
        let t: Tensor<f64> = Tensor::rand(vec![100], Some(42));
        assert_eq!(t.numel(), 100);
        let max = t.max_all().unwrap();
        let min = t.min_all().unwrap();
        assert!(min >= 0.0);
        assert!(max < 1.0);
    }
}
