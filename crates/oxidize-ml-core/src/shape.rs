use crate::error::{TensorError, TensorResult};
use serde::{Deserialize, Serialize};

/// Represents the shape of a tensor (dimensions).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    pub fn new(dims: Vec<usize>) -> Self {
        Shape { dims }
    }

    pub fn from_slice(dims: &[usize]) -> Self {
        Shape {
            dims: dims.to_vec(),
        }
    }

    pub fn scalar() -> Self {
        Shape { dims: vec![] }
    }

    /// Number of dimensions (rank).
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Size along a specific axis.
    pub fn dim(&self, axis: usize) -> TensorResult<usize> {
        self.dims.get(axis).copied().ok_or(TensorError::InvalidAxis {
            axis,
            ndim: self.ndim(),
        })
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        if self.dims.is_empty() {
            1 // scalar
        } else {
            self.dims.iter().product()
        }
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn to_vec(&self) -> Vec<usize> {
        self.dims.clone()
    }

    /// Compute row-major (C-order) strides.
    pub fn strides(&self) -> Vec<usize> {
        if self.dims.is_empty() {
            return vec![];
        }
        let mut strides = vec![1usize; self.dims.len()];
        for i in (0..self.dims.len() - 1).rev() {
            strides[i] = strides[i + 1] * self.dims[i + 1];
        }
        strides
    }

    /// Check if two shapes are compatible for broadcasting (NumPy rules).
    pub fn broadcast_shape(a: &Shape, b: &Shape) -> TensorResult<Shape> {
        let max_ndim = a.ndim().max(b.ndim());
        let mut result = vec![0usize; max_ndim];

        for i in 0..max_ndim {
            let da = if i < a.ndim() {
                a.dims[a.ndim() - 1 - i]
            } else {
                1
            };
            let db = if i < b.ndim() {
                b.dims[b.ndim() - 1 - i]
            } else {
                1
            };

            if da == db {
                result[max_ndim - 1 - i] = da;
            } else if da == 1 {
                result[max_ndim - 1 - i] = db;
            } else if db == 1 {
                result[max_ndim - 1 - i] = da;
            } else {
                return Err(TensorError::BroadcastError {
                    a: a.to_vec(),
                    b: b.to_vec(),
                });
            }
        }

        Ok(Shape::new(result))
    }

    /// Check if this shape can be reshaped to `new_shape`.
    pub fn is_reshapable_to(&self, new_shape: &Shape) -> bool {
        self.numel() == new_shape.numel()
    }

    /// Transpose shape — swap last two dims for matrices.
    pub fn transposed(&self) -> TensorResult<Shape> {
        if self.ndim() < 2 {
            return Err(TensorError::InvalidOperation(
                "Cannot transpose tensor with fewer than 2 dimensions".to_string(),
            ));
        }
        let mut dims = self.dims.clone();
        let n = dims.len();
        dims.swap(n - 2, n - 1);
        Ok(Shape::new(dims))
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(")?;
        for (i, d) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", d)?;
        }
        write!(f, ")")
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Shape::new(dims)
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Shape::from_slice(dims)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_basics() {
        let s = Shape::new(vec![3, 4, 5]);
        assert_eq!(s.ndim(), 3);
        assert_eq!(s.numel(), 60);
        assert_eq!(s.dim(0).unwrap(), 3);
        assert_eq!(s.dim(1).unwrap(), 4);
        assert_eq!(s.dim(2).unwrap(), 5);
        assert!(s.dim(3).is_err());
    }

    #[test]
    fn test_strides() {
        let s = Shape::new(vec![3, 4, 5]);
        assert_eq!(s.strides(), vec![20, 5, 1]);

        let s2 = Shape::new(vec![2, 3]);
        assert_eq!(s2.strides(), vec![3, 1]);
    }

    #[test]
    fn test_broadcast() {
        let a = Shape::new(vec![3, 1]);
        let b = Shape::new(vec![1, 4]);
        let c = Shape::broadcast_shape(&a, &b).unwrap();
        assert_eq!(c.dims(), &[3, 4]);

        let a = Shape::new(vec![5, 3, 1]);
        let b = Shape::new(vec![4]);
        let c = Shape::broadcast_shape(&a, &b).unwrap();
        assert_eq!(c.dims(), &[5, 3, 4]);
    }

    #[test]
    fn test_broadcast_error() {
        let a = Shape::new(vec![3, 4]);
        let b = Shape::new(vec![3, 5]);
        assert!(Shape::broadcast_shape(&a, &b).is_err());
    }

    #[test]
    fn test_transpose() {
        let s = Shape::new(vec![3, 4]);
        let t = s.transposed().unwrap();
        assert_eq!(t.dims(), &[4, 3]);

        let s3 = Shape::new(vec![2, 3, 4]);
        let t3 = s3.transposed().unwrap();
        assert_eq!(t3.dims(), &[2, 4, 3]);
    }

    #[test]
    fn test_scalar() {
        let s = Shape::scalar();
        assert_eq!(s.ndim(), 0);
        assert_eq!(s.numel(), 1);
    }
}
