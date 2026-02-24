use oximl_core::{Tensor, TensorError, TensorResult};

/// A PyTorch-style DataLoader that slices a Dataset into iterable mini-batches.
pub struct DataLoader {
    features: Tensor,
    labels: Tensor,
    batch_size: usize,
    current_idx: usize,
    num_samples: usize,
}

impl DataLoader {
    pub fn new(features: Tensor, labels: Tensor, batch_size: usize) -> TensorResult<Self> {
        let f_shape = features.shape();
        let l_shape = labels.shape();
        
        if f_shape.is_empty() || l_shape.is_empty() {
            return Err(TensorError::InvalidOperation("Cannot batch zero-dimensional tensors".into()));
        }
        
        if f_shape[0] != l_shape[0] {
            return Err(TensorError::InvalidOperation(format!(
                "Feature batch size {} does not match Label batch size {}", f_shape[0], l_shape[0]
            )));
        }

        Ok(DataLoader {
            num_samples: f_shape[0],
            features,
            labels,
            batch_size,
            current_idx: 0,
        })
    }
    
    /// Reset the iterator for the next epoch.
    pub fn reset(&mut self) {
        self.current_idx = 0;
    }
    
    pub fn len(&self) -> usize {
        (self.num_samples as f64 / self.batch_size as f64).ceil() as usize
    }
}

impl Iterator for DataLoader {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.num_samples {
            return None;
        }

        let end_idx = std::cmp::min(self.current_idx + self.batch_size, self.num_samples);
        
        // We know these slices shouldn't fail bounds logic due to constructor validation
        let batch_x = self.features.slice(self.current_idx, end_idx).unwrap();
        let batch_y = self.labels.slice(self.current_idx, end_idx).unwrap();

        self.current_idx = end_idx;

        Some((batch_x, batch_y))
    }
}
