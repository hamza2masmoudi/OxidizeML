use oxidize_ml_core::Tensor;

/// Trait for datasets.
pub trait Dataset {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn get(&self, idx: usize) -> (Tensor<f64>, Tensor<f64>);
}

/// A dataset wrapping feature and label tensors.
pub struct TensorDataset {
    pub features: Tensor<f64>,
    pub labels: Tensor<f64>,
}

impl TensorDataset {
    pub fn new(features: Tensor<f64>, labels: Tensor<f64>) -> Self {
        assert_eq!(features.shape().dim(0).unwrap(), labels.numel());
        TensorDataset { features, labels }
    }
}

impl Dataset for TensorDataset {
    fn len(&self) -> usize {
        self.features.shape().dim(0).unwrap()
    }

    fn get(&self, idx: usize) -> (Tensor<f64>, Tensor<f64>) {
        let row = self.features.row(idx).expect("row access");
        let label = Tensor::scalar(self.labels.data()[idx]);
        (row, label)
    }
}
