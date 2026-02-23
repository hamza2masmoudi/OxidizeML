use oxidize_ml_core::{Float, Tensor};
use rand::distributions::{Distribution, Standard};
use std::collections::HashMap;

/// Encode categorical string labels as integer indices.
pub struct LabelEncoder {
    pub classes: Vec<String>,
    pub class_to_idx: HashMap<String, usize>,
}

impl LabelEncoder {
    pub fn new() -> Self {
        LabelEncoder {
            classes: Vec::new(),
            class_to_idx: HashMap::new(),
        }
    }

    /// Fit the encoder on string labels.
    pub fn fit(&mut self, labels: &[String]) {
        let mut unique: Vec<String> = labels.to_vec();
        unique.sort();
        unique.dedup();
        self.classes = unique;
        self.class_to_idx = self
            .classes
            .iter()
            .enumerate()
            .map(|(i, c)| (c.clone(), i))
            .collect();
    }

    /// Transform string labels to integer tensor.
    pub fn transform<T: Float>(&self, labels: &[String]) -> Tensor<T>
    where
        Standard: Distribution<T>,
    {
        let data: Vec<T> = labels
            .iter()
            .map(|l| T::from_usize(*self.class_to_idx.get(l).expect("unknown label")))
            .collect();
        Tensor::from_slice(&data)
    }

    /// Inverse transform: integer → string.
    pub fn inverse_transform<T: Float>(&self, encoded: &Tensor<T>) -> Vec<String>
    where
        Standard: Distribution<T>,
    {
        encoded
            .data()
            .iter()
            .map(|v| {
                let idx = v.to_f64().round() as usize;
                self.classes[idx].clone()
            })
            .collect()
    }

    pub fn n_classes(&self) -> usize {
        self.classes.len()
    }
}

impl Default for LabelEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// One-hot encode integer labels into a binary matrix.
pub fn one_hot_encode<T: Float>(labels: &Tensor<T>, n_classes: usize) -> Tensor<T>
where
    rand::distributions::Standard: rand::distributions::Distribution<T>,
{
    let n = labels.numel();
    let mut data = vec![T::ZERO; n * n_classes];
    for i in 0..n {
        let cls = labels.data()[i].to_f64().round() as usize;
        if cls < n_classes {
            data[i * n_classes + cls] = T::ONE;
        }
    }
    Tensor::new(data, vec![n, n_classes]).expect("one_hot shape")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_label_encoder() {
        let mut enc = LabelEncoder::new();
        let labels = vec!["cat".into(), "dog".into(), "cat".into(), "fish".into()];
        enc.fit(&labels);
        assert_eq!(enc.n_classes(), 3);

        let encoded: Tensor<f64> = enc.transform(&labels);
        let decoded = enc.inverse_transform(&encoded);
        assert_eq!(decoded, labels);
    }

    #[test]
    fn test_one_hot() {
        let labels: Tensor<f64> = Tensor::from_slice(&[0.0, 1.0, 2.0, 1.0]);
        let oh = one_hot_encode(&labels, 3);
        assert_eq!(oh.shape_vec(), vec![4, 3]);
        assert_eq!(oh.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(oh.get(&[1, 1]).unwrap(), 1.0);
        assert_eq!(oh.get(&[2, 2]).unwrap(), 1.0);
        assert_eq!(oh.get(&[3, 1]).unwrap(), 1.0);
    }
}
