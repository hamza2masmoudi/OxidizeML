use oxidize_ml_core::Tensor;
use crate::dataset::Dataset;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

/// DataLoader for batching and shuffling datasets.
pub struct DataLoader<'a, D: Dataset> {
    dataset: &'a D,
    batch_size: usize,
    shuffle: bool,
    indices: Vec<usize>,
    current: usize,
}

impl<'a, D: Dataset> DataLoader<'a, D> {
    pub fn new(dataset: &'a D, batch_size: usize, shuffle: bool) -> Self {
        let mut indices: Vec<usize> = (0..dataset.len()).collect();
        if shuffle {
            let mut rng = StdRng::seed_from_u64(42);
            indices.shuffle(&mut rng);
        }
        DataLoader {
            dataset,
            batch_size,
            shuffle,
            indices,
            current: 0,
        }
    }

    /// Reset the iterator (reshuffle if needed).
    pub fn reset(&mut self) {
        self.current = 0;
        if self.shuffle {
            let mut rng = StdRng::from_entropy();
            self.indices.shuffle(&mut rng);
        }
    }
}

impl<'a, D: Dataset> Iterator for DataLoader<'a, D> {
    type Item = (Tensor<f64>, Tensor<f64>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.indices.len() {
            return None;
        }

        let end = (self.current + self.batch_size).min(self.indices.len());
        let batch_indices = &self.indices[self.current..end];
        let batch_size = batch_indices.len();

        // Get first sample to determine feature size
        let (first_x, first_y) = self.dataset.get(batch_indices[0]);
        let n_features = first_x.numel();

        let mut x_data = Vec::with_capacity(batch_size * n_features);
        let mut y_data = Vec::with_capacity(batch_size);

        for &idx in batch_indices {
            let (xi, yi) = self.dataset.get(idx);
            x_data.extend_from_slice(xi.data());
            y_data.push(yi.data()[0]);
        }

        self.current = end;

        let x = Tensor::new(x_data, vec![batch_size, n_features]).ok()?;
        let y = Tensor::new(y_data, vec![batch_size]).ok()?;
        Some((x, y))
    }
}
