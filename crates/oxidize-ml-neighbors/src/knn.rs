use oxidize_ml_core::{Float, Tensor, TensorError};
use oxidize_ml_core::error::TensorResult;

/// Distance metric for KNN.
#[derive(Debug, Clone, Copy)]
pub enum DistanceMetric {
    Euclidean,
    Manhattan,
}

/// K-Nearest Neighbors Classifier.
pub struct KNNClassifier<T: Float> {
    pub k: usize,
    pub metric: DistanceMetric,
    x_train: Option<Tensor<T>>,
    y_train: Option<Tensor<T>>,
    pub n_classes: usize,
}

impl<T: Float> KNNClassifier<T> {
    pub fn new(k: usize, metric: DistanceMetric) -> Self {
        KNNClassifier {
            k,
            metric,
            x_train: None,
            y_train: None,
            n_classes: 0,
        }
    }

    pub fn fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> TensorResult<()> {
        self.x_train = Some(x.clone());
        self.y_train = Some(y.clone());
        let max_label = y.data().iter().map(|v| v.to_f64().round() as usize).max().unwrap_or(0);
        self.n_classes = max_label + 1;
        Ok(())
    }

    fn distance(&self, x: &Tensor<T>, i: usize, train: &Tensor<T>, j: usize, d: usize) -> TensorResult<T> {
        match self.metric {
            DistanceMetric::Euclidean => {
                let mut dist = T::ZERO;
                for k in 0..d {
                    let diff = x.get(&[i, k])? - train.get(&[j, k])?;
                    dist = dist + diff * diff;
                }
                Ok(dist.sqrt())
            }
            DistanceMetric::Manhattan => {
                let mut dist = T::ZERO;
                for k in 0..d {
                    dist = dist + (x.get(&[i, k])? - train.get(&[j, k])?).abs();
                }
                Ok(dist)
            }
        }
    }

    pub fn predict(&self, x: &Tensor<T>) -> TensorResult<Tensor<T>> {
        let x_train = self.x_train.as_ref().ok_or_else(|| {
            TensorError::InvalidOperation("Model not fitted".into())
        })?;
        let y_train = self.y_train.as_ref().unwrap();
        let n_test = x.shape().dim(0)?;
        let n_train = x_train.shape().dim(0)?;
        let d = x.shape().dim(1)?;

        let mut predictions = Vec::with_capacity(n_test);

        for i in 0..n_test {
            // Compute all distances
            let mut dists: Vec<(T, usize)> = Vec::with_capacity(n_train);
            for j in 0..n_train {
                let dist = self.distance(x, i, x_train, j, d)?;
                dists.push((dist, j));
            }

            // Partial sort: find k nearest
            dists.sort_by(|a, b| a.0.to_f64().partial_cmp(&b.0.to_f64()).unwrap());

            // Majority vote
            let mut votes = vec![0usize; self.n_classes];
            for idx in 0..self.k.min(dists.len()) {
                let cls = y_train.data()[dists[idx].1].to_f64().round() as usize;
                if cls < self.n_classes {
                    votes[cls] += 1;
                }
            }
            let best = votes.iter().enumerate().max_by_key(|(_, &c)| c).map(|(i, _)| i).unwrap_or(0);
            predictions.push(T::from_usize(best));
        }

        Tensor::new(predictions, vec![n_test])
    }
}

/// K-Nearest Neighbors Regressor.
pub struct KNNRegressor<T: Float> {
    pub k: usize,
    pub metric: DistanceMetric,
    x_train: Option<Tensor<T>>,
    y_train: Option<Tensor<T>>,
}

impl<T: Float> KNNRegressor<T> {
    pub fn new(k: usize, metric: DistanceMetric) -> Self {
        KNNRegressor {
            k,
            metric,
            x_train: None,
            y_train: None,
        }
    }

    pub fn fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> TensorResult<()> {
        self.x_train = Some(x.clone());
        self.y_train = Some(y.clone());
        Ok(())
    }

    pub fn predict(&self, x: &Tensor<T>) -> TensorResult<Tensor<T>> {
        let x_train = self.x_train.as_ref().ok_or_else(|| {
            TensorError::InvalidOperation("Model not fitted".into())
        })?;
        let y_train = self.y_train.as_ref().unwrap();
        let n_test = x.shape().dim(0)?;
        let n_train = x_train.shape().dim(0)?;
        let d = x.shape().dim(1)?;

        let mut predictions = Vec::with_capacity(n_test);

        for i in 0..n_test {
            let mut dists: Vec<(f64, usize)> = Vec::with_capacity(n_train);
            for j in 0..n_train {
                let mut dist = T::ZERO;
                for k in 0..d {
                    let diff = x.get(&[i, k])? - x_train.get(&[j, k])?;
                    dist = dist + diff * diff;
                }
                dists.push((dist.to_f64().sqrt(), j));
            }
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            let mut sum = T::ZERO;
            let k = self.k.min(dists.len());
            for idx in 0..k {
                sum = sum + y_train.data()[dists[idx].1];
            }
            predictions.push(sum / T::from_usize(k));
        }

        Tensor::new(predictions, vec![n_test])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knn_classifier() {
        let x: Tensor<f64> = Tensor::from_vec2d(&[
            vec![0.0, 0.0], vec![0.5, 0.5], vec![1.0, 1.0],
            vec![5.0, 5.0], vec![5.5, 5.5], vec![6.0, 6.0],
        ]).unwrap();
        let y: Tensor<f64> = Tensor::from_slice(&[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut knn = KNNClassifier::new(3, DistanceMetric::Euclidean);
        knn.fit(&x, &y).unwrap();
        let pred = knn.predict(&x).unwrap();

        for i in 0..6 {
            assert!((pred.data()[i] - y.data()[i]).abs() < 0.5);
        }
    }
}
