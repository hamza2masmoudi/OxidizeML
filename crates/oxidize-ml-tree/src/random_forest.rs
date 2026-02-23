use oxidize_ml_core::{Float, Tensor, TensorError};
use oxidize_ml_core::error::TensorResult;
use rand::distributions::{Distribution, Standard};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::decision_tree::{DecisionTreeClassifier, DecisionTreeRegressor};

/// Random Forest Classifier — ensemble of decision trees with bagging.
pub struct RandomForestClassifier<T: Float> {
    pub n_estimators: usize,
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub max_features_ratio: f64,
    pub seed: Option<u64>,
    trees: Vec<DecisionTreeClassifier<T>>,
    feature_subsets: Vec<Vec<usize>>,
    pub n_classes: usize,
}

impl<T: Float> RandomForestClassifier<T>
where
    Standard: Distribution<T>,
{
    pub fn new(n_estimators: usize, max_depth: usize, max_features_ratio: f64) -> Self {
        RandomForestClassifier {
            n_estimators,
            max_depth,
            min_samples_split: 2,
            max_features_ratio,
            seed: Some(42),
            trees: Vec::new(),
            feature_subsets: Vec::new(),
            n_classes: 0,
        }
    }

    pub fn fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> TensorResult<()> {
        let n = x.shape().dim(0)?;
        let p = x.shape().dim(1)?;
        let max_features = ((p as f64 * self.max_features_ratio).ceil() as usize).max(1).min(p);

        let max_label = y.data().iter().map(|v| v.to_f64().round() as usize).max().unwrap_or(0);
        self.n_classes = max_label + 1;

        let mut base_rng = match self.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        self.trees.clear();
        self.feature_subsets.clear();

        for _ in 0..self.n_estimators {
            // Bootstrap sample
            let mut sample_indices: Vec<usize> = (0..n).collect();
            for i in 0..n {
                let j = (rand::Rng::gen::<f64>(&mut base_rng) * n as f64) as usize;
                sample_indices[i] = j.min(n - 1);
            }

            // Feature subsampling
            let mut feature_indices: Vec<usize> = (0..p).collect();
            feature_indices.shuffle(&mut base_rng);
            let selected_features: Vec<usize> = feature_indices[..max_features].to_vec();

            // Create subset data
            let mut x_sub_data = Vec::with_capacity(n * max_features);
            let mut y_sub_data = Vec::with_capacity(n);
            for &i in &sample_indices {
                for &f in &selected_features {
                    x_sub_data.push(x.get(&[i, f])?);
                }
                y_sub_data.push(y.data()[i]);
            }
            let x_sub = Tensor::new(x_sub_data, vec![n, max_features])?;
            let y_sub = Tensor::new(y_sub_data, vec![n])?;

            let mut tree = DecisionTreeClassifier::new(self.max_depth, self.min_samples_split, 1);
            tree.fit(&x_sub, &y_sub)?;

            self.trees.push(tree);
            self.feature_subsets.push(selected_features);
        }

        Ok(())
    }

    pub fn predict(&self, x: &Tensor<T>) -> TensorResult<Tensor<T>> {
        let n = x.shape().dim(0)?;
        let mut predictions = Vec::with_capacity(n);

        for i in 0..n {
            let mut votes = vec![0usize; self.n_classes];
            for (tree_idx, tree) in self.trees.iter().enumerate() {
                let features = &self.feature_subsets[tree_idx];
                let mut row_data = Vec::with_capacity(features.len());
                for &f in features {
                    row_data.push(x.get(&[i, f])?);
                }
                let row = Tensor::new(row_data, vec![1, features.len()])?;
                let pred = tree.predict(&row)?;
                let cls = pred.data()[0].to_f64().round() as usize;
                if cls < self.n_classes {
                    votes[cls] += 1;
                }
            }
            let best = votes.iter().enumerate().max_by_key(|(_, &c)| c).map(|(i, _)| i).unwrap_or(0);
            predictions.push(T::from_usize(best));
        }

        Tensor::new(predictions, vec![n])
    }
}

/// Random Forest Regressor.
pub struct RandomForestRegressor<T: Float> {
    pub n_estimators: usize,
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub max_features_ratio: f64,
    pub seed: Option<u64>,
    trees: Vec<DecisionTreeRegressor<T>>,
    feature_subsets: Vec<Vec<usize>>,
}

impl<T: Float> RandomForestRegressor<T>
where
    Standard: Distribution<T>,
{
    pub fn new(n_estimators: usize, max_depth: usize, max_features_ratio: f64) -> Self {
        RandomForestRegressor {
            n_estimators,
            max_depth,
            min_samples_split: 2,
            max_features_ratio,
            seed: Some(42),
            trees: Vec::new(),
            feature_subsets: Vec::new(),
        }
    }

    pub fn fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> TensorResult<()> {
        let n = x.shape().dim(0)?;
        let p = x.shape().dim(1)?;
        let max_features = ((p as f64 * self.max_features_ratio).ceil() as usize).max(1).min(p);

        let mut base_rng = match self.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        self.trees.clear();
        self.feature_subsets.clear();

        for _ in 0..self.n_estimators {
            let mut sample_indices: Vec<usize> = Vec::with_capacity(n);
            for _ in 0..n {
                let j = (rand::Rng::gen::<f64>(&mut base_rng) * n as f64) as usize;
                sample_indices.push(j.min(n - 1));
            }

            let mut feature_indices: Vec<usize> = (0..p).collect();
            feature_indices.shuffle(&mut base_rng);
            let selected_features: Vec<usize> = feature_indices[..max_features].to_vec();

            let mut x_sub_data = Vec::with_capacity(n * max_features);
            let mut y_sub_data = Vec::with_capacity(n);
            for &i in &sample_indices {
                for &f in &selected_features {
                    x_sub_data.push(x.get(&[i, f])?);
                }
                y_sub_data.push(y.data()[i]);
            }
            let x_sub = Tensor::new(x_sub_data, vec![n, max_features])?;
            let y_sub = Tensor::new(y_sub_data, vec![n])?;

            let mut tree = DecisionTreeRegressor::new(self.max_depth, self.min_samples_split, 1);
            tree.fit(&x_sub, &y_sub)?;

            self.trees.push(tree);
            self.feature_subsets.push(selected_features);
        }

        Ok(())
    }

    pub fn predict(&self, x: &Tensor<T>) -> TensorResult<Tensor<T>> {
        let n = x.shape().dim(0)?;
        let mut predictions = Vec::with_capacity(n);

        for i in 0..n {
            let mut sum = T::ZERO;
            for (tree_idx, tree) in self.trees.iter().enumerate() {
                let features = &self.feature_subsets[tree_idx];
                let mut row_data = Vec::with_capacity(features.len());
                for &f in features {
                    row_data.push(x.get(&[i, f])?);
                }
                let row = Tensor::new(row_data, vec![1, features.len()])?;
                let pred = tree.predict(&row)?;
                sum = sum + pred.data()[0];
            }
            predictions.push(sum / T::from_usize(self.trees.len()));
        }

        Tensor::new(predictions, vec![n])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_forest_classifier() {
        let x: Tensor<f64> = Tensor::from_vec2d(&[
            vec![0.0, 0.0], vec![0.5, 0.5], vec![1.0, 1.0],
            vec![5.0, 5.0], vec![5.5, 5.5], vec![6.0, 6.0],
        ]).unwrap();
        let y: Tensor<f64> = Tensor::from_slice(&[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut rf = RandomForestClassifier::new(10, 5, 1.0);
        rf.fit(&x, &y).unwrap();
        let pred = rf.predict(&x).unwrap();

        for i in 0..6 {
            assert!((pred.data()[i] - y.data()[i]).abs() < 0.5);
        }
    }
}
