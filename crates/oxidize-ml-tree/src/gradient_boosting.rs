use oxidize_ml_core::{Float, Tensor, TensorError};
use oxidize_ml_core::error::TensorResult;
use crate::decision_tree::DecisionTreeRegressor;
use rand::distributions::{Distribution, Standard};

/// Gradient Boosted Trees for Regression.
///
/// Uses gradient descent in function space by sequentially fitting
/// decision trees to the residuals (negative gradient of the loss).
pub struct GradientBoostingRegressor<T: Float> {
    pub n_estimators: usize,
    pub learning_rate: T,
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub subsample: f64,
    trees: Vec<DecisionTreeRegressor<T>>,
    initial_prediction: T,
}

impl<T: Float> GradientBoostingRegressor<T>
where
    Standard: Distribution<T>,
{
    pub fn new(
        n_estimators: usize,
        learning_rate: T,
        max_depth: usize,
        min_samples_split: usize,
        subsample: f64,
    ) -> Self {
        GradientBoostingRegressor {
            n_estimators,
            learning_rate,
            max_depth: if max_depth == 0 { 3 } else { max_depth },
            min_samples_split: if min_samples_split == 0 { 2 } else { min_samples_split },
            subsample: subsample.max(0.1).min(1.0),
            trees: Vec::new(),
            initial_prediction: T::ZERO,
        }
    }

    pub fn fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> TensorResult<()> {
        let n = x.shape().dim(0)?;
        let _p = x.shape().dim(1)?;

        // Initial prediction: mean of y
        let y_sum: T = y.data().iter().copied().sum();
        self.initial_prediction = y_sum / T::from_usize(n);

        // Current predictions
        let mut predictions = vec![self.initial_prediction; n];

        self.trees.clear();

        for _iter in 0..self.n_estimators {
            // Compute residuals (negative gradient of MSE = y - pred)
            let residuals: Vec<T> = y.data().iter().zip(predictions.iter())
                .map(|(&yi, &pi)| yi - pi)
                .collect();
            let residual_tensor = Tensor::new(residuals, vec![n])?;

            // Fit tree to residuals
            let mut tree = DecisionTreeRegressor::new(self.max_depth, self.min_samples_split, 1);
            tree.fit(x, &residual_tensor)?;

            // Update predictions
            let tree_pred = tree.predict(x)?;
            for i in 0..n {
                predictions[i] = predictions[i] + self.learning_rate * tree_pred.data()[i];
            }

            self.trees.push(tree);
        }

        Ok(())
    }

    pub fn predict(&self, x: &Tensor<T>) -> TensorResult<Tensor<T>> {
        let n = x.shape().dim(0)?;
        let mut predictions = vec![self.initial_prediction; n];

        for tree in &self.trees {
            let tree_pred = tree.predict(x)?;
            for i in 0..n {
                predictions[i] = predictions[i] + self.learning_rate * tree_pred.data()[i];
            }
        }

        Tensor::new(predictions, vec![n])
    }

    pub fn n_trees(&self) -> usize {
        self.trees.len()
    }
}

/// Gradient Boosted Trees for Binary Classification.
///
/// Uses log-loss (binary cross-entropy) as the objective.
/// Predictions are log-odds, converted with sigmoid.
pub struct GradientBoostingClassifier<T: Float> {
    pub n_estimators: usize,
    pub learning_rate: T,
    pub max_depth: usize,
    pub min_samples_split: usize,
    trees: Vec<DecisionTreeRegressor<T>>,
    initial_log_odds: T,
}

impl<T: Float> GradientBoostingClassifier<T>
where
    Standard: Distribution<T>,
{
    pub fn new(
        n_estimators: usize,
        learning_rate: T,
        max_depth: usize,
        min_samples_split: usize,
    ) -> Self {
        GradientBoostingClassifier {
            n_estimators,
            learning_rate,
            max_depth: if max_depth == 0 { 3 } else { max_depth },
            min_samples_split: if min_samples_split == 0 { 2 } else { min_samples_split },
            trees: Vec::new(),
            initial_log_odds: T::ZERO,
        }
    }

    fn sigmoid(x: T) -> T {
        T::ONE / (T::ONE + (-x).exp())
    }

    pub fn fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> TensorResult<()> {
        let n = x.shape().dim(0)?;

        // Initial log-odds based on class proportions
        let pos_count: f64 = y.data().iter().map(|v| v.to_f64()).sum();
        let neg_count = n as f64 - pos_count;
        self.initial_log_odds = if neg_count > 0.0 {
            T::from_f64((pos_count / neg_count).max(1e-10).ln())
        } else {
            T::ZERO
        };

        let mut raw_predictions = vec![self.initial_log_odds; n];
        self.trees.clear();

        for _iter in 0..self.n_estimators {
            // Compute pseudo-residuals: y - sigmoid(raw_prediction)
            let residuals: Vec<T> = y.data().iter().zip(raw_predictions.iter())
                .map(|(&yi, &ri)| yi - Self::sigmoid(ri))
                .collect();
            let residual_tensor = Tensor::new(residuals, vec![n])?;

            // Fit tree to pseudo-residuals
            let mut tree = DecisionTreeRegressor::new(self.max_depth, self.min_samples_split, 1);
            tree.fit(x, &residual_tensor)?;

            // Update raw predictions
            let tree_pred = tree.predict(x)?;
            for i in 0..n {
                raw_predictions[i] = raw_predictions[i] + self.learning_rate * tree_pred.data()[i];
            }

            self.trees.push(tree);
        }

        Ok(())
    }

    pub fn predict_proba(&self, x: &Tensor<T>) -> TensorResult<Tensor<T>> {
        let n = x.shape().dim(0)?;
        let mut raw_predictions = vec![self.initial_log_odds; n];

        for tree in &self.trees {
            let tree_pred = tree.predict(x)?;
            for i in 0..n {
                raw_predictions[i] = raw_predictions[i] + self.learning_rate * tree_pred.data()[i];
            }
        }

        let probas: Vec<T> = raw_predictions.iter().map(|&r| Self::sigmoid(r)).collect();
        Tensor::new(probas, vec![n])
    }

    pub fn predict(&self, x: &Tensor<T>) -> TensorResult<Tensor<T>> {
        let probas = self.predict_proba(x)?;
        let preds: Vec<T> = probas.data().iter()
            .map(|&p| if p >= T::HALF { T::ONE } else { T::ZERO })
            .collect();
        Tensor::new(preds, vec![x.shape().dim(0)?])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_boosting_regressor() {
        let x: Tensor<f64> = Tensor::from_vec2d(&[
            vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0],
            vec![6.0], vec![7.0], vec![8.0], vec![9.0], vec![10.0],
        ]).unwrap();
        let y: Tensor<f64> = Tensor::from_slice(&[3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0]);

        let mut model = GradientBoostingRegressor::new(50, 0.1, 3, 2, 1.0);
        model.fit(&x, &y).unwrap();

        let pred = model.predict(&x).unwrap();
        for i in 0..10 {
            assert!((pred.data()[i] - y.data()[i]).abs() < 2.0,
                "prediction {} != expected {} at index {}", pred.data()[i], y.data()[i], i);
        }
    }

    #[test]
    fn test_gradient_boosting_classifier() {
        let x: Tensor<f64> = Tensor::from_vec2d(&[
            vec![0.0, 0.0], vec![0.1, 0.1], vec![0.2, 0.2],
            vec![0.8, 0.8], vec![0.9, 0.9], vec![1.0, 1.0],
        ]).unwrap();
        let y: Tensor<f64> = Tensor::from_slice(&[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut model = GradientBoostingClassifier::new(50, 0.1, 3, 2);
        model.fit(&x, &y).unwrap();

        let pred = model.predict(&x).unwrap();
        for i in 0..6 {
            assert!((pred.data()[i] - y.data()[i]).abs() < 0.5,
                "prediction {} != expected {} at index {}", pred.data()[i], y.data()[i], i);
        }
    }
}
