use oxidize_ml_core::{Float, Tensor, TensorError};
use oxidize_ml_core::error::TensorResult;
use rand::distributions::{Distribution, Standard};

/// A node in the decision tree.
#[derive(Debug, Clone)]
enum TreeNode<T: Float> {
    /// Internal node: splits on feature `feature_idx` at `threshold`.
    Split {
        feature_idx: usize,
        threshold: T,
        left: Box<TreeNode<T>>,
        right: Box<TreeNode<T>>,
    },
    /// Leaf: predicts a class label or regression value.
    Leaf { value: T },
}

/// Decision Tree Classifier using CART algorithm (Gini impurity).
pub struct DecisionTreeClassifier<T: Float> {
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    tree: Option<TreeNode<T>>,
    pub n_classes: usize,
}

impl<T: Float> DecisionTreeClassifier<T>
where
    Standard: Distribution<T>,
{
    pub fn new(max_depth: usize, min_samples_split: usize, min_samples_leaf: usize) -> Self {
        DecisionTreeClassifier {
            max_depth,
            min_samples_split,
            min_samples_leaf,
            tree: None,
            n_classes: 0,
        }
    }

    pub fn fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> TensorResult<()> {
        let n = x.shape().dim(0)?;
        let p = x.shape().dim(1)?;

        // Determine number of classes
        let max_label = y.data().iter().map(|v| v.to_f64().round() as usize).max().unwrap_or(0);
        self.n_classes = max_label + 1;

        let indices: Vec<usize> = (0..n).collect();
        self.tree = Some(self.build_tree(x, y, &indices, p, 0)?);
        Ok(())
    }

    fn build_tree(
        &self,
        x: &Tensor<T>,
        y: &Tensor<T>,
        indices: &[usize],
        n_features: usize,
        depth: usize,
    ) -> TensorResult<TreeNode<T>> {
        // Base cases
        if depth >= self.max_depth || indices.len() < self.min_samples_split || indices.len() < 2 {
            return Ok(TreeNode::Leaf {
                value: self.majority_class(y, indices),
            });
        }

        // Check if all same class
        let first_class = y.data()[indices[0]];
        if indices.iter().all(|&i| (y.data()[i] - first_class).abs() < T::EPSILON) {
            return Ok(TreeNode::Leaf { value: first_class });
        }

        // Find best split
        let mut best_gini = T::from_f64(f64::MAX);
        let mut best_feature = 0;
        let mut best_threshold = T::ZERO;
        let mut best_left = Vec::new();
        let mut best_right = Vec::new();

        for feature in 0..n_features {
            // Get unique sorted values for this feature
            let mut values: Vec<T> = indices.iter().map(|&i| x.get(&[i, feature]).unwrap()).collect();
            values.sort_by(|a, b| a.to_f64().partial_cmp(&b.to_f64()).unwrap());
            values.dedup();

            for w in values.windows(2) {
                let threshold = (w[0] + w[1]) / T::TWO;

                let mut left = Vec::new();
                let mut right = Vec::new();
                for &i in indices {
                    if x.get(&[i, feature]).unwrap() <= threshold {
                        left.push(i);
                    } else {
                        right.push(i);
                    }
                }

                if left.len() < self.min_samples_leaf || right.len() < self.min_samples_leaf {
                    continue;
                }

                let gini = self.weighted_gini(y, &left, &right, indices.len());
                if gini < best_gini {
                    best_gini = gini;
                    best_feature = feature;
                    best_threshold = threshold;
                    best_left = left;
                    best_right = right;
                }
            }
        }

        if best_left.is_empty() || best_right.is_empty() {
            return Ok(TreeNode::Leaf {
                value: self.majority_class(y, indices),
            });
        }

        let left_node = self.build_tree(x, y, &best_left, n_features, depth + 1)?;
        let right_node = self.build_tree(x, y, &best_right, n_features, depth + 1)?;

        Ok(TreeNode::Split {
            feature_idx: best_feature,
            threshold: best_threshold,
            left: Box::new(left_node),
            right: Box::new(right_node),
        })
    }

    fn gini_impurity(&self, y: &Tensor<T>, indices: &[usize]) -> T {
        if indices.is_empty() {
            return T::ZERO;
        }
        let n = T::from_usize(indices.len());
        let mut counts = vec![0usize; self.n_classes];
        for &i in indices {
            let cls = y.data()[i].to_f64().round() as usize;
            if cls < self.n_classes {
                counts[cls] += 1;
            }
        }
        let mut gini = T::ONE;
        for &c in &counts {
            let p = T::from_usize(c) / n;
            gini = gini - p * p;
        }
        gini
    }

    fn weighted_gini(&self, y: &Tensor<T>, left: &[usize], right: &[usize], total: usize) -> T {
        let total_t = T::from_usize(total);
        let left_weight = T::from_usize(left.len()) / total_t;
        let right_weight = T::from_usize(right.len()) / total_t;
        left_weight * self.gini_impurity(y, left) + right_weight * self.gini_impurity(y, right)
    }

    fn majority_class(&self, y: &Tensor<T>, indices: &[usize]) -> T {
        let mut counts = vec![0usize; self.n_classes.max(1)];
        for &i in indices {
            let cls = y.data()[i].to_f64().round() as usize;
            if cls < counts.len() {
                counts[cls] += 1;
            }
        }
        let best = counts.iter().enumerate().max_by_key(|(_, &c)| c).map(|(i, _)| i).unwrap_or(0);
        T::from_usize(best)
    }

    fn predict_one(&self, x: &Tensor<T>, row: usize) -> TensorResult<T> {
        let tree = self.tree.as_ref().ok_or_else(|| {
            TensorError::InvalidOperation("Model not fitted".into())
        })?;
        Ok(self.traverse(tree, x, row)?)
    }

    fn traverse(&self, node: &TreeNode<T>, x: &Tensor<T>, row: usize) -> TensorResult<T> {
        match node {
            TreeNode::Leaf { value } => Ok(*value),
            TreeNode::Split {
                feature_idx,
                threshold,
                left,
                right,
            } => {
                let val = x.get(&[row, *feature_idx])?;
                if val <= *threshold {
                    self.traverse(left, x, row)
                } else {
                    self.traverse(right, x, row)
                }
            }
        }
    }

    pub fn predict(&self, x: &Tensor<T>) -> TensorResult<Tensor<T>> {
        let n = x.shape().dim(0)?;
        let mut predictions = Vec::with_capacity(n);
        for i in 0..n {
            predictions.push(self.predict_one(x, i)?);
        }
        Tensor::new(predictions, vec![n])
    }
}

/// Decision Tree Regressor using CART (MSE criterion).
pub struct DecisionTreeRegressor<T: Float> {
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    tree: Option<TreeNode<T>>,
}

impl<T: Float> DecisionTreeRegressor<T>
where
    Standard: Distribution<T>,
{
    pub fn new(max_depth: usize, min_samples_split: usize, min_samples_leaf: usize) -> Self {
        DecisionTreeRegressor {
            max_depth,
            min_samples_split,
            min_samples_leaf,
            tree: None,
        }
    }

    pub fn fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> TensorResult<()> {
        let n = x.shape().dim(0)?;
        let p = x.shape().dim(1)?;
        let indices: Vec<usize> = (0..n).collect();
        self.tree = Some(self.build_tree(x, y, &indices, p, 0)?);
        Ok(())
    }

    fn build_tree(
        &self,
        x: &Tensor<T>,
        y: &Tensor<T>,
        indices: &[usize],
        n_features: usize,
        depth: usize,
    ) -> TensorResult<TreeNode<T>> {
        if depth >= self.max_depth || indices.len() < self.min_samples_split || indices.len() < 2 {
            return Ok(TreeNode::Leaf {
                value: Self::mean_value(y, indices),
            });
        }

        let mut best_mse = T::from_f64(f64::MAX);
        let mut best_feature = 0;
        let mut best_threshold = T::ZERO;
        let mut best_left = Vec::new();
        let mut best_right = Vec::new();

        for feature in 0..n_features {
            let mut values: Vec<T> = indices.iter().map(|&i| x.get(&[i, feature]).unwrap()).collect();
            values.sort_by(|a, b| a.to_f64().partial_cmp(&b.to_f64()).unwrap());
            values.dedup();

            for w in values.windows(2) {
                let threshold = (w[0] + w[1]) / T::TWO;
                let mut left = Vec::new();
                let mut right = Vec::new();
                for &i in indices {
                    if x.get(&[i, feature]).unwrap() <= threshold {
                        left.push(i);
                    } else {
                        right.push(i);
                    }
                }
                if left.len() < self.min_samples_leaf || right.len() < self.min_samples_leaf {
                    continue;
                }
                let mse = Self::weighted_mse(y, &left, &right, indices.len());
                if mse < best_mse {
                    best_mse = mse;
                    best_feature = feature;
                    best_threshold = threshold;
                    best_left = left;
                    best_right = right;
                }
            }
        }

        if best_left.is_empty() || best_right.is_empty() {
            return Ok(TreeNode::Leaf {
                value: Self::mean_value(y, indices),
            });
        }

        let left = self.build_tree(x, y, &best_left, n_features, depth + 1)?;
        let right = self.build_tree(x, y, &best_right, n_features, depth + 1)?;

        Ok(TreeNode::Split {
            feature_idx: best_feature,
            threshold: best_threshold,
            left: Box::new(left),
            right: Box::new(right),
        })
    }

    fn mean_value(y: &Tensor<T>, indices: &[usize]) -> T {
        if indices.is_empty() {
            return T::ZERO;
        }
        let sum: T = indices.iter().map(|&i| y.data()[i]).sum();
        sum / T::from_usize(indices.len())
    }

    fn mse_value(y: &Tensor<T>, indices: &[usize]) -> T {
        let mean = Self::mean_value(y, indices);
        let n = T::from_usize(indices.len());
        let sum: T = indices.iter().map(|&i| {
            let d = y.data()[i] - mean;
            d * d
        }).sum();
        sum / n
    }

    fn weighted_mse(y: &Tensor<T>, left: &[usize], right: &[usize], total: usize) -> T {
        let t = T::from_usize(total);
        let lw = T::from_usize(left.len()) / t;
        let rw = T::from_usize(right.len()) / t;
        lw * Self::mse_value(y, left) + rw * Self::mse_value(y, right)
    }

    pub fn predict(&self, x: &Tensor<T>) -> TensorResult<Tensor<T>> {
        let n = x.shape().dim(0)?;
        let tree = self.tree.as_ref().ok_or_else(|| {
            TensorError::InvalidOperation("Model not fitted".into())
        })?;
        let mut preds = Vec::with_capacity(n);
        for i in 0..n {
            preds.push(self.traverse(tree, x, i)?);
        }
        Tensor::new(preds, vec![n])
    }

    fn traverse(&self, node: &TreeNode<T>, x: &Tensor<T>, row: usize) -> TensorResult<T> {
        match node {
            TreeNode::Leaf { value } => Ok(*value),
            TreeNode::Split { feature_idx, threshold, left, right } => {
                if x.get(&[row, *feature_idx])? <= *threshold {
                    self.traverse(left, x, row)
                } else {
                    self.traverse(right, x, row)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decision_tree_classifier() {
        let x: Tensor<f64> = Tensor::from_vec2d(&[
            vec![0.0], vec![1.0], vec![2.0], vec![3.0],
            vec![4.0], vec![5.0], vec![6.0], vec![7.0],
        ]).unwrap();
        let y: Tensor<f64> = Tensor::from_slice(&[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);

        let mut tree = DecisionTreeClassifier::new(10, 2, 1);
        tree.fit(&x, &y).unwrap();
        let pred = tree.predict(&x).unwrap();

        // Should get 100% accuracy on training data
        for i in 0..8 {
            assert!((pred.data()[i] - y.data()[i]).abs() < 0.5, "Mismatch at {}", i);
        }
    }

    #[test]
    fn test_decision_tree_regressor() {
        let x: Tensor<f64> = Tensor::from_vec2d(&[
            vec![1.0], vec![2.0], vec![3.0], vec![4.0],
        ]).unwrap();
        let y: Tensor<f64> = Tensor::from_slice(&[2.0, 4.0, 6.0, 8.0]);

        let mut tree = DecisionTreeRegressor::new(10, 2, 1);
        tree.fit(&x, &y).unwrap();
        let pred = tree.predict(&x).unwrap();

        for i in 0..4 {
            assert!((pred.data()[i] - y.data()[i]).abs() < 1.0);
        }
    }
}
