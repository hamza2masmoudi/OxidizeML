use oxidize_ml_core::{Float, Tensor, TensorError};
use oxidize_ml_core::error::TensorResult;

/// Multinomial Naive Bayes classifier.
///
/// Suitable for discrete features (e.g., word counts in text classification).
/// P(x_i | y) follows a multinomial distribution.
pub struct MultinomialNB<T: Float> {
    pub alpha: T,  // Laplace smoothing parameter
    class_log_prior: Vec<f64>,
    feature_log_prob: Vec<Vec<f64>>,  // [n_classes][n_features]
    n_classes: usize,
}

impl<T: Float> MultinomialNB<T> {
    pub fn new(alpha: T) -> Self {
        MultinomialNB {
            alpha,
            class_log_prior: Vec::new(),
            feature_log_prob: Vec::new(),
            n_classes: 0,
        }
    }

    pub fn fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> TensorResult<()> {
        let n = x.shape().dim(0)?;
        let p = x.shape().dim(1)?;
        let alpha = self.alpha.to_f64();

        // Determine classes
        let max_label = y.data().iter()
            .map(|v| v.to_f64().round() as usize)
            .max().unwrap_or(0);
        self.n_classes = max_label + 1;

        // Count class occurrences and feature sums per class
        let mut class_counts = vec![0.0f64; self.n_classes];
        let mut feature_counts = vec![vec![0.0f64; p]; self.n_classes];

        for i in 0..n {
            let cls = y.data()[i].to_f64().round() as usize;
            if cls >= self.n_classes { continue; }
            class_counts[cls] += 1.0;
            for j in 0..p {
                feature_counts[cls][j] += x.get(&[i, j])?.to_f64();
            }
        }

        // Compute log priors
        self.class_log_prior = class_counts.iter()
            .map(|&c| (c / n as f64).ln())
            .collect();

        // Compute log probabilities with Laplace smoothing
        self.feature_log_prob = Vec::with_capacity(self.n_classes);
        for cls in 0..self.n_classes {
            let total: f64 = feature_counts[cls].iter().sum::<f64>() + alpha * p as f64;
            let log_probs: Vec<f64> = feature_counts[cls].iter()
                .map(|&c| ((c + alpha) / total).ln())
                .collect();
            self.feature_log_prob.push(log_probs);
        }

        Ok(())
    }

    pub fn predict(&self, x: &Tensor<T>) -> TensorResult<Tensor<T>> {
        let n = x.shape().dim(0)?;
        let p = x.shape().dim(1)?;
        let mut predictions = Vec::with_capacity(n);

        for i in 0..n {
            let mut best_class = 0;
            let mut best_score = f64::NEG_INFINITY;

            for cls in 0..self.n_classes {
                let mut score = self.class_log_prior[cls];
                for j in 0..p {
                    score += x.get(&[i, j])?.to_f64() * self.feature_log_prob[cls][j];
                }
                if score > best_score {
                    best_score = score;
                    best_class = cls;
                }
            }
            predictions.push(T::from_usize(best_class));
        }

        Tensor::new(predictions, vec![n])
    }

    pub fn predict_log_proba(&self, x: &Tensor<T>) -> TensorResult<Vec<Vec<f64>>> {
        let n = x.shape().dim(0)?;
        let p = x.shape().dim(1)?;
        let mut results = Vec::with_capacity(n);

        for i in 0..n {
            let mut log_probs = Vec::with_capacity(self.n_classes);
            for cls in 0..self.n_classes {
                let mut score = self.class_log_prior[cls];
                for j in 0..p {
                    score += x.get(&[i, j])?.to_f64() * self.feature_log_prob[cls][j];
                }
                log_probs.push(score);
            }
            results.push(log_probs);
        }
        Ok(results)
    }
}

/// Bernoulli Naive Bayes classifier.
///
/// For binary/boolean features. P(x_i | y) follows a Bernoulli distribution.
pub struct BernoulliNB<T: Float> {
    pub alpha: T,
    class_log_prior: Vec<f64>,
    feature_log_prob: Vec<Vec<f64>>,     // log P(x_i=1 | y)
    feature_log_neg_prob: Vec<Vec<f64>>,  // log P(x_i=0 | y)
    n_classes: usize,
}

impl<T: Float> BernoulliNB<T> {
    pub fn new(alpha: T) -> Self {
        BernoulliNB {
            alpha,
            class_log_prior: Vec::new(),
            feature_log_prob: Vec::new(),
            feature_log_neg_prob: Vec::new(),
            n_classes: 0,
        }
    }

    pub fn fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> TensorResult<()> {
        let n = x.shape().dim(0)?;
        let p = x.shape().dim(1)?;
        let alpha = self.alpha.to_f64();

        let max_label = y.data().iter()
            .map(|v| v.to_f64().round() as usize)
            .max().unwrap_or(0);
        self.n_classes = max_label + 1;

        let mut class_counts = vec![0.0f64; self.n_classes];
        let mut feature_counts = vec![vec![0.0f64; p]; self.n_classes];

        for i in 0..n {
            let cls = y.data()[i].to_f64().round() as usize;
            if cls >= self.n_classes { continue; }
            class_counts[cls] += 1.0;
            for j in 0..p {
                if x.get(&[i, j])?.to_f64() > 0.5 {
                    feature_counts[cls][j] += 1.0;
                }
            }
        }

        self.class_log_prior = class_counts.iter()
            .map(|&c| (c / n as f64).ln())
            .collect();

        self.feature_log_prob = Vec::with_capacity(self.n_classes);
        self.feature_log_neg_prob = Vec::with_capacity(self.n_classes);
        for cls in 0..self.n_classes {
            let n_cls = class_counts[cls] + 2.0 * alpha;
            let log_p: Vec<f64> = feature_counts[cls].iter()
                .map(|&c| ((c + alpha) / n_cls).ln())
                .collect();
            let log_np: Vec<f64> = feature_counts[cls].iter()
                .map(|&c| ((n_cls - c - alpha) / n_cls).ln())
                .collect();
            self.feature_log_prob.push(log_p);
            self.feature_log_neg_prob.push(log_np);
        }

        Ok(())
    }

    pub fn predict(&self, x: &Tensor<T>) -> TensorResult<Tensor<T>> {
        let n = x.shape().dim(0)?;
        let p = x.shape().dim(1)?;
        let mut predictions = Vec::with_capacity(n);

        for i in 0..n {
            let mut best_class = 0;
            let mut best_score = f64::NEG_INFINITY;

            for cls in 0..self.n_classes {
                let mut score = self.class_log_prior[cls];
                for j in 0..p {
                    if x.get(&[i, j])?.to_f64() > 0.5 {
                        score += self.feature_log_prob[cls][j];
                    } else {
                        score += self.feature_log_neg_prob[cls][j];
                    }
                }
                if score > best_score {
                    best_score = score;
                    best_class = cls;
                }
            }
            predictions.push(T::from_usize(best_class));
        }

        Tensor::new(predictions, vec![n])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multinomial_nb() {
        // Simple word-count-like data: 2 classes, 3 features
        let x: Tensor<f64> = Tensor::from_vec2d(&[
            vec![3.0, 0.0, 1.0], // class 0: feature 0 dominant
            vec![2.0, 1.0, 0.0],
            vec![4.0, 0.0, 0.0],
            vec![0.0, 3.0, 1.0], // class 1: feature 1 dominant
            vec![1.0, 2.0, 2.0],
            vec![0.0, 4.0, 0.0],
        ]).unwrap();
        let y: Tensor<f64> = Tensor::from_slice(&[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut model = MultinomialNB::new(1.0);
        model.fit(&x, &y).unwrap();

        let pred = model.predict(&x).unwrap();
        // Should mostly classify correctly
        let correct: usize = pred.data().iter().zip(y.data().iter())
            .filter(|(&p, &t)| (p - t).abs() < 0.5)
            .count();
        assert!(correct >= 4, "MultinomialNB accuracy: {}/6", correct);
    }

    #[test]
    fn test_bernoulli_nb() {
        let x: Tensor<f64> = Tensor::from_vec2d(&[
            vec![1.0, 0.0, 1.0], vec![1.0, 1.0, 0.0], vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 1.0], vec![0.0, 1.0, 0.0], vec![0.0, 1.0, 1.0],
        ]).unwrap();
        let y: Tensor<f64> = Tensor::from_slice(&[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut model = BernoulliNB::new(1.0);
        model.fit(&x, &y).unwrap();

        let pred = model.predict(&x).unwrap();
        let correct: usize = pred.data().iter().zip(y.data().iter())
            .filter(|(&p, &t)| (p - t).abs() < 0.5)
            .count();
        assert!(correct >= 4, "BernoulliNB accuracy: {}/6", correct);
    }
}
