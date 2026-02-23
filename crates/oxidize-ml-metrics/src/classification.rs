use oxidize_ml_core::{Float, Tensor};

/// Compute accuracy: fraction of correct predictions.
pub fn accuracy<T: Float>(y_true: &Tensor<T>, y_pred: &Tensor<T>) -> f64 {
    assert_eq!(y_true.numel(), y_pred.numel(), "Length mismatch");
    let n = y_true.numel();
    let correct: usize = y_true
        .data()
        .iter()
        .zip(y_pred.data().iter())
        .filter(|(&a, &b)| (a - b).abs() < T::from_f64(0.5))
        .count();
    correct as f64 / n as f64
}

/// Compute confusion matrix for binary or multiclass classification.
/// Returns a 2D tensor of shape [n_classes, n_classes].
pub fn confusion_matrix<T: Float>(
    y_true: &Tensor<T>,
    y_pred: &Tensor<T>,
    n_classes: usize,
) -> Vec<Vec<usize>> {
    let mut matrix = vec![vec![0usize; n_classes]; n_classes];
    for (&t, &p) in y_true.data().iter().zip(y_pred.data().iter()) {
        let ti = t.to_f64().round() as usize;
        let pi = p.to_f64().round() as usize;
        if ti < n_classes && pi < n_classes {
            matrix[ti][pi] += 1;
        }
    }
    matrix
}

/// Precision for a specific class.
pub fn precision_class<T: Float>(
    y_true: &Tensor<T>,
    y_pred: &Tensor<T>,
    class: usize,
) -> f64 {
    let n = y_true.numel();
    let mut tp = 0usize;
    let mut fp = 0usize;
    for i in 0..n {
        let t = y_true.data()[i].to_f64().round() as usize;
        let p = y_pred.data()[i].to_f64().round() as usize;
        if p == class {
            if t == class {
                tp += 1;
            } else {
                fp += 1;
            }
        }
    }
    if tp + fp == 0 {
        0.0
    } else {
        tp as f64 / (tp + fp) as f64
    }
}

/// Recall for a specific class.
pub fn recall_class<T: Float>(
    y_true: &Tensor<T>,
    y_pred: &Tensor<T>,
    class: usize,
) -> f64 {
    let n = y_true.numel();
    let mut tp = 0usize;
    let mut fn_ = 0usize;
    for i in 0..n {
        let t = y_true.data()[i].to_f64().round() as usize;
        let p = y_pred.data()[i].to_f64().round() as usize;
        if t == class {
            if p == class {
                tp += 1;
            } else {
                fn_ += 1;
            }
        }
    }
    if tp + fn_ == 0 {
        0.0
    } else {
        tp as f64 / (tp + fn_) as f64
    }
}

/// F1 score for a specific class.
pub fn f1_score_class<T: Float>(
    y_true: &Tensor<T>,
    y_pred: &Tensor<T>,
    class: usize,
) -> f64 {
    let p = precision_class(y_true, y_pred, class);
    let r = recall_class(y_true, y_pred, class);
    if p + r == 0.0 {
        0.0
    } else {
        2.0 * p * r / (p + r)
    }
}

/// Macro-averaged precision across all classes.
pub fn precision_macro<T: Float>(
    y_true: &Tensor<T>,
    y_pred: &Tensor<T>,
    n_classes: usize,
) -> f64 {
    let sum: f64 = (0..n_classes)
        .map(|c| precision_class(y_true, y_pred, c))
        .sum();
    sum / n_classes as f64
}

/// Macro-averaged recall across all classes.
pub fn recall_macro<T: Float>(
    y_true: &Tensor<T>,
    y_pred: &Tensor<T>,
    n_classes: usize,
) -> f64 {
    let sum: f64 = (0..n_classes)
        .map(|c| recall_class(y_true, y_pred, c))
        .sum();
    sum / n_classes as f64
}

/// Macro-averaged F1 score.
pub fn f1_macro<T: Float>(
    y_true: &Tensor<T>,
    y_pred: &Tensor<T>,
    n_classes: usize,
) -> f64 {
    let sum: f64 = (0..n_classes)
        .map(|c| f1_score_class(y_true, y_pred, c))
        .sum();
    sum / n_classes as f64
}

/// Log loss (binary cross-entropy) for probabilistic predictions.
///
/// L = -mean(y * log(p) + (1-y) * log(1-p))
pub fn log_loss<T: Float>(y_true: &Tensor<T>, y_pred_proba: &Tensor<T>) -> f64 {
    let n = y_true.numel();
    let eps = 1e-15;
    let mut total = 0.0;
    for i in 0..n {
        let y = y_true.data()[i].to_f64();
        let p = y_pred_proba.data()[i].to_f64().max(eps).min(1.0 - eps);
        total -= y * p.ln() + (1.0 - y) * (1.0 - p).ln();
    }
    total / n as f64
}

/// ROC-AUC for binary classification.
///
/// Computes the Area Under the Receiver Operating Characteristic Curve
/// using the trapezoidal rule over all thresholds.
pub fn roc_auc<T: Float>(y_true: &Tensor<T>, y_scores: &Tensor<T>) -> f64 {
    let n = y_true.numel();
    // Create (score, label) pairs and sort by score descending
    let mut pairs: Vec<(f64, f64)> = y_scores.data().iter()
        .zip(y_true.data().iter())
        .map(|(&s, &t)| (s.to_f64(), t.to_f64().round()))
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    let total_pos = pairs.iter().filter(|(_, t)| *t > 0.5).count() as f64;
    let total_neg = n as f64 - total_pos;

    if total_pos == 0.0 || total_neg == 0.0 {
        return 0.5; // undefined, return random
    }

    let mut auc = 0.0;
    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut prev_tpr = 0.0;
    let mut prev_fpr = 0.0;

    for (_, label) in &pairs {
        if *label > 0.5 {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        let tpr = tp / total_pos;
        let fpr = fp / total_neg;
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0; // trapezoidal rule
        prev_tpr = tpr;
        prev_fpr = fpr;
    }
    auc
}

/// Cohen's Kappa: inter-rater agreement accounting for chance.
///
/// κ = (accuracy - expected_accuracy) / (1 - expected_accuracy)
pub fn cohen_kappa<T: Float>(y_true: &Tensor<T>, y_pred: &Tensor<T>, n_classes: usize) -> f64 {
    let cm = confusion_matrix(y_true, y_pred, n_classes);
    let n = y_true.numel() as f64;
    
    let observed_accuracy = accuracy(y_true, y_pred);
    
    let mut expected_accuracy = 0.0;
    for c in 0..n_classes {
        let row_sum: f64 = cm[c].iter().sum::<usize>() as f64;
        let col_sum: f64 = cm.iter().map(|row| row[c]).sum::<usize>() as f64;
        expected_accuracy += (row_sum / n) * (col_sum / n);
    }

    if (1.0 - expected_accuracy).abs() < 1e-10 {
        return 1.0;
    }
    (observed_accuracy - expected_accuracy) / (1.0 - expected_accuracy)
}

/// Matthews Correlation Coefficient (MCC) for binary classification.
///
/// MCC = (TP·TN - FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))
pub fn mcc_binary<T: Float>(y_true: &Tensor<T>, y_pred: &Tensor<T>) -> f64 {
    let cm = confusion_matrix(y_true, y_pred, 2);
    let tp = cm[1][1] as f64;
    let tn = cm[0][0] as f64;
    let fp = cm[0][1] as f64;
    let fn_ = cm[1][0] as f64;

    let denom = ((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)).sqrt();
    if denom < 1e-10 { return 0.0; }
    (tp * tn - fp * fn_) / denom
}

/// Silhouette score for clustering quality.
///
/// For each sample, computes (b - a) / max(a, b) where:
/// - a = mean distance to points in the same cluster
/// - b = mean distance to points in the nearest other cluster
pub fn silhouette_score<T: Float>(x: &Tensor<T>, labels: &Tensor<T>) -> f64 {
    let n = x.shape().dim(0).unwrap();
    let p = x.shape().dim(1).unwrap();

    if n <= 1 { return 0.0; }

    let mut total_score = 0.0;

    for i in 0..n {
        let label_i = labels.data()[i].to_f64().round() as usize;

        // Compute mean distance to same cluster (a) and to each other cluster
        let mut n_classes: std::collections::HashMap<usize, (f64, usize)> = std::collections::HashMap::new();

        for j in 0..n {
            if i == j { continue; }
            let label_j = labels.data()[j].to_f64().round() as usize;

            // Euclidean distance
            let mut dist = 0.0;
            for k in 0..p {
                let diff = x.get(&[i, k]).unwrap().to_f64() - x.get(&[j, k]).unwrap().to_f64();
                dist += diff * diff;
            }
            dist = dist.sqrt();

            let entry = n_classes.entry(label_j).or_insert((0.0, 0));
            entry.0 += dist;
            entry.1 += 1;
        }

        let a = if let Some(&(sum, count)) = n_classes.get(&label_i) {
            if count > 0 { sum / count as f64 } else { 0.0 }
        } else { 0.0 };

        let mut b = f64::INFINITY;
        for (&cluster, &(sum, count)) in &n_classes {
            if cluster != label_i && count > 0 {
                let mean_dist = sum / count as f64;
                if mean_dist < b { b = mean_dist; }
            }
        }
        if b == f64::INFINITY { b = 0.0; }

        let s = if a.max(b) > 0.0 { (b - a) / a.max(b) } else { 0.0 };
        total_score += s;
    }

    total_score / n as f64
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accuracy() {
        let y_true: Tensor<f64> = Tensor::from_slice(&[0.0, 1.0, 2.0, 1.0, 0.0]);
        let y_pred: Tensor<f64> = Tensor::from_slice(&[0.0, 1.0, 2.0, 0.0, 0.0]);
        let acc = accuracy(&y_true, &y_pred);
        assert!((acc - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_confusion_matrix() {
        let y_true: Tensor<f64> = Tensor::from_slice(&[0.0, 0.0, 1.0, 1.0]);
        let y_pred: Tensor<f64> = Tensor::from_slice(&[0.0, 1.0, 0.0, 1.0]);
        let cm = confusion_matrix(&y_true, &y_pred, 2);
        assert_eq!(cm[0][0], 1); // TN
        assert_eq!(cm[0][1], 1); // FP
        assert_eq!(cm[1][0], 1); // FN
        assert_eq!(cm[1][1], 1); // TP
    }

    #[test]
    fn test_precision_recall() {
        let y_true: Tensor<f64> = Tensor::from_slice(&[1.0, 1.0, 0.0, 0.0, 1.0]);
        let y_pred: Tensor<f64> = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0, 1.0]);
        let p = precision_class(&y_true, &y_pred, 1);
        let r = recall_class(&y_true, &y_pred, 1);
        // TP=2, FP=1, FN=1 → P=2/3, R=2/3
        assert!((p - 2.0 / 3.0).abs() < 1e-10);
        assert!((r - 2.0 / 3.0).abs() < 1e-10);
    }
}
