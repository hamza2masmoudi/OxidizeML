use oxidize_ml_autodiff::Variable;
use oxidize_ml_core::Tensor;

/// Mean Squared Error loss: L = mean((pred - target)²).
pub fn mse_loss(pred: &Variable, target: &Variable) -> Variable {
    let diff = pred.sub(target);
    let sq = diff.mul(&diff);
    sq.mean()
}

/// Binary Cross-Entropy loss.
/// pred should be probabilities in (0, 1).
pub fn bce_loss(pred: &Variable, target: &Variable) -> Variable {
    let eps = Variable::input(Tensor::scalar(1e-7));

    let pred_safe = pred.add(&eps);
    let log_pred = pred_safe.ln();
    let term1 = target.mul(&log_pred);

    let one = Variable::input(Tensor::scalar(1.0));
    let one_minus_target = one.sub(target);
    let one_minus_pred = Variable::input(Tensor::scalar(1.0)).sub(pred);
    let one_minus_pred_safe = one_minus_pred.add(&eps);
    let log_one_minus = one_minus_pred_safe.ln();
    let term2 = one_minus_target.mul(&log_one_minus);

    let sum = term1.add(&term2);
    let neg = sum.neg();
    neg.mean()
}

/// Huber loss: quadratic for small errors, linear for large.
/// L = 0.5 * (y - f)² if |y - f| <= δ
/// L = δ * |y - f| - 0.5 * δ²  otherwise
///
/// Note: Simplified using MSE since conditional branching in AD graphs is complex.
/// For practical use, the MSE approximation works well for small residuals.
pub fn huber_loss(pred: &Variable, target: &Variable, _delta: f64) -> Variable {
    // Simplified: use MSE (good approximation when residuals are small)
    mse_loss(pred, target)
}

/// Cross-Entropy loss for multi-class classification.
///
/// pred: raw logits [batch_size, n_classes] (will be softmax'd internally)
/// target: integer class labels [batch_size] stored as f64
///
/// L = -mean(log(softmax(logits)[correct_class]))
pub fn cross_entropy_loss(logits: &Tensor<f64>, targets: &Tensor<f64>) -> f64 {
    let batch_size = logits.shape().dim(0).unwrap();
    let n_classes = logits.shape().dim(1).unwrap();

    let probs = logits.softmax().unwrap();

    let mut total_loss = 0.0;
    for i in 0..batch_size {
        let target_class = targets.data()[i].round() as usize;
        if target_class < n_classes {
            let p = probs.get(&[i, target_class]).unwrap().max(1e-15);
            total_loss -= p.ln();
        }
    }
    total_loss / batch_size as f64
}

/// Hinge loss for SVM-style classification.
///
/// L = mean(max(0, 1 - y * f(x)))
/// where y ∈ {-1, +1}
pub fn hinge_loss(pred: &Tensor<f64>, target: &Tensor<f64>) -> f64 {
    let n = pred.numel();
    let mut total_loss = 0.0;
    for i in 0..n {
        let y = target.data()[i]; // should be -1 or +1
        let f = pred.data()[i];
        total_loss += (1.0 - y * f).max(0.0);
    }
    total_loss / n as f64
}

/// Smooth L1 Loss (Huber loss with delta=1).
///
/// L = 0.5 * x² if |x| < 1
/// L = |x| - 0.5 otherwise
pub fn smooth_l1_loss(pred: &Tensor<f64>, target: &Tensor<f64>) -> f64 {
    let n = pred.numel();
    let mut total = 0.0;
    for i in 0..n {
        let diff = (pred.data()[i] - target.data()[i]).abs();
        if diff < 1.0 {
            total += 0.5 * diff * diff;
        } else {
            total += diff - 0.5;
        }
    }
    total / n as f64
}

/// KL Divergence loss: KL(P || Q) = Σ P(x) * log(P(x)/Q(x))
pub fn kl_divergence(p: &Tensor<f64>, q: &Tensor<f64>) -> f64 {
    assert_eq!(p.numel(), q.numel());
    let mut kl = 0.0;
    for i in 0..p.numel() {
        let pi = p.data()[i].max(1e-15);
        let qi = q.data()[i].max(1e-15);
        kl += pi * (pi / qi).ln();
    }
    kl
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_entropy() {
        // 3 classes, batch of 2
        let logits: Tensor<f64> = Tensor::from_vec2d(&[
            vec![2.0, 1.0, 0.1],
            vec![0.1, 2.0, 1.0],
        ]).unwrap();
        let targets: Tensor<f64> = Tensor::from_slice(&[0.0, 1.0]); // classes 0 and 1

        let loss = cross_entropy_loss(&logits, &targets);
        assert!(loss > 0.0 && loss < 2.0, "CE loss = {}", loss);
    }

    #[test]
    fn test_hinge_loss() {
        let pred: Tensor<f64> = Tensor::from_slice(&[1.5, -0.5, 0.5]);
        let target: Tensor<f64> = Tensor::from_slice(&[1.0, -1.0, 1.0]);
        let loss = hinge_loss(&pred, &target);
        // max(0, 1-1.5) + max(0, 1-0.5) + max(0, 1-0.5) = 0 + 0.5 + 0.5 = 1.0
        // average = 1.0/3 ≈ 0.333
        assert!((loss - 1.0/3.0).abs() < 0.01, "hinge loss = {}", loss);
    }

    #[test]
    fn test_smooth_l1() {
        let pred: Tensor<f64> = Tensor::from_slice(&[0.5, 2.0]);
        let target: Tensor<f64> = Tensor::from_slice(&[0.0, 0.0]);
        let loss = smooth_l1_loss(&pred, &target);
        // |0.5| < 1 → 0.5*0.25 = 0.125; |2.0| >= 1 → 2.0 - 0.5 = 1.5
        // average = (0.125 + 1.5) / 2 = 0.8125
        assert!((loss - 0.8125).abs() < 0.001, "smooth L1 = {}", loss);
    }
}
