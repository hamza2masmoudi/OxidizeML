use oxidize_ml_core::{Float, Tensor};

/// Mean Squared Error.
pub fn mse<T: Float>(y_true: &Tensor<T>, y_pred: &Tensor<T>) -> f64 {
    assert_eq!(y_true.numel(), y_pred.numel());
    let n = y_true.numel();
    let sum: f64 = y_true
        .data()
        .iter()
        .zip(y_pred.data().iter())
        .map(|(&t, &p)| {
            let d = (t - p).to_f64();
            d * d
        })
        .sum();
    sum / n as f64
}

/// Root Mean Squared Error.
pub fn rmse<T: Float>(y_true: &Tensor<T>, y_pred: &Tensor<T>) -> f64 {
    mse(y_true, y_pred).sqrt()
}

/// Mean Absolute Error.
pub fn mae<T: Float>(y_true: &Tensor<T>, y_pred: &Tensor<T>) -> f64 {
    assert_eq!(y_true.numel(), y_pred.numel());
    let n = y_true.numel();
    let sum: f64 = y_true
        .data()
        .iter()
        .zip(y_pred.data().iter())
        .map(|(&t, &p)| (t - p).to_f64().abs())
        .sum();
    sum / n as f64
}

/// R² (coefficient of determination).
pub fn r2_score<T: Float>(y_true: &Tensor<T>, y_pred: &Tensor<T>) -> f64 {
    let n = y_true.numel() as f64;
    let mean_true: f64 = y_true.data().iter().map(|v| v.to_f64()).sum::<f64>() / n;

    let ss_res: f64 = y_true
        .data()
        .iter()
        .zip(y_pred.data().iter())
        .map(|(&t, &p)| {
            let d = t.to_f64() - p.to_f64();
            d * d
        })
        .sum();

    let ss_tot: f64 = y_true
        .data()
        .iter()
        .map(|&t| {
            let d = t.to_f64() - mean_true;
            d * d
        })
        .sum();

    if ss_tot < 1e-15 {
        return 0.0;
    }
    1.0 - ss_res / ss_tot
}

/// Adjusted R² — R² adjusted for number of predictors.
///
/// adj_R² = 1 - (1 - R²) * (n - 1) / (n - p - 1)
pub fn adjusted_r2<T: Float>(y_true: &Tensor<T>, y_pred: &Tensor<T>, n_features: usize) -> f64 {
    let r2 = r2_score(y_true, y_pred);
    let n = y_true.numel() as f64;
    let p = n_features as f64;
    if n - p - 1.0 <= 0.0 { return r2; }
    1.0 - (1.0 - r2) * (n - 1.0) / (n - p - 1.0)
}

/// Mean Absolute Percentage Error.
///
/// MAPE = mean(|y - ŷ| / |y|) * 100
pub fn mape<T: Float>(y_true: &Tensor<T>, y_pred: &Tensor<T>) -> f64 {
    let n = y_true.numel();
    let sum: f64 = y_true.data().iter().zip(y_pred.data().iter())
        .map(|(&t, &p)| {
            let t_f = t.to_f64();
            if t_f.abs() < 1e-15 { 0.0 }
            else { ((t_f - p.to_f64()) / t_f).abs() }
        })
        .sum();
    (sum / n as f64) * 100.0
}

/// Mean Squared Log Error.
///
/// MSLE = mean((log(1 + y) - log(1 + ŷ))²)
pub fn msle<T: Float>(y_true: &Tensor<T>, y_pred: &Tensor<T>) -> f64 {
    let n = y_true.numel();
    let sum: f64 = y_true.data().iter().zip(y_pred.data().iter())
        .map(|(&t, &p)| {
            let lt = (1.0 + t.to_f64().max(0.0)).ln();
            let lp = (1.0 + p.to_f64().max(0.0)).ln();
            (lt - lp) * (lt - lp)
        })
        .sum();
    sum / n as f64
}

/// Explained Variance Score.
///
/// EV = 1 - Var(y - ŷ) / Var(y)
pub fn explained_variance<T: Float>(y_true: &Tensor<T>, y_pred: &Tensor<T>) -> f64 {
    let n = y_true.numel() as f64;
    let residuals: Vec<f64> = y_true.data().iter().zip(y_pred.data().iter())
        .map(|(&t, &p)| t.to_f64() - p.to_f64())
        .collect();

    let res_mean = residuals.iter().sum::<f64>() / n;
    let var_res = residuals.iter().map(|&r| (r - res_mean) * (r - res_mean)).sum::<f64>() / n;

    let y_mean: f64 = y_true.data().iter().map(|v| v.to_f64()).sum::<f64>() / n;
    let var_y: f64 = y_true.data().iter().map(|v| { let d = v.to_f64() - y_mean; d * d }).sum::<f64>() / n;

    if var_y < 1e-15 { return 0.0; }
    1.0 - var_res / var_y
}

/// Maximum absolute error.
pub fn max_error<T: Float>(y_true: &Tensor<T>, y_pred: &Tensor<T>) -> f64 {
    y_true.data().iter().zip(y_pred.data().iter())
        .map(|(&t, &p)| (t.to_f64() - p.to_f64()).abs())
        .fold(0.0_f64, f64::max)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse() {
        let y_true: Tensor<f64> = Tensor::from_slice(&[1.0, 2.0, 3.0]);
        let y_pred: Tensor<f64> = Tensor::from_slice(&[1.0, 2.0, 3.0]);
        assert!((mse(&y_true, &y_pred)).abs() < 1e-10);
    }

    #[test]
    fn test_r2_perfect() {
        let y_true: Tensor<f64> = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let y_pred: Tensor<f64> = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let r2 = r2_score(&y_true, &y_pred);
        assert!((r2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mae() {
        let y_true: Tensor<f64> = Tensor::from_slice(&[1.0, 2.0, 3.0]);
        let y_pred: Tensor<f64> = Tensor::from_slice(&[1.5, 2.5, 3.5]);
        assert!((mae(&y_true, &y_pred) - 0.5).abs() < 1e-10);
    }
}
