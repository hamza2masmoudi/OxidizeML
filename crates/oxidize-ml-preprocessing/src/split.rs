use oxidize_ml_core::{Float, Tensor};
use oxidize_ml_core::error::TensorResult;
use rand::distributions::{Distribution, Standard};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

/// Split data into training and test sets.
///
/// Returns `(X_train, X_test, y_train, y_test)`.
pub fn train_test_split<T: Float>(
    x: &Tensor<T>,
    y: &Tensor<T>,
    test_ratio: f64,
    seed: Option<u64>,
) -> TensorResult<(Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>)>
where
    Standard: Distribution<T>,
{
    let n = x.shape().dim(0)?;
    assert_eq!(n, y.numel(), "X rows must match y length");

    let mut indices: Vec<usize> = (0..n).collect();
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };
    indices.shuffle(&mut rng);

    let test_size = (n as f64 * test_ratio).round() as usize;
    let train_size = n - test_size;

    let cols = x.shape().dim(1)?;

    // Build train/test X
    let mut x_train_data = Vec::with_capacity(train_size * cols);
    let mut x_test_data = Vec::with_capacity(test_size * cols);
    let mut y_train_data = Vec::with_capacity(train_size);
    let mut y_test_data = Vec::with_capacity(test_size);

    for &idx in &indices[..train_size] {
        for j in 0..cols {
            x_train_data.push(x.get(&[idx, j])?);
        }
        y_train_data.push(y.data()[idx]);
    }
    for &idx in &indices[train_size..] {
        for j in 0..cols {
            x_test_data.push(x.get(&[idx, j])?);
        }
        y_test_data.push(y.data()[idx]);
    }

    Ok((
        Tensor::new(x_train_data, vec![train_size, cols])?,
        Tensor::new(x_test_data, vec![test_size, cols])?,
        Tensor::new(y_train_data, vec![train_size])?,
        Tensor::new(y_test_data, vec![test_size])?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train_test_split() {
        let x: Tensor<f64> = Tensor::from_vec2d(&[
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
            vec![7.0, 8.0],
            vec![9.0, 10.0],
        ]).unwrap();
        let y: Tensor<f64> = Tensor::from_slice(&[0.0, 1.0, 0.0, 1.0, 0.0]);

        let (x_train, x_test, y_train, y_test) =
            train_test_split(&x, &y, 0.4, Some(42)).unwrap();

        assert_eq!(x_train.shape().dim(0).unwrap(), 3);
        assert_eq!(x_test.shape().dim(0).unwrap(), 2);
        assert_eq!(y_train.numel(), 3);
        assert_eq!(y_test.numel(), 2);
    }
}
