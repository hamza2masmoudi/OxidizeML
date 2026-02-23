use oxidize_ml_core::Tensor;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Load the Iris dataset (150 samples, 4 features, 3 classes).
pub fn load_iris() -> (Tensor<f64>, Tensor<f64>) {
    // Hardcoded Iris data — subset for compactness, full 150 in practice
    // 4 features: sepal_length, sepal_width, petal_length, petal_width
    let features: Vec<f64> = vec![
        // Setosa (class 0) - 10 samples
        5.1,3.5,1.4,0.2, 4.9,3.0,1.4,0.2, 4.7,3.2,1.3,0.2, 4.6,3.1,1.5,0.2,
        5.0,3.6,1.4,0.2, 5.4,3.9,1.7,0.4, 4.6,3.4,1.4,0.3, 5.0,3.4,1.5,0.2,
        4.4,2.9,1.4,0.2, 4.9,3.1,1.5,0.1,
        // Versicolor (class 1) - 10 samples
        7.0,3.2,4.7,1.4, 6.4,3.2,4.5,1.5, 6.9,3.1,4.9,1.5, 5.5,2.3,4.0,1.3,
        6.5,2.8,4.6,1.5, 5.7,2.8,4.5,1.3, 6.3,3.3,4.7,1.6, 4.9,2.4,3.3,1.0,
        6.6,2.9,4.6,1.3, 5.2,2.7,3.9,1.4,
        // Virginica (class 2) - 10 samples
        6.3,3.3,6.0,2.5, 5.8,2.7,5.1,1.9, 7.1,3.0,5.9,2.1, 6.3,2.9,5.6,1.8,
        6.5,3.0,5.8,2.2, 7.6,3.0,6.6,2.1, 4.9,2.5,4.5,1.7, 7.3,2.9,6.3,1.8,
        6.7,2.5,5.8,1.8, 7.2,3.6,6.1,2.5,
    ];
    let labels: Vec<f64> = vec![
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
        1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,
        2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,
    ];

    (
        Tensor::new(features, vec![30, 4]).expect("iris features"),
        Tensor::new(labels, vec![30]).expect("iris labels"),
    )
}

/// Generate synthetic classification data (Gaussian blobs).
pub fn make_blobs(
    n_samples: usize,
    n_features: usize,
    n_centers: usize,
    cluster_std: f64,
    seed: Option<u64>,
) -> (Tensor<f64>, Tensor<f64>) {
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    // Generate center positions spread out
    let mut centers = vec![0.0; n_centers * n_features];
    for c in 0..n_centers {
        for f in 0..n_features {
            centers[c * n_features + f] = (c as f64) * 5.0 + rand::Rng::gen::<f64>(&mut rng);
        }
    }

    let samples_per_center = n_samples / n_centers;
    let mut features = Vec::with_capacity(n_samples * n_features);
    let mut labels = Vec::with_capacity(n_samples);

    for c in 0..n_centers {
        let actual_samples = if c == n_centers - 1 {
            n_samples - samples_per_center * (n_centers - 1)
        } else {
            samples_per_center
        };

        for _ in 0..actual_samples {
            for f in 0..n_features {
                // Box-Muller for normal distribution
                let u1: f64 = rand::Rng::gen::<f64>(&mut rng).max(1e-10);
                let u2: f64 = rand::Rng::gen::<f64>(&mut rng);
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                features.push(centers[c * n_features + f] + z * cluster_std);
            }
            labels.push(c as f64);
        }
    }

    let n = labels.len();
    (
        Tensor::new(features, vec![n, n_features]).expect("blobs features"),
        Tensor::new(labels, vec![n]).expect("blobs labels"),
    )
}

/// Generate synthetic regression data: y = Xw + noise.
pub fn make_regression(
    n_samples: usize,
    n_features: usize,
    noise: f64,
    seed: Option<u64>,
) -> (Tensor<f64>, Tensor<f64>) {
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    // Random true weights
    let true_weights: Vec<f64> = (0..n_features)
        .map(|_| rand::Rng::gen::<f64>(&mut rng) * 10.0 - 5.0)
        .collect();

    let mut features = Vec::with_capacity(n_samples * n_features);
    let mut labels = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let mut y = 0.0;
        for f in 0..n_features {
            let x: f64 = rand::Rng::gen::<f64>(&mut rng) * 2.0 - 1.0;
            features.push(x);
            y += x * true_weights[f];
        }
        // Add noise
        let u1: f64 = rand::Rng::gen::<f64>(&mut rng).max(1e-10);
        let u2: f64 = rand::Rng::gen::<f64>(&mut rng);
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        labels.push(y + z * noise);
    }

    (
        Tensor::new(features, vec![n_samples, n_features]).expect("regression features"),
        Tensor::new(labels, vec![n_samples]).expect("regression labels"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_iris() {
        let (x, y) = load_iris();
        assert_eq!(x.shape_vec(), vec![30, 4]);
        assert_eq!(y.numel(), 30);
    }

    #[test]
    fn test_make_blobs() {
        let (x, y) = make_blobs(100, 2, 3, 0.5, Some(42));
        assert_eq!(x.shape_vec(), vec![100, 2]);
        assert_eq!(y.numel(), 100);
    }

    #[test]
    fn test_make_regression() {
        let (x, y) = make_regression(50, 3, 0.1, Some(42));
        assert_eq!(x.shape_vec(), vec![50, 3]);
        assert_eq!(y.numel(), 50);
    }
}
