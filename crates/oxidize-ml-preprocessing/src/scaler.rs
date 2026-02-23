use oxidize_ml_core::{Float, Tensor};
use oxidize_ml_core::error::TensorResult;
use rand::distributions::{Distribution, Standard};

/// Standardize features by removing the mean and scaling to unit variance.
pub struct StandardScaler<T: Float> {
    pub mean: Option<Tensor<T>>,
    pub std: Option<Tensor<T>>,
}

impl<T: Float> StandardScaler<T>
where
    Standard: Distribution<T>,
{
    pub fn new() -> Self {
        StandardScaler {
            mean: None,
            std: None,
        }
    }

    /// Compute mean and std from training data (2D: [samples, features]).
    pub fn fit(&mut self, x: &Tensor<T>) -> TensorResult<()> {
        self.mean = Some(x.mean_axis(0)?);
        self.std = Some(x.std_axis(0)?);
        Ok(())
    }

    /// Transform data using fitted mean and std.
    pub fn transform(&self, x: &Tensor<T>) -> TensorResult<Tensor<T>> {
        let mean = self.mean.as_ref().expect("fit() must be called before transform()");
        let std = self.std.as_ref().expect("fit() must be called before transform()");

        let mean_2d = mean.unsqueeze(0)?;
        let std_2d = std.unsqueeze(0)?;

        // (x - mean) / std
        let centered = x.sub(&mean_2d)?;
        // Add epsilon to avoid division by zero
        let std_safe = std_2d.apply(|v| if v.abs() < T::EPSILON { T::ONE } else { v });
        centered.div(&std_safe)
    }

    /// Fit and transform in one step.
    pub fn fit_transform(&mut self, x: &Tensor<T>) -> TensorResult<Tensor<T>> {
        self.fit(x)?;
        self.transform(x)
    }
}

/// Scale features to [0, 1] range.
pub struct MinMaxScaler<T: Float> {
    pub min: Option<Tensor<T>>,
    pub max: Option<Tensor<T>>,
}

impl<T: Float> MinMaxScaler<T>
where
    Standard: Distribution<T>,
{
    pub fn new() -> Self {
        MinMaxScaler {
            min: None,
            max: None,
        }
    }

    pub fn fit(&mut self, x: &Tensor<T>) -> TensorResult<()> {
        let cols = x.shape().dim(1)?;
        let rows = x.shape().dim(0)?;

        let mut min_vals = vec![T::INFINITY; cols];
        let mut max_vals = vec![T::NEG_INFINITY; cols];

        for i in 0..rows {
            for j in 0..cols {
                let v = x.get(&[i, j])?;
                if v < min_vals[j] {
                    min_vals[j] = v;
                }
                if v > max_vals[j] {
                    max_vals[j] = v;
                }
            }
        }

        self.min = Some(Tensor::from_slice(&min_vals));
        self.max = Some(Tensor::from_slice(&max_vals));
        Ok(())
    }

    pub fn transform(&self, x: &Tensor<T>) -> TensorResult<Tensor<T>> {
        let min = self.min.as_ref().expect("fit() first").unsqueeze(0)?;
        let max = self.max.as_ref().expect("fit() first").unsqueeze(0)?;

        let range = max.sub(&min)?;
        let range_safe = range.apply(|v| if v.abs() < T::EPSILON { T::ONE } else { v });
        let centered = x.sub(&min)?;
        centered.div(&range_safe)
    }

    pub fn fit_transform(&mut self, x: &Tensor<T>) -> TensorResult<Tensor<T>> {
        self.fit(x)?;
        self.transform(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_scaler() {
        let x: Tensor<f64> = Tensor::from_vec2d(&[
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ]).unwrap();

        let mut scaler = StandardScaler::new();
        let transformed = scaler.fit_transform(&x).unwrap();

        // Mean should be ~0
        let mean = transformed.mean_axis(0).unwrap();
        assert!(mean.data()[0].abs() < 1e-10);
        assert!(mean.data()[1].abs() < 1e-10);
    }

    #[test]
    fn test_minmax_scaler() {
        let x: Tensor<f64> = Tensor::from_vec2d(&[
            vec![1.0, 10.0],
            vec![5.0, 20.0],
            vec![3.0, 30.0],
        ]).unwrap();

        let mut scaler = MinMaxScaler::new();
        let transformed = scaler.fit_transform(&x).unwrap();

        // Min should be 0, max should be 1
        for j in 0..2 {
            let col = transformed.col(j).unwrap();
            let min = col.min_all().unwrap();
            let max = col.max_all().unwrap();
            assert!(min.abs() < 1e-10);
            assert!((max - 1.0).abs() < 1e-10);
        }
    }
}
