use oxidize_ml_core::{Tensor, TensorError};
use oxidize_ml_core::error::TensorResult;

/// Trait for unsupervised transformers (scalers, encoders, etc.).
pub trait Transformer {
    fn fit(&mut self, x: &Tensor<f64>) -> TensorResult<()>;
    fn transform(&self, x: &Tensor<f64>) -> TensorResult<Tensor<f64>>;
    fn fit_transform(&mut self, x: &Tensor<f64>) -> TensorResult<Tensor<f64>> {
        self.fit(x)?;
        self.transform(x)
    }
}

/// Trait for supervised estimators.
pub trait Estimator {
    fn fit(&mut self, x: &Tensor<f64>, y: &Tensor<f64>) -> TensorResult<()>;
    fn predict(&self, x: &Tensor<f64>) -> TensorResult<Tensor<f64>>;
}

/// A machine learning pipeline: chain transformers + final estimator.
pub struct Pipeline {
    transformers: Vec<Box<dyn Transformer>>,
    estimator: Option<Box<dyn Estimator>>,
}

impl Pipeline {
    pub fn new() -> Self {
        Pipeline {
            transformers: Vec::new(),
            estimator: None,
        }
    }

    /// Add a transformer step.
    pub fn add_transformer(mut self, transformer: Box<dyn Transformer>) -> Self {
        self.transformers.push(transformer);
        self
    }

    /// Set the final estimator.
    pub fn set_estimator(mut self, estimator: Box<dyn Estimator>) -> Self {
        self.estimator = Some(estimator);
        self
    }

    /// Fit all transformers and the estimator.
    pub fn fit(&mut self, x: &Tensor<f64>, y: &Tensor<f64>) -> TensorResult<()> {
        let mut current_x = x.clone();

        for t in &mut self.transformers {
            current_x = t.fit_transform(&current_x)?;
        }

        if let Some(est) = &mut self.estimator {
            est.fit(&current_x, y)?;
        }

        Ok(())
    }

    /// Transform through all transformers and predict with the estimator.
    pub fn predict(&self, x: &Tensor<f64>) -> TensorResult<Tensor<f64>> {
        let mut current_x = x.clone();

        for t in &self.transformers {
            current_x = t.transform(&current_x)?;
        }

        match &self.estimator {
            Some(est) => est.predict(&current_x),
            None => Err(TensorError::InvalidOperation("No estimator set".into())),
        }
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}
