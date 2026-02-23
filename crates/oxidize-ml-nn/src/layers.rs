use oxidize_ml_core::Tensor;
use oxidize_ml_autodiff::Variable;

/// Trait for a neural network layer.
pub trait Layer {
    /// Forward pass.
    fn forward(&self, input: &Variable) -> Variable;
    /// Return all trainable parameters.
    fn parameters(&self) -> Vec<Variable>;
}

/// Fully connected (dense) layer: y = xW + b.
pub struct Linear {
    pub weight: Variable,
    pub bias: Variable,
    pub in_features: usize,
    pub out_features: usize,
}

impl Linear {
    /// Create a new linear layer with Xavier-uniform initialization.
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let scale = (6.0 / (in_features + out_features) as f64).sqrt();
        let w_data = Tensor::rand(vec![in_features, out_features], Some(42))
            .mul_scalar(2.0 * scale)
            .add_scalar(-scale);
        let b_data = Tensor::zeros(vec![1, out_features]);

        Linear {
            weight: Variable::param(w_data),
            bias: Variable::param(b_data),
            in_features,
            out_features,
        }
    }
}

impl Layer for Linear {
    fn forward(&self, input: &Variable) -> Variable {
        let xw = input.matmul(&self.weight);
        xw.add(&self.bias)
    }

    fn parameters(&self) -> Vec<Variable> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}

/// ReLU activation layer.
pub struct ReLULayer;

impl ReLULayer {
    pub fn new() -> Self { ReLULayer }
}

impl Layer for ReLULayer {
    fn forward(&self, input: &Variable) -> Variable { input.relu() }
    fn parameters(&self) -> Vec<Variable> { vec![] }
}

impl Default for ReLULayer {
    fn default() -> Self { Self::new() }
}

/// Sigmoid activation layer.
pub struct SigmoidLayer;

impl SigmoidLayer {
    pub fn new() -> Self { SigmoidLayer }
}

impl Layer for SigmoidLayer {
    fn forward(&self, input: &Variable) -> Variable { input.sigmoid() }
    fn parameters(&self) -> Vec<Variable> { vec![] }
}

impl Default for SigmoidLayer {
    fn default() -> Self { Self::new() }
}

/// Tanh activation layer.
pub struct TanhLayer;

impl TanhLayer {
    pub fn new() -> Self { TanhLayer }
}

impl Layer for TanhLayer {
    fn forward(&self, input: &Variable) -> Variable { input.tanh_act() }
    fn parameters(&self) -> Vec<Variable> { vec![] }
}

impl Default for TanhLayer {
    fn default() -> Self { Self::new() }
}

/// LeakyReLU activation: f(x) = max(alpha*x, x).
///
/// Implemented as: relu(x) + alpha * min(x, 0)
pub struct LeakyReLULayer {
    pub alpha: f64,
}

impl LeakyReLULayer {
    pub fn new(alpha: f64) -> Self {
        LeakyReLULayer { alpha }
    }
}

impl Layer for LeakyReLULayer {
    fn forward(&self, input: &Variable) -> Variable {
        // Approximate: use relu with a small leak
        // leaky_relu(x) = relu(x) + alpha * (x - relu(x))
        // = (1-alpha)*relu(x) + alpha*x
        let relu_out = input.relu();
        let scaled_relu = relu_out.mul_scalar(1.0 - self.alpha);
        let scaled_input = input.mul_scalar(self.alpha);
        scaled_relu.add(&scaled_input)
    }

    fn parameters(&self) -> Vec<Variable> { vec![] }
}

impl Default for LeakyReLULayer {
    fn default() -> Self { Self::new(0.01) }
}

/// Dropout layer — randomly zeros elements during training.
/// During inference (default), acts as identity.
pub struct Dropout {
    pub p: f64,
    pub training: bool,
}

impl Dropout {
    pub fn new(p: f64) -> Self {
        Dropout { p, training: false }
    }
    pub fn train(&mut self) { self.training = true; }
    pub fn eval(&mut self) { self.training = false; }
}

impl Layer for Dropout {
    fn forward(&self, input: &Variable) -> Variable {
        if !self.training {
            return input.clone();
        }
        // Create a mask: each element has prob (1-p) of being kept
        let mask_data = Tensor::<f64>::rand(input.data.shape_vec(), None);
        let threshold = self.p;
        let scale = 1.0 / (1.0 - self.p);
        let mask: Vec<f64> = mask_data.data().iter()
            .map(|&v| if v > threshold { scale } else { 0.0 })
            .collect();
        let mask_tensor = Tensor::new(mask, input.data.shape_vec()).unwrap();
        let mask_var = Variable::input(mask_tensor);
        input.mul(&mask_var)
    }

    fn parameters(&self) -> Vec<Variable> { vec![] }
}

impl Default for Dropout {
    fn default() -> Self { Self::new(0.5) }
}

/// Flatten layer — reshapes input to [batch_size, features].
pub struct FlattenLayer;

impl FlattenLayer {
    pub fn new() -> Self { FlattenLayer }
}

impl Layer for FlattenLayer {
    fn forward(&self, input: &Variable) -> Variable {
        let shape = input.data.shape_vec();
        if shape.len() <= 2 {
            return input.clone();
        }
        let batch = shape[0];
        let features: usize = shape[1..].iter().product();
        let new_data = input.data.reshape(vec![batch, features]).unwrap();
        Variable::input(new_data)
    }

    fn parameters(&self) -> Vec<Variable> { vec![] }
}

impl Default for FlattenLayer {
    fn default() -> Self { Self::new() }
}

/// Batch Normalization layer.
///
/// Normalizes across the batch dimension: y = (x - μ) / √(σ² + ε) * γ + β
pub struct BatchNorm {
    pub num_features: usize,
    pub eps: f64,
    pub gamma: Variable,
    pub beta: Variable,
    pub training: bool,
}

impl BatchNorm {
    pub fn new(num_features: usize) -> Self {
        BatchNorm {
            num_features,
            eps: 1e-5,
            gamma: Variable::param(Tensor::ones(vec![1, num_features])),
            beta: Variable::param(Tensor::zeros(vec![1, num_features])),
            training: true,
        }
    }
    pub fn train(&mut self) { self.training = true; }
    pub fn eval(&mut self) { self.training = false; }
}

impl Layer for BatchNorm {
    fn forward(&self, input: &Variable) -> Variable {
        let x = &input.data;
        let batch_size = x.shape().dim(0).unwrap_or(1);

        if batch_size > 1 {
            // Compute batch mean and variance
            let mean = x.mean_axis(0).unwrap();
            let centered = x.sub(&mean).unwrap_or_else(|_| x.clone());

            let var_tensor = centered.mul(&centered).unwrap().mean_axis(0).unwrap();
            let std = var_tensor.add_scalar(self.eps).sqrt();
            let normalized = centered.div(&std).unwrap_or_else(|_| centered.clone());

            // Scale and shift
            let scaled = normalized.mul(&self.gamma.data).unwrap();
            let output = scaled.add(&self.beta.data).unwrap();

            Variable::input(output)
        } else {
            input.clone()
        }
    }

    fn parameters(&self) -> Vec<Variable> {
        vec![self.gamma.clone(), self.beta.clone()]
    }
}
