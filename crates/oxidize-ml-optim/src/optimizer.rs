use std::collections::HashMap;
use oxidize_ml_core::Tensor;
use oxidize_ml_autodiff::graph::NodeId;

/// Trait for optimizers.
pub trait Optimizer {
    /// Perform one optimization step using computed gradients.
    fn step(&mut self, grads: &HashMap<NodeId, Tensor<f64>>) -> Vec<Tensor<f64>>;
}

/// Stochastic Gradient Descent with optional momentum.
pub struct SGD {
    pub lr: f64,
    pub momentum: f64,
    param_ids: Vec<NodeId>,
    param_values: Vec<Tensor<f64>>,
    velocities: Vec<Tensor<f64>>,
}

impl SGD {
    pub fn new(param_ids: Vec<NodeId>, param_values: Vec<Tensor<f64>>, lr: f64, momentum: f64) -> Self {
        let velocities = param_values.iter().map(|p| Tensor::zeros(p.shape_vec())).collect();
        SGD {
            lr,
            momentum,
            param_ids,
            param_values,
            velocities,
        }
    }
}

impl Optimizer for SGD {
    fn step(&mut self, grads: &HashMap<NodeId, Tensor<f64>>) -> Vec<Tensor<f64>> {
        for (i, id) in self.param_ids.iter().enumerate() {
            if let Some(grad) = grads.get(id) {
                // v = momentum * v - lr * grad
                self.velocities[i] = self.velocities[i]
                    .mul_scalar(self.momentum)
                    .sub(&grad.mul_scalar(self.lr))
                    .expect("sgd velocity update");

                // param += v
                self.param_values[i] = self.param_values[i]
                    .add(&self.velocities[i])
                    .expect("sgd param update");
            }
        }
        self.param_values.clone()
    }
}

/// Adam optimizer.
pub struct Adam {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub t: usize,
    param_ids: Vec<NodeId>,
    param_values: Vec<Tensor<f64>>,
    m: Vec<Tensor<f64>>, // first moment
    v: Vec<Tensor<f64>>, // second moment
}

impl Adam {
    pub fn new(param_ids: Vec<NodeId>, param_values: Vec<Tensor<f64>>, lr: f64) -> Self {
        let m = param_values.iter().map(|p| Tensor::zeros(p.shape_vec())).collect();
        let v = param_values.iter().map(|p| Tensor::zeros(p.shape_vec())).collect();
        Adam {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            t: 0,
            param_ids,
            param_values,
            m,
            v,
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self, grads: &HashMap<NodeId, Tensor<f64>>) -> Vec<Tensor<f64>> {
        self.t += 1;
        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        for (i, id) in self.param_ids.iter().enumerate() {
            if let Some(grad) = grads.get(id) {
                self.m[i] = self.m[i]
                    .mul_scalar(self.beta1)
                    .add(&grad.mul_scalar(1.0 - self.beta1))
                    .expect("adam m update");

                let grad_sq = grad.mul(grad).expect("grad²");
                self.v[i] = self.v[i]
                    .mul_scalar(self.beta2)
                    .add(&grad_sq.mul_scalar(1.0 - self.beta2))
                    .expect("adam v update");

                let m_hat = self.m[i].mul_scalar(1.0 / bias_correction1);
                let v_hat = self.v[i].mul_scalar(1.0 / bias_correction2);

                let denom = v_hat.sqrt().add_scalar(self.epsilon);
                let update = m_hat.div(&denom).expect("adam update").mul_scalar(self.lr);
                self.param_values[i] = self.param_values[i]
                    .sub(&update)
                    .expect("adam param update");
            }
        }
        self.param_values.clone()
    }
}

/// RMSProp optimizer.
///
/// Maintains a running average of squared gradients and normalizes
/// the gradient by this average. This helps with non-stationary objectives.
///
/// v = α * v + (1 - α) * grad²
/// param -= lr * grad / (√v + ε)
pub struct RMSProp {
    pub lr: f64,
    pub alpha: f64,
    pub epsilon: f64,
    pub weight_decay: f64,
    param_ids: Vec<NodeId>,
    param_values: Vec<Tensor<f64>>,
    v: Vec<Tensor<f64>>,
}

impl RMSProp {
    pub fn new(param_ids: Vec<NodeId>, param_values: Vec<Tensor<f64>>, lr: f64) -> Self {
        let v = param_values.iter().map(|p| Tensor::zeros(p.shape_vec())).collect();
        RMSProp {
            lr,
            alpha: 0.99,
            epsilon: 1e-8,
            weight_decay: 0.0,
            param_ids,
            param_values,
            v,
        }
    }

    pub fn with_weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }
}

impl Optimizer for RMSProp {
    fn step(&mut self, grads: &HashMap<NodeId, Tensor<f64>>) -> Vec<Tensor<f64>> {
        for (i, id) in self.param_ids.iter().enumerate() {
            if let Some(grad) = grads.get(id) {
                // Apply weight decay
                let grad = if self.weight_decay > 0.0 {
                    grad.add(&self.param_values[i].mul_scalar(self.weight_decay))
                        .expect("weight decay")
                } else {
                    grad.clone()
                };

                // v = α * v + (1 - α) * grad²
                let grad_sq = grad.mul(&grad).expect("grad²");
                self.v[i] = self.v[i]
                    .mul_scalar(self.alpha)
                    .add(&grad_sq.mul_scalar(1.0 - self.alpha))
                    .expect("rmsprop v update");

                // param -= lr * grad / (√v + ε)
                let denom = self.v[i].sqrt().add_scalar(self.epsilon);
                let update = grad.div(&denom).expect("rmsprop div").mul_scalar(self.lr);
                self.param_values[i] = self.param_values[i]
                    .sub(&update)
                    .expect("rmsprop param update");
            }
        }
        self.param_values.clone()
    }
}

/// AdaGrad optimizer.
///
/// Adapts the learning rate per parameter based on historical gradients.
/// Good for sparse data.
///
/// G += grad²
/// param -= lr * grad / (√G + ε)
pub struct AdaGrad {
    pub lr: f64,
    pub epsilon: f64,
    param_ids: Vec<NodeId>,
    param_values: Vec<Tensor<f64>>,
    g: Vec<Tensor<f64>>,
}

impl AdaGrad {
    pub fn new(param_ids: Vec<NodeId>, param_values: Vec<Tensor<f64>>, lr: f64) -> Self {
        let g = param_values.iter().map(|p| Tensor::zeros(p.shape_vec())).collect();
        AdaGrad {
            lr,
            epsilon: 1e-8,
            param_ids,
            param_values,
            g,
        }
    }
}

impl Optimizer for AdaGrad {
    fn step(&mut self, grads: &HashMap<NodeId, Tensor<f64>>) -> Vec<Tensor<f64>> {
        for (i, id) in self.param_ids.iter().enumerate() {
            if let Some(grad) = grads.get(id) {
                // G += grad²
                let grad_sq = grad.mul(grad).expect("grad²");
                self.g[i] = self.g[i].add(&grad_sq).expect("adagrad g update");

                // param -= lr * grad / (√G + ε)
                let denom = self.g[i].sqrt().add_scalar(self.epsilon);
                let update = grad.div(&denom).expect("adagrad div").mul_scalar(self.lr);
                self.param_values[i] = self.param_values[i]
                    .sub(&update)
                    .expect("adagrad param update");
            }
        }
        self.param_values.clone()
    }
}

