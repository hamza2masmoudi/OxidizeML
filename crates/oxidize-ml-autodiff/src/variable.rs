use oxidize_ml_core::Tensor;
use crate::graph::{Graph, NodeId, Op, with_graph};

/// A variable in the computation graph — wraps a tensor with grad tracking.
#[derive(Debug, Clone)]
pub struct Variable {
    pub node_id: NodeId,
    pub data: Tensor<f64>,
}

impl Variable {
    /// Create a new leaf variable (parameter) that requires gradients.
    pub fn new(data: Tensor<f64>, requires_grad: bool) -> Self {
        let node_id = with_graph(|g| g.add_node(Op::Leaf, data.clone(), requires_grad));
        Variable { node_id, data }
    }

    /// Create a parameter (requires grad by default).
    pub fn param(data: Tensor<f64>) -> Self {
        Self::new(data, true)
    }

    /// Create an input (no grad by default).
    pub fn input(data: Tensor<f64>) -> Self {
        Self::new(data, false)
    }

    pub fn shape_vec(&self) -> Vec<usize> {
        self.data.shape_vec()
    }

    pub fn numel(&self) -> usize {
        self.data.numel()
    }

    /// Element-wise addition.
    pub fn add(&self, other: &Variable) -> Variable {
        let result = self.data.add(&other.data).expect("add: shape mismatch");
        let node_id = with_graph(|g| {
            g.add_node(Op::Add(self.node_id, other.node_id), result.clone(), true)
        });
        Variable {
            node_id,
            data: result,
        }
    }

    /// Element-wise subtraction.
    pub fn sub(&self, other: &Variable) -> Variable {
        let result = self.data.sub(&other.data).expect("sub: shape mismatch");
        let node_id = with_graph(|g| {
            g.add_node(Op::Sub(self.node_id, other.node_id), result.clone(), true)
        });
        Variable {
            node_id,
            data: result,
        }
    }

    /// Element-wise multiplication.
    pub fn mul(&self, other: &Variable) -> Variable {
        let result = self.data.mul(&other.data).expect("mul: shape mismatch");
        let node_id = with_graph(|g| {
            g.add_node(Op::Mul(self.node_id, other.node_id), result.clone(), true)
        });
        Variable {
            node_id,
            data: result,
        }
    }

    /// Element-wise division.
    pub fn div(&self, other: &Variable) -> Variable {
        let result = self.data.div(&other.data).expect("div: shape mismatch");
        let node_id = with_graph(|g| {
            g.add_node(Op::Div(self.node_id, other.node_id), result.clone(), true)
        });
        Variable {
            node_id,
            data: result,
        }
    }

    /// Matrix multiplication.
    pub fn matmul(&self, other: &Variable) -> Variable {
        let result = self.data.matmul(&other.data).expect("matmul: shape mismatch");
        let node_id = with_graph(|g| {
            g.add_node(Op::MatMul(self.node_id, other.node_id), result.clone(), true)
        });
        Variable {
            node_id,
            data: result,
        }
    }

    /// Negation.
    pub fn neg(&self) -> Variable {
        let result = self.data.mul_scalar(oxidize_ml_core::Float::NEG_ONE);
        let node_id = with_graph(|g| g.add_node(Op::Neg(self.node_id), result.clone(), true));
        Variable {
            node_id,
            data: result,
        }
    }

    /// Exponential.
    pub fn exp(&self) -> Variable {
        let result = self.data.exp();
        let node_id = with_graph(|g| g.add_node(Op::Exp(self.node_id), result.clone(), true));
        Variable {
            node_id,
            data: result,
        }
    }

    /// Natural logarithm.
    pub fn ln(&self) -> Variable {
        let result = self.data.ln();
        let node_id = with_graph(|g| g.add_node(Op::Ln(self.node_id), result.clone(), true));
        Variable {
            node_id,
            data: result,
        }
    }

    /// ReLU activation.
    pub fn relu(&self) -> Variable {
        let result = self.data.relu();
        let node_id = with_graph(|g| g.add_node(Op::Relu(self.node_id), result.clone(), true));
        Variable {
            node_id,
            data: result,
        }
    }

    /// Sigmoid activation.
    pub fn sigmoid(&self) -> Variable {
        let result = self.data.sigmoid();
        let node_id =
            with_graph(|g| g.add_node(Op::Sigmoid(self.node_id), result.clone(), true));
        Variable {
            node_id,
            data: result,
        }
    }

    /// Tanh activation.
    pub fn tanh_act(&self) -> Variable {
        let result = self.data.tanh_elem();
        let node_id = with_graph(|g| g.add_node(Op::Tanh(self.node_id), result.clone(), true));
        Variable {
            node_id,
            data: result,
        }
    }

    /// Multiply by scalar.
    pub fn mul_scalar(&self, s: f64) -> Variable {
        let result = self.data.mul_scalar(s);
        let node_id = with_graph(|g| {
            g.add_node(Op::MulScalar(self.node_id, s), result.clone(), true)
        });
        Variable {
            node_id,
            data: result,
        }
    }

    /// Add scalar.
    pub fn add_scalar(&self, s: f64) -> Variable {
        let result = self.data.add_scalar(s);
        let node_id = with_graph(|g| {
            g.add_node(Op::AddScalar(self.node_id, s), result.clone(), true)
        });
        Variable {
            node_id,
            data: result,
        }
    }

    /// Sum all elements to a scalar.
    pub fn sum(&self) -> Variable {
        let s = self.data.sum_all();
        let result = Tensor::scalar(s);
        let node_id =
            with_graph(|g| g.add_node(Op::SumAll(self.node_id), result.clone(), true));
        Variable {
            node_id,
            data: result,
        }
    }

    /// Mean of all elements.
    pub fn mean(&self) -> Variable {
        let m = self.data.mean_all();
        let result = Tensor::scalar(m);
        let node_id =
            with_graph(|g| g.add_node(Op::MeanAll(self.node_id), result.clone(), true));
        Variable {
            node_id,
            data: result,
        }
    }

    /// Transpose last two dims.
    pub fn t(&self) -> Variable {
        let result = self.data.t().expect("transpose failed");
        let node_id =
            with_graph(|g| g.add_node(Op::Transpose(self.node_id), result.clone(), true));
        Variable {
            node_id,
            data: result,
        }
    }

    /// Power (element-wise, scalar exponent).
    pub fn pow(&self, n: f64) -> Variable {
        let result = self.data.powf(oxidize_ml_core::Float::from_f64(n));
        let node_id = with_graph(|g| {
            g.add_node(Op::Pow(self.node_id, n), result.clone(), true)
        });
        Variable {
            node_id,
            data: result,
        }
    }
}
