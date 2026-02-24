use oximl_core::{Tensor, TensorResult};
use crate::graph::{Graph, NodeId, Op};
use std::sync::Arc;

/// A differentiable variable backed by the thread-safe computation graph.
#[derive(Debug, Clone)]
pub struct Variable {
    pub node_id: NodeId,
    pub data: Tensor,
    pub graph: Arc<Graph>,
}

impl Variable {
    /// Create a new leaf parameter that requires gradients.
    pub fn param(data: Tensor, graph: Arc<Graph>) -> Self {
        let node_id = graph.push_node(Op::Leaf, data.clone(), true);
        Variable { node_id, data, graph }
    }

    /// Create a new input variable (e.g. training data) that does not compute gradients.
    pub fn input(data: Tensor, graph: Arc<Graph>) -> Self {
        let node_id = graph.push_node(Op::Leaf, data.clone(), false);
        Variable { node_id, data, graph }
    }

    /// Element-wise addition.
    pub fn add(&self, rhs: &Variable) -> TensorResult<Variable> {
        let out_data = (&self.data + &rhs.data)?;
        let requires_grad = self.get_node().requires_grad || rhs.get_node().requires_grad;
        
        let node_id = self.graph.push_node(
            Op::Add(self.node_id, rhs.node_id),
            out_data.clone(),
            requires_grad,
        );

        Ok(Variable { node_id, data: out_data, graph: self.graph.clone() })
    }

    /// Matrix multiplication.
    pub fn matmul(&self, rhs: &Variable) -> TensorResult<Variable> {
        let out_data = self.data.matmul(&rhs.data)?;
        let requires_grad = self.get_node().requires_grad || rhs.get_node().requires_grad;
        
        let node_id = self.graph.push_node(
            Op::MatMul(self.node_id, rhs.node_id),
            out_data.clone(),
            requires_grad,
        );

        Ok(Variable { node_id, data: out_data, graph: self.graph.clone() })
    }

    /// Transpose
    pub fn t(&self) -> Variable {
        let out_data = self.data.t();
        let requires_grad = self.get_node().requires_grad;
        let node_id = self.graph.push_node(Op::Transpose(self.node_id), out_data.clone(), requires_grad);
        Variable { node_id, data: out_data, graph: self.graph.clone() }
    }

    /// Reshape
    pub fn reshape(&self, shape: &[usize]) -> TensorResult<Variable> {
        let out_data = self.data.reshape(shape)?;
        let requires_grad = self.get_node().requires_grad;
        let node_id = self.graph.push_node(Op::Reshape(self.node_id), out_data.clone(), requires_grad);
        Ok(Variable { node_id, data: out_data, graph: self.graph.clone() })
    }

    /// Scalar multiplication
    pub fn scalar_mul(&self, scalar: f64) -> TensorResult<Variable> {
        let out_data = self.data.scalar_mul(scalar)?;
        let requires_grad = self.get_node().requires_grad;
        let node_id = self.graph.push_node(Op::ScalarMul(self.node_id, scalar), out_data.clone(), requires_grad);
        Ok(Variable { node_id, data: out_data, graph: self.graph.clone() })
    }

    /// Softmax
    pub fn softmax(&self) -> TensorResult<Variable> {
        let out_data = self.data.softmax()?;
        let requires_grad = self.get_node().requires_grad;
        let node_id = self.graph.push_node(Op::Softmax(self.node_id), out_data.clone(), requires_grad);
        Ok(Variable { node_id, data: out_data, graph: self.graph.clone() })
    }

    fn get_node(&self) -> crate::graph::Node {
        self.graph.get_node(self.node_id).expect("Node must exist in graph")
    }
}
