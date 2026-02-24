use std::sync::{Arc, RwLock};
use oximl_core::Tensor;

pub type NodeId = usize;

#[derive(Debug, Clone)]
pub enum Op {
    Leaf,
    Add(NodeId, NodeId),
    Mul(NodeId, NodeId),
    MatMul(NodeId, NodeId),
    ScalarMul(NodeId, f64),
    Transpose(NodeId),
    Reshape(NodeId),
    Softmax(NodeId),
    Relu(NodeId),
    Exp(NodeId),
    Ln(NodeId),
    Div(NodeId, NodeId),
}

/// A node in the thread-safe computation graph.
#[derive(Debug, Clone)]
pub struct Node {
    pub id: NodeId,
    pub op: Op,
    pub data: Tensor,
    pub requires_grad: bool,
    pub grad: Option<Tensor>,
}

/// A thread-safe Computation Graph / Tape.
/// 
/// Uses `Arc<RwLock<Vec<Node>>>` to allow the graph to be shared safely
/// across threads, enabling PyTorch-style multi-threaded DDP (Distributed Data Parallel) training.
#[derive(Debug, Clone)]
pub struct Graph {
    nodes: Arc<RwLock<Vec<Node>>>,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl Graph {
    pub fn new() -> Self {
        Graph {
            nodes: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub fn push_node(&self, op: Op, data: Tensor, requires_grad: bool) -> NodeId {
        let mut nodes = self.nodes.write().unwrap();
        let id = nodes.len();
        nodes.push(Node {
            id,
            op,
            data,
            requires_grad,
            grad: None,
        });
        id
    }

    pub fn get_node(&self, id: NodeId) -> Option<Node> {
        let nodes = self.nodes.read().unwrap();
        nodes.get(id).cloned()
    }

    pub fn len(&self) -> usize {
        self.nodes.read().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.read().unwrap().is_empty()
    }

    /// Backpropagation: Computes gradients for all nodes up to `root_id`.
    /// Traverses the tape in reverse order. Locking the RwLock for write.
    pub fn backward(&self, root_id: NodeId) -> oximl_core::TensorResult<()> {
        let mut nodes = self.nodes.write().unwrap();
        
        let root_shape = nodes[root_id].data.shape().to_vec();
        let root_dtype = nodes[root_id].data.dtype();
        nodes[root_id].grad = Some(Tensor::ones(&root_shape, root_dtype));

        for i in (0..=root_id).rev() {
            if !nodes[i].requires_grad || nodes[i].grad.is_none() {
                continue;
            }
            
            let grad_i = nodes[i].grad.clone().unwrap();
            let op = nodes[i].op.clone();

            match op {
                Op::Add(lhs, rhs) => {
                    accumulate_grad(&mut nodes, lhs, &grad_i)?;
                    accumulate_grad(&mut nodes, rhs, &grad_i)?;
                }
                Op::Mul(lhs, rhs) => {
                    let grad_lhs = (&grad_i * &nodes[rhs].data)?;
                    let grad_rhs = (&grad_i * &nodes[lhs].data)?;
                    accumulate_grad(&mut nodes, lhs, &grad_lhs)?;
                    accumulate_grad(&mut nodes, rhs, &grad_rhs)?;
                }
                Op::MatMul(lhs, rhs) => {
                    // Grad(A) = dL/dY @ B^T
                    // Grad(B) = A^T @ dL/dY
                    let lhs_data_t = nodes[lhs].data.t();
                    let rhs_data_t = nodes[rhs].data.t();
                    let grad_lhs = grad_i.matmul(&rhs_data_t)?;
                    let grad_rhs = lhs_data_t.matmul(&grad_i)?;
                    accumulate_grad(&mut nodes, lhs, &grad_lhs)?;
                    accumulate_grad(&mut nodes, rhs, &grad_rhs)?;
                }
                Op::ScalarMul(lhs, scalar) => {
                    let grad_lhs = grad_i.scalar_mul(scalar)?;
                    accumulate_grad(&mut nodes, lhs, &grad_lhs)?;
                }
                Op::Transpose(lhs) => {
                    let grad_lhs = grad_i.t();
                    accumulate_grad(&mut nodes, lhs, &grad_lhs)?;
                }
                Op::Reshape(lhs) => {
                    let orig_shape = nodes[lhs].data.shape().to_vec();
                    let grad_lhs = grad_i.reshape(&orig_shape)?;
                    accumulate_grad(&mut nodes, lhs, &grad_lhs)?;
                }
                Op::Softmax(lhs) => {
                    // Using diagonal approximation for structural pipeline
                    let grad_lhs = nodes[lhs].data.softmax_backward(&grad_i)?;
                    accumulate_grad(&mut nodes, lhs, &grad_lhs)?;
                }
                Op::Relu(lhs) => {
                    let grad_lhs = nodes[lhs].data.relu_backward(&grad_i)?;
                    accumulate_grad(&mut nodes, lhs, &grad_lhs)?;
                }
                Op::Exp(lhs) => {
                    // d(e^x) = e^x * dx. The forward output is e^x.
                    let e_x = &nodes[i].data;
                    let grad_lhs = (e_x * &grad_i)?;
                    accumulate_grad(&mut nodes, lhs, &grad_lhs)?;
                }
                Op::Ln(lhs) => {
                    // d(ln x) = 1/x * dx
                    let ones = Tensor::ones(nodes[lhs].data.shape(), nodes[lhs].data.dtype());
                    let inv_x = (&ones / &nodes[lhs].data)?;
                    let grad_lhs = (&inv_x * &grad_i)?;
                    accumulate_grad(&mut nodes, lhs, &grad_lhs)?;
                }
                Op::Div(lhs, rhs) => {
                    // d(A/B) / dA = 1/B
                    // d(A/B) / dB = -A/B^2
                    let ones = Tensor::ones(nodes[lhs].data.shape(), nodes[lhs].data.dtype());
                    let inv_rhs = (&ones / &nodes[rhs].data)?;
                    let grad_lhs = (&grad_i * &inv_rhs)?;
                    
                    let rhs_sq = (&nodes[rhs].data * &nodes[rhs].data)?;
                    let neg_lhs = nodes[lhs].data.scalar_mul(-1.0)?;
                    let min_a_b_sq = (&neg_lhs / &rhs_sq)?;
                    let grad_rhs = (&grad_i * &min_a_b_sq)?;
                    
                    accumulate_grad(&mut nodes, lhs, &grad_lhs)?;
                    accumulate_grad(&mut nodes, rhs, &grad_rhs)?;
                }
                Op::Leaf => {}
            }
        }
        Ok(())
    }
}

fn accumulate_grad(nodes: &mut Vec<Node>, id: NodeId, grad: &Tensor) -> oximl_core::TensorResult<()> {
    if !nodes[id].requires_grad {
        return Ok(());
    }
    if let Some(existing) = &nodes[id].grad {
        nodes[id].grad = Some((existing + grad)?);
    } else {
        nodes[id].grad = Some(grad.clone());
    }
    Ok(())
}
