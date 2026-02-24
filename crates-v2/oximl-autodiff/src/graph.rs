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
}

/// A node in the thread-safe computation graph.
#[derive(Debug, Clone)]
pub struct Node {
    pub id: NodeId,
    pub op: Op,
    pub data: Tensor,
    pub requires_grad: bool,
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
}
