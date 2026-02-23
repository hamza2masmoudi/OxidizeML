use oxidize_ml_core::Tensor;
use std::cell::RefCell;

/// Unique identifier for a node in the computation graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

/// The operation that produced a node.
#[derive(Debug, Clone)]
pub enum Op {
    /// Leaf node (parameter or input).
    Leaf,
    /// Element-wise addition.
    Add(NodeId, NodeId),
    /// Element-wise subtraction.
    Sub(NodeId, NodeId),
    /// Element-wise multiplication.
    Mul(NodeId, NodeId),
    /// Element-wise division.
    Div(NodeId, NodeId),
    /// Matrix multiplication.
    MatMul(NodeId, NodeId),
    /// Negation.
    Neg(NodeId),
    /// Exponential.
    Exp(NodeId),
    /// Natural logarithm.
    Ln(NodeId),
    /// Power (element-wise by scalar).
    Pow(NodeId, f64),
    /// ReLU.
    Relu(NodeId),
    /// Sigmoid.
    Sigmoid(NodeId),
    /// Tanh.
    Tanh(NodeId),
    /// Sum all elements to scalar.
    SumAll(NodeId),
    /// Mean all elements.
    MeanAll(NodeId),
    /// Transpose.
    Transpose(NodeId),
    /// Multiply by scalar.
    MulScalar(NodeId, f64),
    /// Add scalar.
    AddScalar(NodeId, f64),
}

/// A node in the computation graph.
#[derive(Debug, Clone)]
pub struct Node {
    pub id: NodeId,
    pub op: Op,
    pub shape: Vec<usize>,
    pub value: Tensor<f64>,
    pub requires_grad: bool,
}

/// The computation graph — arena of nodes.
#[derive(Debug)]
pub struct Graph {
    pub nodes: Vec<Node>,
}

impl Graph {
    pub fn new() -> Self {
        Graph { nodes: Vec::new() }
    }

    /// Add a node and return its ID.
    pub fn add_node(&mut self, op: Op, value: Tensor<f64>, requires_grad: bool) -> NodeId {
        let id = NodeId(self.nodes.len());
        let shape = value.shape_vec();
        self.nodes.push(Node {
            id,
            op,
            shape,
            value,
            requires_grad,
        });
        id
    }

    pub fn get(&self, id: NodeId) -> &Node {
        &self.nodes[id.0]
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

// Thread-local graph for convenient usage
thread_local! {
    static CURRENT_GRAPH: RefCell<Graph> = RefCell::new(Graph::new());
}

/// Execute a closure with the current thread-local graph.
pub fn with_graph<F, R>(f: F) -> R
where
    F: FnOnce(&mut Graph) -> R,
{
    CURRENT_GRAPH.with(|g| f(&mut g.borrow_mut()))
}

/// Reset the thread-local graph.
pub fn reset_graph() {
    CURRENT_GRAPH.with(|g| {
        *g.borrow_mut() = Graph::new();
    });
}
