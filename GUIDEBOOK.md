# OxidizeML: Zero to Hero 🦀
## A Complete Guide to Building a Machine Learning Library in Rust

> *From absolute Rust beginner to understanding every line of a production ML library.*

---

# Table of Contents

1. [Why Rust for Machine Learning?](#chapter-1-why-rust-for-machine-learning)
2. [Rust Fundamentals You Need](#chapter-2-rust-fundamentals-you-need)
3. [Project Architecture — The Big Picture](#chapter-3-project-architecture--the-big-picture)
4. [The Tensor Engine (`oxidize-ml-core`)](#chapter-4-the-tensor-engine)
5. [Linear Algebra (`oxidize-ml-linalg`)](#chapter-5-linear-algebra)
6. [Automatic Differentiation (`oxidize-ml-autodiff`)](#chapter-6-automatic-differentiation)
7. [Data Preprocessing (`oxidize-ml-preprocessing`)](#chapter-7-data-preprocessing)
8. [Evaluation Metrics (`oxidize-ml-metrics`)](#chapter-8-evaluation-metrics)
9. [Linear Models (`oxidize-ml-linear`)](#chapter-9-linear-models)
10. [Tree-Based Models (`oxidize-ml-tree`)](#chapter-10-tree-based-models)
11. [Clustering (`oxidize-ml-cluster`)](#chapter-11-clustering)
12. [K-Nearest Neighbors (`oxidize-ml-neighbors`)](#chapter-12-k-nearest-neighbors)
13. [Support Vector Machines (`oxidize-ml-svm`)](#chapter-13-support-vector-machines)
14. [Naive Bayes (`oxidize-ml-naive-bayes`)](#chapter-14-naive-bayes)
15. [Neural Networks (`oxidize-ml-nn`)](#chapter-15-neural-networks)
16. [Optimizers & Loss Functions (`oxidize-ml-optim`, `oxidize-ml-loss`)](#chapter-16-optimizers--loss-functions)
17. [Data Loading & I/O (`oxidize-ml-data`, `oxidize-ml-io`)](#chapter-17-data-loading--io)
18. [Built-in Datasets & Pipelines (`oxidize-ml-datasets`, `oxidize-ml-pipeline`)](#chapter-18-built-in-datasets--pipelines)
19. [The Umbrella Crate (`oxidize-ml`)](#chapter-19-the-umbrella-crate)
20. [Design Decisions & Trade-offs](#chapter-20-design-decisions--trade-offs)
21. [What's Next — Extending OxidizeML](#chapter-21-whats-next)

---

# Chapter 1: Why Rust for Machine Learning?

## 1.1 The Problem with Existing ML Libraries

Most machine learning libraries are written in Python (scikit-learn, PyTorch, TensorFlow). Python is wonderful for prototyping, but it has fundamental limitations:

- **Speed**: Python is interpreted and slow. The "fast" parts of NumPy/PyTorch are actually written in C/C++/Fortran underneath.
- **Memory Safety**: C/C++ extensions can segfault, leak memory, or have buffer overflows.
- **Concurrency**: Python's Global Interpreter Lock (GIL) prevents true multi-threaded execution.
- **Deployment**: Deploying Python ML models requires packaging the entire Python runtime.

## 1.2 Why Rust Is the Answer

Rust gives us the best of both worlds:

| Property | Python | C++ | **Rust** |
|----------|--------|-----|----------|
| Speed | Slow | Fast | **Fast** |
| Memory Safety | GC | Manual (unsafe) | **Guaranteed at compile time** |
| Concurrency | GIL-limited | Data races possible | **Fearless concurrency** |
| Deployment | Heavy runtime | Native binary | **Native binary** |
| Developer Experience | Excellent | Painful | **Good (with learning curve)** |

Rust's **ownership system** ensures that memory bugs are caught at compile time, not at runtime. This means no segfaults, no data races, no use-after-free — ever. For a numerical library that manipulates large arrays of floating-point data, this is invaluable.

## 1.3 What We're Building

**OxidizeML** is a complete, production-grade machine learning library written in 100% pure Rust. No C dependencies. No Python. Just Rust.

It includes:
- A **tensor engine** (like NumPy arrays)
- **Linear algebra** decompositions (LU, QR, Cholesky)
- **Automatic differentiation** (like PyTorch autograd)
- **Classical ML algorithms** (linear regression, decision trees, KNN, SVM, k-means, etc.)
- **Neural network layers** (dense, activations, sequential models)
- **Optimizers** (SGD, Adam)
- **Data utilities** (CSV I/O, datasets, pipelines)

All organized as a **Cargo workspace** with 19 interconnected crates.

---

# Chapter 2: Rust Fundamentals You Need

## 2.1 Ownership & Borrowing — The Core Idea

Rust's most important concept is **ownership**. Every value in Rust has exactly one owner. When the owner goes out of scope, the value is dropped (freed).

```rust
fn main() {
    let v = vec![1, 2, 3];  // v owns the vector
    let w = v;               // ownership MOVES to w
    // println!("{:?}", v);  // ERROR! v no longer owns the data
    println!("{:?}", w);     // OK — w is the owner now
}
```

Instead of moving, you can **borrow** (take a reference):

```rust
fn print_vec(data: &Vec<f64>) {  // & means "borrow, don't take ownership"
    println!("{:?}", data);
}

fn main() {
    let v = vec![1.0, 2.0, 3.0];
    print_vec(&v);    // lend v to the function
    println!("{:?}", v); // v is still valid!
}
```

**Why this matters for ML**: When we pass a tensor to a function (like `model.fit(&x, &y)`), we use `&` (borrowing) so the caller keeps ownership of their data. This prevents accidental data corruption and makes the API safe.

## 2.2 Generics and Traits

A **trait** in Rust is like an interface in Java or a protocol in Swift. It defines a set of methods that a type must implement.

```rust
// We define a trait called "Float" for numeric types
pub trait Float: Copy + Clone + PartialOrd + ... {
    const ZERO: Self;
    const ONE: Self;
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    fn from_f64(val: f64) -> Self;
    // ... more math operations
}
```

Then we implement it for concrete types:

```rust
impl Float for f32 {
    const ZERO: f32 = 0.0;
    const ONE: f32 = 1.0;
    fn sqrt(self) -> f32 { self.sqrt() }
    fn exp(self) -> f32 { self.exp() }
    fn from_f64(val: f64) -> f32 { val as f32 }
}

impl Float for f64 {
    const ZERO: f64 = 0.0;
    const ONE: f64 = 1.0;
    fn sqrt(self) -> f64 { self.sqrt() }
    fn exp(self) -> f64 { self.exp() }
    fn from_f64(val: f64) -> f64 { val }
}
```

**Generics** let us write code that works for *any* type implementing a trait:

```rust
// This Tensor works with f32 OR f64 — we don't have to write two versions!
pub struct Tensor<T: Float> {
    data: Vec<T>,
    shape: Shape,
}
```

**Why this matters**: Our entire library is generic over `T: Float`, meaning every algorithm works with both `f32` (32-bit float, faster, less precise) and `f64` (64-bit float, standard precision). One codebase, two precisions.

## 2.3 Error Handling — Result and the `?` Operator

Rust doesn't use exceptions. Instead, functions that can fail return `Result<T, E>`:

```rust
// This function might fail (e.g., shape mismatch)
pub fn matmul(&self, other: &Tensor<T>) -> TensorResult<Tensor<T>> {
    if self.shape.dim(1)? != other.shape.dim(0)? {
        return Err(TensorError::DimensionMismatch("...".into()));
    }
    // ... compute the product ...
    Ok(result)
}
```

The `?` operator propagates errors upward:

```rust
let xtx = xt.matmul(&x_aug)?;   // If matmul fails, return the error immediately
let xtx_inv = inv(&xtx)?;        // Same here
let w = xtx_inv.matmul(&xty)?;   // And here
```

**Why this matters**: Every tensor operation that can fail (wrong shapes, singular matrix, index out of bounds) returns a `Result`. This forces us to handle errors explicitly — no silent NaN propagation, no cryptic segfaults.

## 2.4 The Cargo Workspace

A **Cargo workspace** is a collection of related Rust packages (called "crates") that share a single `Cargo.lock` file and output directory.

```
ML_in rust/
├── Cargo.toml          ← root workspace definition
├── crates/
│   ├── oxidize-ml-core/       ← the tensor engine
│   │   ├── Cargo.toml        ← this crate's dependencies
│   │   └── src/
│   │       ├── lib.rs         ← entry point
│   │       ├── tensor.rs      ← Tensor struct & operations
│   │       ├── shape.rs       ← Shape type
│   │       ├── dtype.rs       ← Float trait
│   │       └── error.rs       ← Error types
│   ├── oxidize-ml-linalg/     ← linear algebra
│   ├── oxidize-ml-autodiff/   ← automatic differentiation
│   ├── ...                   ← 16 more crates
│   └── oxidize-ml/            ← umbrella crate (re-exports everything)
```

The root `Cargo.toml` lists all crates:

```toml
[workspace]
members = [
    "crates/oxidize-ml-core",
    "crates/oxidize-ml-linalg",
    "crates/oxidize-ml-autodiff",
    # ... all 19 crates
]

[workspace.dependencies]
rand = "0.8"
serde = { version = "1", features = ["derive"] }
thiserror = "1"
# ... shared dependencies
```

**Why a workspace?** Each crate compiles independently, has its own API boundary, and can be used standalone. A user who only needs tensors can depend on `oxidize-ml-core` without pulling in neural network code. This is modular design.

## 2.5 Key Rust Patterns Used in OxidizeML

### The Builder Pattern
```rust
let model = Sequential::new()
    .add(Box::new(Linear::new(784, 128)))
    .add(Box::new(ReLULayer::new()))
    .add(Box::new(Linear::new(128, 10)));
```

### The `impl` Block
Rust groups methods into `impl` blocks on structs:
```rust
impl<T: Float> Tensor<T> {
    pub fn zeros(shape: Vec<usize>) -> Self { ... }
    pub fn add(&self, other: &Tensor<T>) -> TensorResult<Tensor<T>> { ... }
}
```

### Derive Macros
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor<T: Float> { ... }
```
`derive` automatically generates implementations — `Debug` for printing, `Clone` for copying, `Serialize`/`Deserialize` for JSON support.

---

# Chapter 3: Project Architecture — The Big Picture

## 3.1 The Dependency Graph

OxidizeML's 19 crates form a layered architecture. Each layer depends only on layers below it:

```
┌─────────────────────────────────────────────────────────┐
│                    oxidize-ml (umbrella)                  │
├─────────────────────────────────────────────────────────┤
│  pipeline │ datasets │ io │ data │ loss │ optim │ nn    │  ← Utilities
├─────────────────────────────────────────────────────────┤
│  linear │ tree │ cluster │ neighbors │ svm │ naive-bayes│  ← ML Algorithms
├─────────────────────────────────────────────────────────┤
│           preprocessing  │  metrics                     │  ← Data Tools
├─────────────────────────────────────────────────────────┤
│                  autodiff  │  linalg                    │  ← Math Engine
├─────────────────────────────────────────────────────────┤
│                       core (Tensor)                     │  ← Foundation
└─────────────────────────────────────────────────────────┘
```

**Key principle**: Nothing in `core` knows about `linear` or `tree`. But `linear` and `tree` both know about `core`. This is **dependency inversion** — high-level modules depend on low-level abstractions, not the other way around.

## 3.2 The Data Flow

A typical ML workflow in OxidizeML looks like:

```
Raw Data  →  Preprocessing  →  Model.fit()  →  Model.predict()  →  Metrics
  (CSV)       (Scaling)        (Training)       (Inference)        (Accuracy)
```

In code:
```rust
// 1. Load data
let (data, headers) = oxidize_ml::io::read_csv("data.csv")?;

// 2. Split features and labels
let x = /* features */;
let y = /* labels */;

// 3. Preprocess
let mut scaler = StandardScaler::new();
let x_scaled = scaler.fit_transform(&x)?;

// 4. Train
let mut model = RandomForestClassifier::new(100, None, None, None);
model.fit(&x_scaled, &y)?;

// 5. Predict & Evaluate
let predictions = model.predict(&x_scaled)?;
let acc = accuracy(&y, &predictions);
```

## 3.3 API Design Philosophy

Every ML model in OxidizeML follows the **scikit-learn convention**:

| Method | Purpose |
|--------|---------|
| `Model::new(...)` | Create with hyperparameters |
| `model.fit(&x, &y)` | Train on data |
| `model.predict(&x)` | Generate predictions |

Every preprocessor follows:

| Method | Purpose |
|--------|---------|
| `Scaler::new()` | Create |
| `scaler.fit(&x)` | Learn statistics from training data |
| `scaler.transform(&x)` | Apply transformation |
| `scaler.fit_transform(&x)` | Fit + transform in one step |

This consistency means that once you learn one model, you know how to use them all.

---

# Chapter 4: The Tensor Engine

## 4.1 What Is a Tensor?

A **tensor** is a multi-dimensional array of numbers. It's the fundamental data structure of all ML:

| Rank | Name | Example | Shape |
|------|------|---------|-------|
| 0 | Scalar | `3.14` | `()` |
| 1 | Vector | `[1, 2, 3]` | `(3,)` |
| 2 | Matrix | `[[1,2],[3,4]]` | `(2, 2)` |
| 3 | 3D Tensor | Batch of images | `(32, 28, 28)` |

In OxidizeML, our `Tensor<T>` stores data as a **flat vector** in row-major (C) order:

```rust
pub struct Tensor<T: Float> {
    data: Vec<T>,       // flat storage: all elements in one contiguous block
    shape: Shape,       // logical dimensions: [rows, cols, ...]
}
```

For a 2×3 matrix `[[1,2,3],[4,5,6]]`:
- `data = [1, 2, 3, 4, 5, 6]` (flattened row by row)
- `shape = Shape { dims: [2, 3] }`

## 4.2 The Shape Type

`Shape` knows how to navigate the flat array:

```rust
pub struct Shape {
    dims: Vec<usize>,  // e.g., [2, 3, 4] for a 2×3×4 tensor
}

impl Shape {
    // How many elements total? Product of all dims.
    pub fn numel(&self) -> usize {
        self.dims.iter().product()  // 2*3*4 = 24
    }

    // Strides: how many elements to skip for each dimension
    pub fn strides(&self) -> Vec<usize> {
        // For shape [2, 3, 4]:
        // stride[2] = 1    (moving one step in last dim = 1 element)
        // stride[1] = 4    (moving one step in middle dim = 4 elements)
        // stride[0] = 12   (moving one step in first dim = 12 elements)
        // Result: [12, 4, 1]
    }
}
```

**Strides** are the key insight. To access element `[i, j, k]` of a 3D tensor:
```
flat_index = i * strides[0] + j * strides[1] + k * strides[2]
```

This is how we convert multi-dimensional indices to flat array positions in O(1) time.

## 4.3 Broadcasting

Broadcasting is NumPy's elegant rule for combining arrays of different shapes. OxidizeML implements the same rules:

```
Shape [3, 1] + Shape [1, 4] → Shape [3, 4]
Shape [5, 3, 1] + Shape [4] → Shape [5, 3, 4]
```

**Rules**:
1. Align shapes from the right
2. Dimensions must be equal OR one of them must be 1
3. Dimension of 1 gets "stretched" to match the other

The implementation walks through each output element, computing the source index in each input tensor by checking if the dimension is 1 (broadcast) or not:

```rust
fn broadcast_binary_op<F: Fn(T, T) -> T>(
    &self, other: &Tensor<T>, op: F,
) -> TensorResult<Tensor<T>> {
    let out_shape = Shape::broadcast_shape(&self.shape, &other.shape)?;
    // For each output element:
    //   1. Convert flat index to multi-dim index
    //   2. Map to input indices (clamping broadcast dims to 0)
    //   3. Apply op
}
```

## 4.4 Matrix Multiplication

Matrix multiplication (`matmul`) is the single most important operation in ML. For matrices A (m×k) and B (k×n), the result C (m×n) has:

```
C[i][j] = Σ A[i][p] * B[p][j]  for p = 0..k
```

Our implementation uses the classic triple-loop algorithm:

```rust
pub fn matmul(&self, other: &Tensor<T>) -> TensorResult<Tensor<T>> {
    let m = self.shape.dim(0)?;   // rows of A
    let k = self.shape.dim(1)?;   // cols of A = rows of B
    let n = other.shape.dim(1)?;  // cols of B

    let mut result = vec![T::ZERO; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = T::ZERO;
            for p in 0..k {
                sum = sum + self.data[i * k + p] * other.data[p * n + j];
            }
            result[i * n + j] = sum;
        }
    }
    Tensor::new(result, vec![m, n])
}
```

> **Performance note**: This O(n³) implementation is correct but not optimized. Production systems use BLAS (Strassen, tiling, SIMD). OxidizeML could optionally use `openblas` or `blis` for 10-100x speedup on large matrices.

## 4.5 Reduction Operations

Reductions collapse a dimension by applying an aggregation:

```rust
// Sum along axis 0 of a 3×4 matrix → 1×4 vector
// Sum along axis 1 of a 3×4 matrix → 3×1 vector

pub fn sum_axis(&self, axis: usize) -> TensorResult<Tensor<T>> {
    let outer = product_of_dims_before_axis;
    let axis_size = dims[axis];
    let inner = product_of_dims_after_axis;

    for o in 0..outer {
        for a in 0..axis_size {      // iterate along the axis being reduced
            for i in 0..inner {
                result[o * inner + i] += self.data[o * axis_size * inner + a * inner + i];
            }
        }
    }
}
```

This pattern (outer × axis × inner) generalizes to any number of dimensions. The same structure is used for `mean_axis`, `var_axis`, and `argmax_axis`.

## 4.6 Activation Functions

Activations are element-wise non-linearities. They're implemented using the `apply` method:

```rust
pub fn apply<F: Fn(T) -> T>(&self, f: F) -> Tensor<T> {
    Tensor {
        data: self.data.iter().map(|&x| f(x)).collect(),
        shape: self.shape.clone(),
    }
}

pub fn relu(&self) -> Tensor<T> { self.apply(|x| x.max(T::ZERO)) }
pub fn sigmoid(&self) -> Tensor<T> { self.apply(|x| T::ONE / (T::ONE + (-x).exp())) }
```

---

# Chapter 5: Linear Algebra

## 5.1 Why We Need It

Most ML algorithms reduce to linear algebra at their core:
- **Linear regression**: solving `(XᵀX)⁻¹Xᵀy`
- **PCA**: eigendecomposition of the covariance matrix
- **Neural networks**: chains of matrix multiplications

## 5.2 LU Decomposition

**LU decomposition** factors a matrix A into:
```
P·A = L·U
```
where L is lower-triangular, U is upper-triangular, and P is a permutation matrix (for numerical stability via partial pivoting).

**Algorithm** (Gaussian elimination with pivoting):
1. For each column k:
   - Find the row with the largest value below the diagonal (pivot)
   - Swap that row with row k
   - Eliminate all entries below the diagonal by subtracting multiples of row k

**Used for**: solving Ax=b, computing determinants, matrix inverse.

## 5.3 QR Decomposition

**QR decomposition** factors A = Q·R where Q is orthogonal (QᵀQ = I) and R is upper-triangular.

We use **Householder reflections**: for each column, we find a reflection that zeros out everything below the diagonal. This is numerically more stable than Gram-Schmidt.

**Used for**: least-squares problems, eigenvalue algorithms.

## 5.4 Cholesky Decomposition

For symmetric positive-definite (SPD) matrices: A = L·Lᵀ

This is 2x faster than LU and numerically more stable for SPD matrices. Used in Gaussian processes and regularized regression.

## 5.5 Solving Linear Systems

```rust
// solve(A, b) finds x such that Ax = b
// 1. Decompose: PA = LU
// 2. Forward substitution: Ly = Pb
// 3. Back substitution: Ux = y
```

```rust
// lstsq(A, b) finds x that minimizes ||Ax - b||²
// 1. Decompose: A = QR
// 2. Compute: Qᵀb
// 3. Back substitution: Rx = Qᵀb (first n rows)
```

---

# Chapter 6: Automatic Differentiation

## 6.1 The Core Idea

Automatic differentiation (autodiff) computes **exact** derivatives of any function composed of elementary operations. It's neither symbolic differentiation (like Wolfram Alpha) nor numerical differentiation (finite differences). It's a clever application of the **chain rule**.

For example, to compute the gradient of `f(x) = (x² + 3x) * sin(x)`:
- **Symbolic**: expand, simplify, differentiate → complex expression
- **Numerical**: `(f(x+ε) - f(x-ε)) / 2ε` → approximate, ε-sensitive
- **Automatic**: track every operation, apply chain rule backward → exact, efficient

## 6.2 The Computation Graph

Every operation is recorded in a **directed acyclic graph (DAG)**:

```rust
pub struct Node {
    pub op: Op,                    // What operation produced this node?
    pub inputs: Vec<NodeId>,       // Which nodes were the inputs?
    pub shape: Vec<usize>,         // Shape of the output tensor
    pub requires_grad: bool,       // Do we need gradients for this?
}

pub enum Op {
    Input,           // leaf node (data or parameter)
    Add, Sub, Mul, Div,
    MatMul,
    Exp, Ln, Pow(f64),
    ReLU, Sigmoid, Tanh,
    Sum, Mean, Transpose,
    // ... more operations
}
```

When you write:
```rust
let x = Variable::param(tensor_x);  // Recorded as Input node
let y = Variable::param(tensor_y);
let z = x.mul(&y);                  // Recorded as Mul node with inputs [x, y]
let loss = z.sum().mean();           // Two more nodes
```

This builds a graph: `x → Mul → Sum → Mean → loss`

## 6.3 Reverse-Mode AD (Backpropagation)

To compute gradients, we traverse the graph **backward** (from output to inputs):

```rust
pub fn backward(loss_node: NodeId) -> HashMap<NodeId, Tensor<f64>> {
    // 1. Start with ∂loss/∂loss = 1
    // 2. Topological sort (reverse order)
    // 3. For each node, compute ∂loss/∂input using the chain rule

    for node in reverse_topological_order {
        match node.op {
            Op::Add => {
                // ∂/∂a (a + b) = 1, ∂/∂b (a + b) = 1
                grad[input_a] += grad[node] * 1;
                grad[input_b] += grad[node] * 1;
            }
            Op::Mul => {
                // ∂/∂a (a * b) = b, ∂/∂b (a * b) = a
                grad[input_a] += grad[node] * value[input_b];
                grad[input_b] += grad[node] * value[input_a];
            }
            Op::ReLU => {
                // ∂/∂x relu(x) = 1 if x > 0, else 0
                grad[input] += grad[node] * (value[input] > 0 ? 1 : 0);
            }
            // ... similar rules for all operations
        }
    }
}
```

**Why reverse mode?** For a function f: ℝⁿ → ℝ (like a loss function), reverse-mode computes all n partial derivatives in **one backward pass**. Forward-mode would need n passes. Since ML losses are scalar-valued, reverse-mode is O(n) vs O(n²).

## 6.4 The Variable Type

`Variable` wraps a `Tensor<f64>` and records operations into the computation graph:

```rust
pub struct Variable {
    pub data: Tensor<f64>,
    pub node_id: Option<NodeId>,
    pub requires_grad: bool,
}
```

Mathematical operations on `Variable` automatically record nodes:
```rust
impl Variable {
    pub fn add(&self, other: &Variable) -> Variable {
        let result_data = self.data.add(&other.data).expect("add");
        let node_id = with_graph(|g| {
            g.add_node(Op::Add, vec![self.node_id, other.node_id], ...)
        });
        Variable { data: result_data, node_id: Some(node_id), requires_grad: true }
    }
}
```

This is the same pattern PyTorch uses for `torch.Tensor` with `requires_grad=True`.

---

# Chapter 7: Data Preprocessing

## 7.1 StandardScaler

Standardization transforms features to have mean 0 and standard deviation 1:

```
x_scaled = (x - μ) / σ
```

This is critical because many ML algorithms (gradient descent, SVM, KNN) are sensitive to feature scale. A feature ranging from 0-1000 would dominate a feature ranging from 0-1.

```rust
pub struct StandardScaler<T: Float> {
    pub mean: Option<Tensor<T>>,
    pub std: Option<Tensor<T>>,
}

impl StandardScaler {
    pub fn fit(&mut self, x: &Tensor<T>) {
        self.mean = Some(x.mean_axis(0));  // per-feature mean
        self.std = Some(x.var_axis(0).sqrt());  // per-feature std
    }

    pub fn transform(&self, x: &Tensor<T>) -> Tensor<T> {
        // (x - mean) / std, broadcast across all samples
    }
}
```

## 7.2 Train/Test Split

To evaluate a model honestly, we split data into training and testing sets:

```rust
pub fn train_test_split<T: Float>(
    x: &Tensor<T>, y: &Tensor<T>,
    test_ratio: f64,          // e.g., 0.2 for 80/20 split
    seed: Option<u64>,        // for reproducibility
) -> (Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>)
// Returns: (x_train, x_test, y_train, y_test)
```

The implementation shuffles indices first (using a seeded RNG for reproducibility), then splits.

## 7.3 Label Encoding

Converts string labels to integers:
```rust
// ["cat", "dog", "cat", "fish"] → [0, 1, 0, 2]
```

And one-hot encoding converts integers to binary vectors:
```rust
// [0, 1, 2] with 3 classes → [[1,0,0], [0,1,0], [0,0,1]]
```

---

# Chapter 8: Evaluation Metrics

## 8.1 Classification Metrics

**Accuracy**: fraction of correct predictions.
```rust
pub fn accuracy<T: Float>(y_true: &Tensor<T>, y_pred: &Tensor<T>) -> f64 {
    let correct = y_true.data().iter().zip(y_pred.data().iter())
        .filter(|(&t, &p)| (t - p).abs() < T::HALF)
        .count();
    correct as f64 / y_true.numel() as f64
}
```

**Precision, Recall, F1**: per-class and macro-averaged metrics computed from the confusion matrix.

## 8.2 Regression Metrics

**MSE** (Mean Squared Error): average of squared differences.
**RMSE**: square root of MSE (same units as target).
**R² score**: 1 - (SS_res / SS_tot), where 1.0 = perfect, 0.0 = baseline.

---

# Chapter 9: Linear Models

## 9.1 Linear Regression (OLS)

The classic: find weights **w** that minimize `||Xw - y||²`.

**Normal equation solution**: `w = (XᵀX)⁻¹Xᵀy`

```rust
pub fn fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> TensorResult<()> {
    // 1. Augment X with column of 1s for intercept
    let x_aug = concat([ones_column, x], axis=1);

    // 2. Compute normal equation
    let xt = x_aug.t()?;              // Xᵀ
    let xtx = xt.matmul(&x_aug)?;     // XᵀX
    let xty = xt.matmul(&y_reshaped)?; // Xᵀy
    let xtx_inv = inv(&xtx)?;         // (XᵀX)⁻¹
    let w = xtx_inv.matmul(&xty)?;    // w = (XᵀX)⁻¹Xᵀy

    // 3. Extract bias (first element) and weights (rest)
    self.bias = w[0];
    self.weights = w[1..];
    Ok(())
}
```

## 9.2 Ridge Regression (L2)

Adds L2 penalty to prevent overfitting: `w = (XᵀX + αI)⁻¹Xᵀy`

The regularization term αI makes XᵀX always invertible (even for collinear features) and shrinks weights toward zero.

## 9.3 Lasso Regression (L1)

Uses L1 penalty which produces **sparse** solutions (some weights become exactly zero). Solved via **coordinate descent** with soft thresholding.

## 9.4 Logistic Regression

For binary classification. Uses the sigmoid function and gradient descent:

```
P(y=1|x) = sigmoid(xᵀw + b)
Loss = -[y·log(p) + (1-y)·log(1-p)]   (binary cross-entropy)
Update: w -= lr * ∇Loss
```

---

# Chapter 10: Tree-Based Models

## 10.1 Decision Trees (CART)

Decision trees recursively partition the feature space by finding the best feature and threshold to split on.

**Classification**: uses **Gini impurity** as the splitting criterion:
```
Gini(S) = 1 - Σ pᵢ²
```
where pᵢ is the proportion of class i in set S. Pure nodes have Gini = 0.

**Regression**: uses **Mean Squared Error** — the split that minimizes the variance of each child.

```rust
pub struct TreeNode<T: Float> {
    feature_idx: Option<usize>,    // which feature to split on
    threshold: Option<T>,          // split value
    left: Option<Box<TreeNode<T>>>,
    right: Option<Box<TreeNode<T>>>,
    value: Option<T>,              // leaf prediction
}
```

## 10.2 Random Forest

An **ensemble** of decision trees, each trained on a random subset of data (bagging) with a random subset of features:

```rust
pub fn fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) {
    for i in 0..n_estimators {
        // 1. Bootstrap sample (random sampling with replacement)
        // 2. Train a decision tree on the bootstrap sample
        //    (with random feature subsampling at each split)
        // 3. Store the tree
    }
}

pub fn predict(&self, x: &Tensor<T>) -> Tensor<T> {
    // Majority vote (classification) or average (regression)
    // across all trees
}
```

---

# Chapter 11: Clustering

## 11.1 K-Means

Partitions data into k clusters by iteratively:
1. **Assign** each point to the nearest centroid
2. **Update** centroids as the mean of assigned points
3. Repeat until convergence

**K-means++ initialization**: instead of random centroids, choose initial centroids that are far apart. This dramatically improves convergence.

## 11.2 DBSCAN

Density-based clustering: finds clusters of arbitrary shape. Core points have ≥ `min_samples` neighbors within distance `eps`. Border points are within `eps` of a core point. Noise points belong to no cluster.

---

# Chapter 12: K-Nearest Neighbors

The simplest ML algorithm: to predict for a new point, find the k training points closest to it, then:
- **Classification**: majority vote among the k neighbors
- **Regression**: average of the k neighbors' values

Supports **Euclidean** and **Manhattan** distance metrics.

---

# Chapter 13: Support Vector Machines

SVM finds the hyperplane that maximizes the **margin** between classes. Our implementation uses the **Simplified SMO** (Sequential Minimal Optimization) algorithm to solve the dual problem.

Supports kernels:
- **Linear**: `K(x,y) = xᵀy`
- **RBF**: `K(x,y) = exp(-γ||x-y||²)`
- **Polynomial**: `K(x,y) = (xᵀy + c)ᵈ`

---

# Chapter 14: Naive Bayes

Applies Bayes' theorem with the "naive" assumption that features are independent:

```
P(class|features) ∝ P(class) × Π P(feature_i|class)
```

For **Gaussian Naive Bayes**, each feature is modeled as a normal distribution per class. Classification uses log-probabilities for numerical stability.

---

# Chapter 15: Neural Networks

## 15.1 The Layer Trait

```rust
pub trait Layer {
    fn forward(&self, input: &Variable) -> Variable;
    fn parameters(&self) -> Vec<Variable>;
}
```

Every layer implements this interface, making them composable.

## 15.2 Linear (Dense) Layer

`y = xW + b` where W has shape [in_features, out_features].

Initialized with **Xavier uniform** initialization: weights sampled from `Uniform(-√(6/(in+out)), √(6/(in+out)))`, which prevents vanishing/exploding gradients.

## 15.3 Sequential Model

Chains layers in order:

```rust
let model = Sequential::new()
    .add(Box::new(Linear::new(784, 128)))   // 784 → 128
    .add(Box::new(ReLULayer::new()))         // non-linearity
    .add(Box::new(Linear::new(128, 10)));    // 128 → 10

let output = model.forward(&input);
let params = model.parameters();  // collects W, b from all Linear layers
```

---

# Chapter 16: Optimizers & Loss Functions

## 16.1 SGD with Momentum

```
v = β·v - lr·∇L        (momentum accumulator)
θ = θ + v              (parameter update)
```

Momentum helps escape local minima and smooths noisy gradients.

## 16.2 Adam

The most popular optimizer, combining momentum and RMSProp:

```
m = β₁·m + (1-β₁)·∇L           (first moment estimate)
v = β₂·v + (1-β₂)·(∇L)²        (second moment estimate)
m̂ = m / (1-β₁ᵗ)                 (bias-corrected)
v̂ = v / (1-β₂ᵗ)                 (bias-corrected)
θ = θ - lr · m̂ / (√v̂ + ε)       (update)
```

## 16.3 MSE Loss

`L = mean((prediction - target)²)`

Implemented as autodiff-compatible Variable operations so gradients flow through.

---

# Chapter 17: Data Loading & I/O

## 17.1 Dataset Trait & DataLoader

```rust
pub trait Dataset {
    fn len(&self) -> usize;
    fn get(&self, idx: usize) -> (Tensor<f64>, Tensor<f64>);
}
```

`DataLoader` wraps a Dataset and provides batched iteration with shuffling:
```rust
let loader = DataLoader::new(&dataset, batch_size=32, shuffle=true);
for (x_batch, y_batch) in loader {
    // train on batch
}
```

## 17.2 CSV I/O

Read/write tensors from/to CSV files. Used for loading real-world datasets.

## 17.3 Model Serialization

Save/load model weights as JSON using `serde`:
```rust
let mut weights = ModelWeights::new();
weights.add("layer1_w", &model_weights);
save_model(&weights, "model.json")?;
```

---

# Chapter 18: Built-in Datasets & Pipelines

## 18.1 Built-in Datasets

- **Iris**: 30-sample subset of the classic 4-feature, 3-class dataset
- **make_blobs**: synthetic Gaussian clusters for classification
- **make_regression**: synthetic linear data with noise for regression

## 18.2 Pipeline API

Composable ML pipelines:

```rust
let mut pipeline = Pipeline::new()
    .add_transformer(Box::new(my_scaler))
    .set_estimator(Box::new(my_model));

pipeline.fit(&x_train, &y_train)?;
let predictions = pipeline.predict(&x_test)?;
```

---

# Chapter 19: The Umbrella Crate

`oxidize-ml` re-exports everything under a clean namespace:

```rust
use oxidize_ml::core::Tensor;
use oxidize_ml::linear::LinearRegression;
use oxidize_ml::metrics::accuracy;
use oxidize_ml::datasets::load_iris;
```

---

# Chapter 20: Design Decisions & Trade-offs

## 20.1 Pure Rust vs. FFI to BLAS

We chose **pure Rust** for simplicity and portability. This means:
- ✅ No C/C++ build dependencies
- ✅ Cross-compiles easily (including WASM)
- ✅ All code is memory-safe
- ❌ Matrix multiplication is slower than optimized BLAS

**Future**: optional `openblas` feature flag for production speed.

## 20.2 Generic `Float` Trait vs. Concrete `f64`

We chose generics so users can choose precision. The trade-off:
- ✅ Works with `f32` (faster, less memory) and `f64` (precise)
- ❌ More complex type signatures and trait bounds

## 20.3 Thread-Local Computation Graph

The autodiff uses a thread-local graph (stored in `thread_local!`). This means:
- ✅ No need to pass graph references around
- ✅ Each thread gets its own graph (safe parallelism)
- ❌ Can't share Variables across threads

## 20.4 Arena Allocation for Graph Nodes

Graph nodes are stored in a `Vec<Node>` (arena), and `NodeId` is just a `usize` index. This means:
- ✅ Very fast allocation (just push to Vec)
- ✅ Cache-friendly memory layout
- ❌ Nodes are never individually freed (only when the entire graph is reset)

---

# Chapter 21: What's Next

## Immediate Extensions
1. **Conv2D, RNN/LSTM layers** for deep learning
2. **Learning rate schedulers** (cosine annealing, step decay)
3. **Dropout and BatchNorm** for regularization
4. **Gradient Boosting** (XGBoost-style)
5. **PCA and t-SNE** for dimensionality reduction

## Performance
1. **Optional BLAS backend** via `openblas-sys` or `blis`
2. **SIMD vectorization** using `std::simd`
3. **Rayon parallelism** for data-parallel operations
4. **GPU support** via `wgpu` or `vulkan`

## Ecosystem
1. **Python bindings** via `pyo3` (so Python users can use OxidizeML)
2. **ONNX model import/export**
3. **Benchmarks** against scikit-learn and PyTorch
4. **Documentation** on docs.rs

---

# Appendix A: Complete Crate Summary

| # | Crate | Lines | Purpose |
|---|-------|-------|---------|
| 1 | `oxidize-ml-core` | ~1260 | Tensor engine with broadcasting, matmul, activations |
| 2 | `oxidize-ml-linalg` | ~430 | LU, QR, Cholesky, determinant, inverse, solve |
| 3 | `oxidize-ml-autodiff` | ~700 | Computation graph + reverse-mode AD |
| 4 | `oxidize-ml-preprocessing` | ~350 | Scalers, encoders, train/test split |
| 5 | `oxidize-ml-metrics` | ~290 | Classification & regression metrics |
| 6 | `oxidize-ml-linear` | ~340 | OLS, Ridge, Lasso, Logistic Regression |
| 7 | `oxidize-ml-tree` | ~480 | Decision Trees (CART), Random Forest |
| 8 | `oxidize-ml-cluster` | ~290 | K-Means (k-means++), DBSCAN |
| 9 | `oxidize-ml-neighbors` | ~180 | KNN classifier/regressor |
| 10 | `oxidize-ml-svm` | ~230 | SVC with kernels (SMO) |
| 11 | `oxidize-ml-naive-bayes` | ~140 | Gaussian Naive Bayes |
| 12 | `oxidize-ml-nn` | ~170 | Layer trait, Linear, Sequential |
| 13 | `oxidize-ml-optim` | ~130 | SGD (momentum), Adam |
| 14 | `oxidize-ml-loss` | ~50 | MSE, BCE loss functions |
| 15 | `oxidize-ml-data` | ~100 | Dataset trait, DataLoader |
| 16 | `oxidize-ml-io` | ~110 | CSV I/O, model serialization |
| 17 | `oxidize-ml-datasets` | ~140 | Iris, make_blobs, make_regression |
| 18 | `oxidize-ml-pipeline` | ~90 | Transformer + Estimator pipeline |
| 19 | `oxidize-ml` | ~80 | Umbrella re-export crate |
| | **Total** | **~5,000+** | |

---

# Appendix B: How to Build & Test

```bash
# Clone and enter the project
cd "ML_in rust"

# Build everything
cargo build --workspace

# Run all tests
cargo test --workspace

# Build in release mode (optimized)
cargo build --workspace --release

# Run a specific crate's tests
cargo test -p oxidize-ml-core
cargo test -p oxidize-ml-linalg
```

---

# Appendix C: Glossary

| Term | Definition |
|------|-----------|
| **Tensor** | Multi-dimensional array of numbers |
| **Shape** | The dimensions of a tensor, e.g., [2, 3] for a 2×3 matrix |
| **Stride** | Number of elements to skip in memory for each index step |
| **Broadcasting** | Automatic expansion of tensor dimensions for element-wise ops |
| **Matmul** | Matrix multiplication: C = A × B |
| **Autodiff** | Automatic computation of derivatives via the chain rule |
| **Backpropagation** | Reverse-mode autodiff applied to neural networks |
| **Crate** | A Rust package (library or binary) |
| **Trait** | A set of methods that types can implement (like an interface) |
| **Ownership** | Rust's system where each value has exactly one owner |
| **Borrowing** | Using a reference (`&`) to access data without taking ownership |
| **`Result<T, E>`** | Rust's error handling type: either `Ok(value)` or `Err(error)` |
| **Workspace** | A collection of related Rust crates sharing build artifacts |
| **OLS** | Ordinary Least Squares — minimizing sum of squared residuals |
| **CART** | Classification and Regression Trees |
| **Gini impurity** | Measure of how mixed the classes are at a tree node |
| **Bagging** | Bootstrap Aggregating — training on random subsets |
| **SMO** | Sequential Minimal Optimization — SVM training algorithm |

---

*Built with 🦀 and ❤️ — OxidizeML v0.1.0*
