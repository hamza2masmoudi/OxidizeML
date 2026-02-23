# OxidizeML 🦀

**A production-grade Machine Learning library written in pure Rust.**

No C/C++ dependencies. No garbage collector. Just fast, safe, expressive ML.

## Quick Start

```rust
use oxidize_ml::core::Tensor;
use oxidize_ml::linear::LinearRegression;
use oxidize_ml::metrics::r2_score;

fn main() {
    // Create data
    let x = Tensor::from_vec2d(&[
        vec![1.0, 1.0],
        vec![2.0, 2.0],
        vec![3.0, 3.0],
    ]).unwrap();
    let y = Tensor::from_slice(&[3.0, 5.0, 7.0]);

    // Fit model
    let mut model = LinearRegression::new(true);
    model.fit(&x, &y).unwrap();

    // Predict
    let predictions = model.predict(&x).unwrap();
    let r2 = r2_score(&y, &predictions);
    println!("R² = {:.4}", r2);
}
```

## Features

| Module | What's Inside |
|--------|---------------|
| `core` | N-dimensional Tensor with broadcasting, matmul, activations |
| `linalg` | LU, QR, Cholesky decompositions; solve, lstsq, inverse |
| `autodiff` | Reverse-mode automatic differentiation with computation graph |
| `preprocessing` | StandardScaler, MinMaxScaler, LabelEncoder, train/test split |
| `linear` | Linear Regression, Ridge, Lasso, Logistic Regression |
| `tree` | Decision Trees (CART), Random Forest |
| `cluster` | K-Means (k-means++), DBSCAN |
| `neighbors` | KNN Classifier/Regressor |
| `svm` | SVC with Linear/RBF/Polynomial kernels |
| `naive_bayes` | Gaussian Naive Bayes |
| `metrics` | Accuracy, Precision, Recall, F1, MSE, RMSE, MAE, R² |
| `nn` | Linear layer, ReLU/Sigmoid/Tanh, Sequential model |
| `optim` | SGD (momentum), Adam |
| `loss` | MSE Loss, BCE Loss |
| `data` | Dataset trait, DataLoader with batching |
| `io` | CSV I/O, model save/load |
| `datasets` | Iris, make_blobs, make_regression |
| `pipeline` | Composable Transformer + Estimator chains |

## Architecture

```
oxidize-ml (umbrella)
├── oxidize-ml-core        # Tensor engine
├── oxidize-ml-linalg      # Linear algebra
├── oxidize-ml-autodiff    # Automatic differentiation
├── oxidize-ml-preprocessing
├── oxidize-ml-linear      # Linear models
├── oxidize-ml-tree        # Tree-based models
├── oxidize-ml-cluster     # Clustering
├── oxidize-ml-neighbors   # KNN
├── oxidize-ml-svm         # Support Vector Machines
├── oxidize-ml-naive-bayes # Naive Bayes
├── oxidize-ml-metrics     # Evaluation
├── oxidize-ml-nn          # Neural network layers
├── oxidize-ml-optim       # Optimizers
├── oxidize-ml-loss        # Loss functions
├── oxidize-ml-data        # Data loading
├── oxidize-ml-io          # I/O
├── oxidize-ml-datasets    # Built-in datasets
└── oxidize-ml-pipeline    # Pipeline API
```

## License

MIT
