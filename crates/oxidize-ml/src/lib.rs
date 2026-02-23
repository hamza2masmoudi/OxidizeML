//! # OxidizeML 🦀
//!
//! A production-grade Machine Learning library written in pure Rust.
//!
//! ## Modules
//!
//! - **core** — Tensor engine: N-dimensional arrays with broadcasting, arithmetic, reductions
//! - **linalg** — Linear algebra: LU, QR, Cholesky, SVD, matrix inverse, linear solvers
//! - **autodiff** — Automatic differentiation: computation graph with reverse-mode AD
//! - **preprocessing** — StandardScaler, MinMaxScaler, LabelEncoder, train/test split
//! - **linear** — Linear models: OLS, Ridge, Lasso, ElasticNet, Logistic Regression
//! - **tree** — Tree models: Decision Tree (CART), Random Forest, Gradient Boosting
//! - **cluster** — Clustering: K-Means (with k-means++), DBSCAN
//! - **neighbors** — KNN: classifier and regressor with Euclidean/Manhattan distance
//! - **svm** — Support Vector Machines: SVC/SVR with kernel support
//! - **naive_bayes** — Naive Bayes: Gaussian NB
//! - **metrics** — Evaluation: accuracy, precision, recall, F1, MSE, RMSE, R²
//! - **nn** — Neural networks: Linear layer, ReLU/Sigmoid/Tanh, Sequential
//! - **optim** — Optimizers: SGD (momentum), Adam
//! - **loss** — Loss functions: MSE, BCE
//! - **data** — Data loading: Dataset trait, DataLoader with batching
//! - **io** — I/O: CSV read/write, model serialization
//! - **datasets** — Built-in: Iris, make_blobs, make_regression
//! - **pipeline** — Pipeline: composable Transformer + Estimator chains

/// Core tensor engine.
pub use oxidize_ml_core as core;

/// Linear algebra operations.
pub use oxidize_ml_linalg as linalg;

/// Automatic differentiation.
pub use oxidize_ml_autodiff as autodiff;

/// Data preprocessing.
pub use oxidize_ml_preprocessing as preprocessing;

/// Linear models.
pub use oxidize_ml_linear as linear;

/// Tree-based models.
pub use oxidize_ml_tree as tree;

/// Clustering algorithms.
pub use oxidize_ml_cluster as cluster;

/// Nearest neighbors.
pub use oxidize_ml_neighbors as neighbors;

/// Support vector machines.
pub use oxidize_ml_svm as svm;

/// Naive Bayes classifiers.
pub use oxidize_ml_naive_bayes as naive_bayes;

/// Evaluation metrics.
pub use oxidize_ml_metrics as metrics;

/// Neural network layers.
pub use oxidize_ml_nn as nn;

/// Optimizers.
pub use oxidize_ml_optim as optim;

/// Loss functions.
pub use oxidize_ml_loss as loss;

/// Data loading utilities.
pub use oxidize_ml_data as data;

/// I/O utilities.
pub use oxidize_ml_io as io;

/// Built-in datasets.
pub use oxidize_ml_datasets as datasets;

/// Pipeline API.
pub use oxidize_ml_pipeline as pipeline;
