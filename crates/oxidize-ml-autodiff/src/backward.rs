use std::collections::HashMap;
use oxidize_ml_core::{Float, Tensor};
use crate::graph::{NodeId, Op, with_graph};
use crate::variable::Variable;

/// Compute gradients via reverse-mode automatic differentiation.
///
/// Returns a map from NodeId to gradient Tensor for all nodes that require grad.
pub fn backward(loss: &Variable) -> HashMap<NodeId, Tensor<f64>> {
    with_graph(|graph| {
        let n = graph.len();
        let mut grads: HashMap<NodeId, Tensor<f64>> = HashMap::new();

        // Seed: gradient of loss w.r.t. itself is 1
        let loss_shape = graph.get(loss.node_id).shape.clone();
        let seed = if loss_shape.is_empty() || (loss_shape.len() == 1 && loss_shape[0] == 1) {
            Tensor::scalar(1.0)
        } else {
            Tensor::ones(loss_shape)
        };
        grads.insert(loss.node_id, seed);

        // Reverse topological order (nodes are added in forward order)
        for idx in (0..n).rev() {
            let node_id = NodeId(idx);
            let grad = match grads.get(&node_id) {
                Some(g) => g.clone(),
                None => continue,
            };

            let op = graph.get(node_id).op.clone();

            match op {
                Op::Leaf => {
                    // Leaf nodes accumulate gradients — already stored
                }
                Op::Add(a, b) => {
                    accumulate_grad(&mut grads, a, &grad, &graph.get(a).shape);
                    accumulate_grad(&mut grads, b, &grad, &graph.get(b).shape);
                }
                Op::Sub(a, b) => {
                    accumulate_grad(&mut grads, a, &grad, &graph.get(a).shape);
                    let neg_grad = grad.mul_scalar(-1.0);
                    accumulate_grad(&mut grads, b, &neg_grad, &graph.get(b).shape);
                }
                Op::Mul(a, b) => {
                    // d/da (a*b) = b * grad
                    let ga = grad.mul(&graph.get(b).value).expect("mul grad");
                    accumulate_grad(&mut grads, a, &ga, &graph.get(a).shape);
                    // d/db (a*b) = a * grad
                    let gb = grad.mul(&graph.get(a).value).expect("mul grad");
                    accumulate_grad(&mut grads, b, &gb, &graph.get(b).shape);
                }
                Op::Div(a, b) => {
                    // d/da (a/b) = grad / b
                    let ga = grad.div(&graph.get(b).value).expect("div grad");
                    accumulate_grad(&mut grads, a, &ga, &graph.get(a).shape);
                    // d/db (a/b) = -a * grad / b²
                    let neg_a = graph.get(a).value.mul_scalar(-1.0);
                    let b_sq = graph.get(b).value.mul(&graph.get(b).value).expect("b²");
                    let gb = neg_a.mul(&grad).expect("neg_a * grad").div(&b_sq).expect("/ b²");
                    accumulate_grad(&mut grads, b, &gb, &graph.get(b).shape);
                }
                Op::MatMul(a, b) => {
                    // d/dA (A @ B) = grad @ Bᵀ
                    let bt = graph.get(b).value.t().expect("transpose B");
                    let ga = grad.matmul(&bt).expect("grad @ Bᵀ");
                    accumulate_grad(&mut grads, a, &ga, &graph.get(a).shape);
                    // d/dB (A @ B) = Aᵀ @ grad
                    let at = graph.get(a).value.t().expect("transpose A");
                    let gb = at.matmul(&grad).expect("Aᵀ @ grad");
                    accumulate_grad(&mut grads, b, &gb, &graph.get(b).shape);
                }
                Op::Neg(a) => {
                    let ga = grad.mul_scalar(-1.0);
                    accumulate_grad(&mut grads, a, &ga, &graph.get(a).shape);
                }
                Op::Exp(a) => {
                    // d/da exp(a) = exp(a) * grad
                    let ga = graph.get(node_id).value.mul(&grad).expect("exp grad");
                    accumulate_grad(&mut grads, a, &ga, &graph.get(a).shape);
                }
                Op::Ln(a) => {
                    // d/da ln(a) = grad / a
                    let ga = grad.div(&graph.get(a).value).expect("ln grad");
                    accumulate_grad(&mut grads, a, &ga, &graph.get(a).shape);
                }
                Op::Pow(a, n) => {
                    // d/da a^n = n * a^(n-1) * grad
                    let am1 = graph.get(a).value.powf(Float::from_f64(n - 1.0));
                    let ga = am1.mul_scalar(n).mul(&grad).expect("pow grad");
                    accumulate_grad(&mut grads, a, &ga, &graph.get(a).shape);
                }
                Op::Relu(a) => {
                    // d/da relu(a) = (a > 0) * grad
                    let mask = graph.get(a).value.apply(|x| {
                        if x > 0.0 { 1.0 } else { 0.0 }
                    });
                    let ga = mask.mul(&grad).expect("relu grad");
                    accumulate_grad(&mut grads, a, &ga, &graph.get(a).shape);
                }
                Op::Sigmoid(a) => {
                    // d/da σ(a) = σ(a) * (1 - σ(a)) * grad
                    let sig = &graph.get(node_id).value;
                    let one_minus = sig.apply(|x| 1.0 - x);
                    let ga = sig.mul(&one_minus).expect("sig*(1-sig)")
                        .mul(&grad).expect("* grad");
                    accumulate_grad(&mut grads, a, &ga, &graph.get(a).shape);
                }
                Op::Tanh(a) => {
                    // d/da tanh(a) = (1 - tanh²(a)) * grad
                    let th = &graph.get(node_id).value;
                    let th_sq = th.mul(th).expect("tanh²");
                    let one_minus = th_sq.apply(|x| 1.0 - x);
                    let ga = one_minus.mul(&grad).expect("tanh grad");
                    accumulate_grad(&mut grads, a, &ga, &graph.get(a).shape);
                }
                Op::SumAll(a) => {
                    // Gradient of sum: ones with the shape of a
                    let ga = Tensor::ones(graph.get(a).shape.clone());
                    let ga = ga.mul_scalar(grad.item().unwrap_or(1.0));
                    accumulate_grad(&mut grads, a, &ga, &graph.get(a).shape);
                }
                Op::MeanAll(a) => {
                    let numel = graph.get(a).value.numel();
                    let scale = 1.0 / numel as f64;
                    let ga = Tensor::full(graph.get(a).shape.clone(), scale);
                    let ga = ga.mul_scalar(grad.item().unwrap_or(1.0));
                    accumulate_grad(&mut grads, a, &ga, &graph.get(a).shape);
                }
                Op::Transpose(a) => {
                    let ga = grad.t().expect("transpose grad");
                    accumulate_grad(&mut grads, a, &ga, &graph.get(a).shape);
                }
                Op::MulScalar(a, s) => {
                    let ga = grad.mul_scalar(s);
                    accumulate_grad(&mut grads, a, &ga, &graph.get(a).shape);
                }
                Op::AddScalar(a, _s) => {
                    accumulate_grad(&mut grads, a, &grad, &graph.get(a).shape);
                }
            }
        }

        grads
    })
}

/// Accumulate gradient into the map, handling broadcasting reduction.
fn accumulate_grad(
    grads: &mut HashMap<NodeId, Tensor<f64>>,
    node_id: NodeId,
    incoming_grad: &Tensor<f64>,
    target_shape: &[usize],
) {
    // Reduce grad if it was broadcast
    let grad = reduce_broadcast(incoming_grad, target_shape);

    grads
        .entry(node_id)
        .and_modify(|existing| {
            *existing = existing.add(&grad).expect("grad accumulation");
        })
        .or_insert(grad);
}

/// Reduce a gradient tensor to match the target shape (undo broadcasting).
fn reduce_broadcast(grad: &Tensor<f64>, target_shape: &[usize]) -> Tensor<f64> {
    let grad_shape = grad.shape_vec();
    if grad_shape == target_shape {
        return grad.clone();
    }

    // If target is scalar
    if target_shape.is_empty() || (target_shape.len() == 1 && target_shape[0] == 1) {
        return Tensor::scalar(grad.sum_all());
    }

    let mut result = grad.clone();
    let grad_ndim = grad_shape.len();
    let target_ndim = target_shape.len();

    // Sum over leading dimensions that were broadcast
    if grad_ndim > target_ndim {
        for _ in 0..(grad_ndim - target_ndim) {
            result = result.sum_axis(0).expect("reduce leading dims");
        }
    }

    // Sum over dimensions that are 1 in target but > 1 in grad
    let result_shape = result.shape_vec();
    for (i, (&gs, &ts)) in result_shape.iter().zip(target_shape.iter()).enumerate() {
        if ts == 1 && gs > 1 {
            result = result.sum_axis(i).expect("reduce broadcast dim");
            result = result.unsqueeze(i).expect("restore dim");
        }
    }

    // Final reshape to match target
    if result.shape_vec() != target_shape {
        result = result.reshape(target_shape.to_vec()).unwrap_or_else(|_| {
            Tensor::full(target_shape.to_vec(), result.sum_all() / result.numel() as f64)
        });
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::reset_graph;

    #[test]
    fn test_simple_gradient() {
        reset_graph();

        // f(x) = x², df/dx = 2x
        let x = Variable::param(Tensor::scalar(3.0));
        let y = x.mul(&x);       // x²
        let grads = backward(&y);

        let dx = grads.get(&x.node_id).unwrap();
        assert!((dx.item().unwrap() - 6.0).abs() < 1e-10); // 2 * 3 = 6
    }

    #[test]
    fn test_chain_rule() {
        reset_graph();

        // f(x) = (x + 2)², df/dx = 2(x + 2)
        let x = Variable::param(Tensor::scalar(1.0));
        let y = x.add_scalar(2.0); // x + 2 = 3
        let z = y.mul(&y);         // 9
        let grads = backward(&z);

        let dx = grads.get(&x.node_id).unwrap();
        assert!((dx.item().unwrap() - 6.0).abs() < 1e-10); // 2*(1+2) = 6
    }

    #[test]
    fn test_matmul_gradient() {
        reset_graph();

        // f = sum(A @ B), df/dA = Bᵀ (broadcast), df/dB = Aᵀ (broadcast)
        let a = Variable::param(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap());
        let b = Variable::param(Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap());
        let c = a.matmul(&b);
        let loss = c.sum();
        let grads = backward(&loss);

        // df/dA should be ones @ Bᵀ = [[sum(col1 of B), sum(col2 of B)], ...]
        let da = grads.get(&a.node_id).unwrap();
        assert_eq!(da.shape_vec(), vec![2, 2]);
    }

    #[test]
    fn test_relu_gradient() {
        reset_graph();

        let x = Variable::param(Tensor::new(vec![-1.0, 2.0, -3.0, 4.0], vec![2, 2]).unwrap());
        let y = x.relu();
        let loss = y.sum();
        let grads = backward(&loss);

        let dx = grads.get(&x.node_id).unwrap();
        // ReLU grad: 0 where x < 0, 1 where x >= 0
        assert_eq!(dx.data(), &[0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_sigmoid_gradient() {
        reset_graph();

        let x = Variable::param(Tensor::scalar(0.0));
        let y = x.sigmoid();
        let grads = backward(&y);

        let dx = grads.get(&x.node_id).unwrap();
        // σ(0) = 0.5, σ'(0) = 0.5 * 0.5 = 0.25
        assert!((dx.item().unwrap() - 0.25).abs() < 1e-10);
    }
}
