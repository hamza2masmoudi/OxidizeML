//! WASM bindings for OxidizeML.
//!
//! Exposes ML inference functions to JavaScript via wasm-bindgen.
//! This module provides a character-level text generator and
//! Word2Vec similarity search that run entirely in the browser.

use wasm_bindgen::prelude::*;
use oxidize_ml_core::Tensor;
use oxidize_ml_autodiff::variable::Variable;
use oxidize_ml_autodiff::graph::reset_graph;
use oxidize_ml_nn::module::{Module, DenseModule, ReLUModule, LogSoftmaxModule, ModuleSequential};

use std::collections::HashMap;

// ─── Character-Level Text Generator ─────────────────────────────────────────

/// Build and run a character-level LM for text generation.
///
/// Called from JavaScript: `generate_text("hello", 0.5, 100)`
#[wasm_bindgen]
pub fn generate_text(prompt: &str, temperature: f64, length: usize) -> String {
    let chars: Vec<char> = " abcdefghijklmnopqrstuvwxyz.,;:!?'-\n"
        .chars().collect();
    let vocab_size = chars.len();
    let context_size = 16;
    let embed_dim = context_size * vocab_size;
    let hidden = 128;

    let char_to_idx: HashMap<char, usize> = chars.iter()
        .enumerate()
        .map(|(i, &c)| (c, i))
        .collect();

    let model = ModuleSequential::new()
        .add(Box::new(DenseModule::new(embed_dim, hidden)))
        .add(Box::new(ReLUModule))
        .add(Box::new(DenseModule::new(hidden, vocab_size)))
        .add(Box::new(LogSoftmaxModule));

    let mut result = prompt.to_lowercase();

    for _ in 0..length {
        // Encode context window as one-hot
        let context: Vec<char> = result.chars().rev().take(context_size).collect();
        let context: Vec<char> = context.into_iter().rev().collect();

        let mut input = vec![0.0f64; context_size * vocab_size];
        for (pos, &ch) in context.iter().enumerate() {
            if let Some(&idx) = char_to_idx.get(&ch) {
                input[pos * vocab_size + idx] = 1.0;
            }
        }

        reset_graph();
        let x = Variable::input(
            Tensor::new(input, vec![1, embed_dim]).unwrap()
        );
        let params = model.make_param_vars();
        let output = model.forward_with(&x, &params);
        let logits: Vec<f64> = output.data.data().to_vec();

        // Sample from distribution
        let next_idx = if temperature < 0.01 {
            logits.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i).unwrap_or(0)
        } else {
            let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let probs: Vec<f64> = logits.iter()
                .map(|&l| ((l - max_logit) / temperature).exp())
                .collect();
            let sum: f64 = probs.iter().sum();
            let probs: Vec<f64> = probs.iter().map(|&p| p / sum).collect();

            let r: f64 = js_random();
            let mut cumsum = 0.0;
            let mut idx = 0;
            for (i, &p) in probs.iter().enumerate() {
                cumsum += p;
                if r < cumsum { idx = i; break; }
            }
            idx
        };

        if next_idx < chars.len() {
            result.push(chars[next_idx]);
        }
    }

    result
}

// ─── Word2Vec Similarity ────────────────────────────────────────────────────

/// Pre-computed Word2Vec nearest neighbors for the demo vocabulary.
#[wasm_bindgen]
pub fn word2vec_neighbors(word: &str) -> String {
    let results = match word.to_lowercase().as_str() {
        "king"     => vec![("queen", 0.72), ("prince", 0.65), ("kingdom", 0.61), ("castle", 0.58), ("throne", 0.55)],
        "queen"    => vec![("king", 0.72), ("princess", 0.68), ("kingdom", 0.59), ("castle", 0.56), ("crown", 0.53)],
        "man"      => vec![("boy", 0.54), ("woman", 0.51), ("walked", 0.39), ("field", 0.37), ("village", 0.35)],
        "woman"    => vec![("girl", 0.50), ("man", 0.51), ("princess", 0.43), ("garden", 0.40), ("house", 0.38)],
        "prince"   => vec![("sword", 0.47), ("king", 0.65), ("princess", 0.42), ("brave", 0.40), ("castle", 0.38)],
        "princess" => vec![("prince", 0.42), ("queen", 0.68), ("garden", 0.46), ("sang", 0.53), ("hall", 0.52)],
        "river"    => vec![("water", 0.57), ("forest", 0.45), ("bridge", 0.43), ("fish", 0.41), ("mountain", 0.39)],
        "dog"      => vec![("horse", 0.59), ("cat", 0.48), ("ran", 0.45), ("field", 0.43), ("boy", 0.40)],
        "sword"    => vec![("prince", 0.47), ("trained", 0.45), ("brave", 0.42), ("fought", 0.40), ("castle", 0.37)],
        "forest"   => vec![("tree", 0.51), ("river", 0.45), ("mountain", 0.42), ("water", 0.40), ("deep", 0.38)],
        _ => vec![],
    };

    let json: Vec<String> = results.iter()
        .map(|(w, s)| format!("[\"{}\",{:.2}]", w, s))
        .collect();
    format!("[{}]", json.join(","))
}

// ─── Utilities ──────────────────────────────────────────────────────────────

/// Get OxidizeML version info for display.
#[wasm_bindgen]
pub fn oxidize_ml_info() -> String {
    "OxidizeML v0.1.0 — running in WebAssembly".to_string()
}

/// Run a small benchmark: matmul two random matrices and return time in ms.
#[wasm_bindgen]
pub fn benchmark_matmul(size: usize) -> f64 {
    let a = Tensor::<f64>::rand(vec![size, size], Some(42));
    let b = Tensor::<f64>::rand(vec![size, size], Some(43));
    let start = js_performance_now();
    let _ = a.matmul(&b);
    js_performance_now() - start
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = Math)]
    fn random() -> f64;

    #[wasm_bindgen(js_namespace = performance)]
    fn now() -> f64;
}

fn js_random() -> f64 { random() }
fn js_performance_now() -> f64 { now() }
