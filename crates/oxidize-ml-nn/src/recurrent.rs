use oxidize_ml_core::Tensor;
use oxidize_ml_autodiff::Variable;

/// Simple RNN Cell.
///
/// h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b)
pub struct RNNCell {
    pub w_ih: Variable,  // [hidden_size, input_size]
    pub w_hh: Variable,  // [hidden_size, hidden_size]
    pub bias: Variable,  // [1, hidden_size]
    pub hidden_size: usize,
    pub input_size: usize,
}

impl RNNCell {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let scale = (1.0 / hidden_size as f64).sqrt();
        let w_ih = Tensor::rand(vec![input_size, hidden_size], Some(42))
            .mul_scalar(2.0 * scale).add_scalar(-scale);
        let w_hh = Tensor::rand(vec![hidden_size, hidden_size], Some(43))
            .mul_scalar(2.0 * scale).add_scalar(-scale);
        let bias = Tensor::zeros(vec![1, hidden_size]);

        RNNCell {
            w_ih: Variable::param(w_ih),
            w_hh: Variable::param(w_hh),
            bias: Variable::param(bias),
            hidden_size,
            input_size,
        }
    }

    /// Forward one time step.
    /// x: [batch, input_size], h: [batch, hidden_size]
    /// Returns new h: [batch, hidden_size]
    pub fn forward(&self, x: &Variable, h: &Variable) -> Variable {
        let xw = x.matmul(&self.w_ih);       // [batch, hidden_size]
        let hw = h.matmul(&self.w_hh);        // [batch, hidden_size]
        let pre_act = xw.add(&hw).add(&self.bias);
        pre_act.tanh_act()
    }

    /// Forward over a sequence.
    /// x: [seq_len, batch, input_size]
    /// Returns final hidden state [batch, hidden_size]
    pub fn forward_seq(&self, inputs: &[Variable], h0: &Variable) -> Vec<Variable> {
        let mut h = h0.clone();
        let mut outputs = Vec::with_capacity(inputs.len());
        for x in inputs {
            h = self.forward(x, &h);
            outputs.push(h.clone());
        }
        outputs
    }

    pub fn parameters(&self) -> Vec<Variable> {
        vec![self.w_ih.clone(), self.w_hh.clone(), self.bias.clone()]
    }
}

/// GRU Cell (Gated Recurrent Unit).
///
/// z_t = σ(W_z @ [h_{t-1}, x_t])    -- update gate
/// r_t = σ(W_r @ [h_{t-1}, x_t])    -- reset gate
/// ĥ_t = tanh(W @ [r_t * h_{t-1}, x_t])  -- candidate
/// h_t = (1 - z_t) * h_{t-1} + z_t * ĥ_t
pub struct GRUCell {
    pub input_size: usize,
    pub hidden_size: usize,
    // Update gate
    pub w_z_x: Variable,  // [input_size, hidden_size]
    pub w_z_h: Variable,  // [hidden_size, hidden_size]
    pub b_z: Variable,    // [1, hidden_size]
    // Reset gate
    pub w_r_x: Variable,
    pub w_r_h: Variable,
    pub b_r: Variable,
    // Candidate
    pub w_n_x: Variable,
    pub w_n_h: Variable,
    pub b_n: Variable,
}

impl GRUCell {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let scale = (1.0 / hidden_size as f64).sqrt();
        let mk = |seed: u64| {
            Tensor::rand(vec![input_size, hidden_size], Some(seed))
                .mul_scalar(2.0 * scale).add_scalar(-scale)
        };
        let mkh = |seed: u64| {
            Tensor::rand(vec![hidden_size, hidden_size], Some(seed))
                .mul_scalar(2.0 * scale).add_scalar(-scale)
        };

        GRUCell {
            input_size, hidden_size,
            w_z_x: Variable::param(mk(10)), w_z_h: Variable::param(mkh(11)),
            b_z: Variable::param(Tensor::zeros(vec![1, hidden_size])),
            w_r_x: Variable::param(mk(20)), w_r_h: Variable::param(mkh(21)),
            b_r: Variable::param(Tensor::zeros(vec![1, hidden_size])),
            w_n_x: Variable::param(mk(30)), w_n_h: Variable::param(mkh(31)),
            b_n: Variable::param(Tensor::zeros(vec![1, hidden_size])),
        }
    }

    /// Forward one time step.
    pub fn forward(&self, x: &Variable, h: &Variable) -> Variable {
        // Update gate: z = σ(x @ W_z_x + h @ W_z_h + b_z)
        let z = x.matmul(&self.w_z_x).add(&h.matmul(&self.w_z_h)).add(&self.b_z).sigmoid();

        // Reset gate: r = σ(x @ W_r_x + h @ W_r_h + b_r)
        let r = x.matmul(&self.w_r_x).add(&h.matmul(&self.w_r_h)).add(&self.b_r).sigmoid();

        // Candidate: n = tanh(x @ W_n_x + (r * h) @ W_n_h + b_n)
        let rh = r.mul(h);
        let n = x.matmul(&self.w_n_x).add(&rh.matmul(&self.w_n_h)).add(&self.b_n).tanh_act();

        // h_new = (1 - z) * h + z * n
        let one = Variable::input(Tensor::ones(z.data.shape_vec()));
        let one_minus_z = one.sub(&z);
        one_minus_z.mul(h).add(&z.mul(&n))
    }

    pub fn forward_seq(&self, inputs: &[Variable], h0: &Variable) -> Vec<Variable> {
        let mut h = h0.clone();
        let mut outputs = Vec::with_capacity(inputs.len());
        for x in inputs {
            h = self.forward(x, &h);
            outputs.push(h.clone());
        }
        outputs
    }

    pub fn parameters(&self) -> Vec<Variable> {
        vec![
            self.w_z_x.clone(), self.w_z_h.clone(), self.b_z.clone(),
            self.w_r_x.clone(), self.w_r_h.clone(), self.b_r.clone(),
            self.w_n_x.clone(), self.w_n_h.clone(), self.b_n.clone(),
        ]
    }
}

/// LSTM Cell (Long Short-Term Memory).
///
/// i_t = σ(W_i @ [h_{t-1}, x_t] + b_i)   -- input gate
/// f_t = σ(W_f @ [h_{t-1}, x_t] + b_f)   -- forget gate
/// o_t = σ(W_o @ [h_{t-1}, x_t] + b_o)   -- output gate
/// g_t = tanh(W_g @ [h_{t-1}, x_t] + b_g) -- cell candidate
/// c_t = f_t * c_{t-1} + i_t * g_t         -- cell state
/// h_t = o_t * tanh(c_t)                   -- hidden state
pub struct LSTMCell {
    pub input_size: usize,
    pub hidden_size: usize,
    // Input gate
    pub w_i_x: Variable, pub w_i_h: Variable, pub b_i: Variable,
    // Forget gate
    pub w_f_x: Variable, pub w_f_h: Variable, pub b_f: Variable,
    // Output gate
    pub w_o_x: Variable, pub w_o_h: Variable, pub b_o: Variable,
    // Cell candidate
    pub w_g_x: Variable, pub w_g_h: Variable, pub b_g: Variable,
}

impl LSTMCell {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let scale = (1.0 / hidden_size as f64).sqrt();
        let mk = |seed: u64| {
            Tensor::rand(vec![input_size, hidden_size], Some(seed))
                .mul_scalar(2.0 * scale).add_scalar(-scale)
        };
        let mkh = |seed: u64| {
            Tensor::rand(vec![hidden_size, hidden_size], Some(seed))
                .mul_scalar(2.0 * scale).add_scalar(-scale)
        };
        let bz = || Tensor::zeros(vec![1, hidden_size]);

        LSTMCell {
            input_size, hidden_size,
            w_i_x: Variable::param(mk(40)), w_i_h: Variable::param(mkh(41)), b_i: Variable::param(bz()),
            w_f_x: Variable::param(mk(50)), w_f_h: Variable::param(mkh(51)),
            b_f: Variable::param(Tensor::ones(vec![1, hidden_size])), // Forget gate bias init to 1
            w_o_x: Variable::param(mk(60)), w_o_h: Variable::param(mkh(61)), b_o: Variable::param(bz()),
            w_g_x: Variable::param(mk(70)), w_g_h: Variable::param(mkh(71)), b_g: Variable::param(bz()),
        }
    }

    /// Forward one step. Returns (h_new, c_new).
    pub fn forward(&self, x: &Variable, h: &Variable, c: &Variable) -> (Variable, Variable) {
        let i = x.matmul(&self.w_i_x).add(&h.matmul(&self.w_i_h)).add(&self.b_i).sigmoid();
        let f = x.matmul(&self.w_f_x).add(&h.matmul(&self.w_f_h)).add(&self.b_f).sigmoid();
        let o = x.matmul(&self.w_o_x).add(&h.matmul(&self.w_o_h)).add(&self.b_o).sigmoid();
        let g = x.matmul(&self.w_g_x).add(&h.matmul(&self.w_g_h)).add(&self.b_g).tanh_act();

        let c_new = f.mul(c).add(&i.mul(&g));
        let h_new = o.mul(&c_new.tanh_act());
        (h_new, c_new)
    }

    /// Forward over a sequence.
    pub fn forward_seq(&self, inputs: &[Variable], h0: &Variable, c0: &Variable)
        -> (Vec<Variable>, Variable, Variable)
    {
        let mut h = h0.clone();
        let mut c = c0.clone();
        let mut outputs = Vec::with_capacity(inputs.len());
        for x in inputs {
            let (h_new, c_new) = self.forward(x, &h, &c);
            h = h_new;
            c = c_new;
            outputs.push(h.clone());
        }
        (outputs, h, c)
    }

    pub fn parameters(&self) -> Vec<Variable> {
        vec![
            self.w_i_x.clone(), self.w_i_h.clone(), self.b_i.clone(),
            self.w_f_x.clone(), self.w_f_h.clone(), self.b_f.clone(),
            self.w_o_x.clone(), self.w_o_h.clone(), self.b_o.clone(),
            self.w_g_x.clone(), self.w_g_h.clone(), self.b_g.clone(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rnn_cell() {
        let cell = RNNCell::new(4, 8);
        let x = Variable::input(Tensor::rand(vec![2, 4], Some(1)));
        let h = Variable::input(Tensor::zeros(vec![2, 8]));
        let h_new = cell.forward(&x, &h);
        assert_eq!(h_new.data.shape_vec(), vec![2, 8]);
    }

    #[test]
    fn test_gru_cell() {
        let cell = GRUCell::new(4, 8);
        let x = Variable::input(Tensor::rand(vec![2, 4], Some(1)));
        let h = Variable::input(Tensor::zeros(vec![2, 8]));
        let h_new = cell.forward(&x, &h);
        assert_eq!(h_new.data.shape_vec(), vec![2, 8]);
    }

    #[test]
    fn test_lstm_cell() {
        let cell = LSTMCell::new(4, 8);
        let x = Variable::input(Tensor::rand(vec![2, 4], Some(1)));
        let h = Variable::input(Tensor::zeros(vec![2, 8]));
        let c = Variable::input(Tensor::zeros(vec![2, 8]));
        let (h_new, c_new) = cell.forward(&x, &h, &c);
        assert_eq!(h_new.data.shape_vec(), vec![2, 8]);
        assert_eq!(c_new.data.shape_vec(), vec![2, 8]);
    }

    #[test]
    fn test_lstm_sequence() {
        let cell = LSTMCell::new(3, 5);
        let seq: Vec<Variable> = (0..4)
            .map(|i| Variable::input(Tensor::rand(vec![1, 3], Some(i))))
            .collect();
        let h0 = Variable::input(Tensor::zeros(vec![1, 5]));
        let c0 = Variable::input(Tensor::zeros(vec![1, 5]));
        let (outputs, h_final, _c_final) = cell.forward_seq(&seq, &h0, &c0);
        assert_eq!(outputs.len(), 4);
        assert_eq!(h_final.data.shape_vec(), vec![1, 5]);
    }
}
