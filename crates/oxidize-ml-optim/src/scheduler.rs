/// Learning rate schedulers for optimizers.
///
/// These modify the learning rate over training steps.

/// Step decay: multiply LR by gamma every step_size epochs.
pub struct StepLR {
    pub initial_lr: f64,
    pub step_size: usize,
    pub gamma: f64,
    pub current_epoch: usize,
}

impl StepLR {
    pub fn new(initial_lr: f64, step_size: usize, gamma: f64) -> Self {
        StepLR { initial_lr, step_size, gamma, current_epoch: 0 }
    }

    pub fn step(&mut self) { self.current_epoch += 1; }

    pub fn get_lr(&self) -> f64 {
        self.initial_lr * self.gamma.powi((self.current_epoch / self.step_size) as i32)
    }
}

/// Exponential decay: LR = initial_lr * gamma^epoch
pub struct ExponentialLR {
    pub initial_lr: f64,
    pub gamma: f64,
    pub current_epoch: usize,
}

impl ExponentialLR {
    pub fn new(initial_lr: f64, gamma: f64) -> Self {
        ExponentialLR { initial_lr, gamma, current_epoch: 0 }
    }

    pub fn step(&mut self) { self.current_epoch += 1; }

    pub fn get_lr(&self) -> f64 {
        self.initial_lr * self.gamma.powi(self.current_epoch as i32)
    }
}

/// Cosine annealing: LR oscillates following a cosine curve.
///
/// LR = η_min + 0.5 * (η_max - η_min) * (1 + cos(π * epoch / T_max))
pub struct CosineAnnealingLR {
    pub initial_lr: f64,
    pub min_lr: f64,
    pub t_max: usize,
    pub current_epoch: usize,
}

impl CosineAnnealingLR {
    pub fn new(initial_lr: f64, t_max: usize) -> Self {
        CosineAnnealingLR { initial_lr, min_lr: 0.0, t_max, current_epoch: 0 }
    }

    pub fn with_min_lr(mut self, min_lr: f64) -> Self {
        self.min_lr = min_lr;
        self
    }

    pub fn step(&mut self) { self.current_epoch += 1; }

    pub fn get_lr(&self) -> f64 {
        let progress = self.current_epoch as f64 / self.t_max as f64;
        self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1.0 + (std::f64::consts::PI * progress).cos())
    }
}

/// Warmup-then-decay: linearly increases LR for warmup_steps, then decays.
pub struct WarmupLR {
    pub target_lr: f64,
    pub warmup_steps: usize,
    pub current_step: usize,
}

impl WarmupLR {
    pub fn new(target_lr: f64, warmup_steps: usize) -> Self {
        WarmupLR { target_lr, warmup_steps, current_step: 0 }
    }

    pub fn step(&mut self) { self.current_step += 1; }

    pub fn get_lr(&self) -> f64 {
        if self.current_step < self.warmup_steps {
            self.target_lr * (self.current_step as f64 / self.warmup_steps as f64)
        } else {
            self.target_lr
        }
    }
}

/// Reduce LR on plateau: reduce when a metric has stopped improving.
pub struct ReduceLROnPlateau {
    pub lr: f64,
    pub factor: f64,
    pub patience: usize,
    pub min_lr: f64,
    best_metric: f64,
    epochs_without_improvement: usize,
}

impl ReduceLROnPlateau {
    pub fn new(initial_lr: f64, factor: f64, patience: usize) -> Self {
        ReduceLROnPlateau {
            lr: initial_lr,
            factor,
            patience,
            min_lr: 1e-7,
            best_metric: f64::INFINITY,
            epochs_without_improvement: 0,
        }
    }

    /// Call with current metric value (lower is better, e.g., loss).
    pub fn step(&mut self, metric: f64) {
        if metric < self.best_metric {
            self.best_metric = metric;
            self.epochs_without_improvement = 0;
        } else {
            self.epochs_without_improvement += 1;
            if self.epochs_without_improvement >= self.patience {
                self.lr = (self.lr * self.factor).max(self.min_lr);
                self.epochs_without_improvement = 0;
            }
        }
    }

    pub fn get_lr(&self) -> f64 { self.lr }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step_lr() {
        let mut sched = StepLR::new(0.1, 10, 0.5);
        assert!((sched.get_lr() - 0.1).abs() < 1e-10);
        for _ in 0..10 { sched.step(); }
        assert!((sched.get_lr() - 0.05).abs() < 1e-10);
        for _ in 0..10 { sched.step(); }
        assert!((sched.get_lr() - 0.025).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_annealing() {
        let mut sched = CosineAnnealingLR::new(0.1, 100);
        assert!((sched.get_lr() - 0.1).abs() < 1e-10);
        for _ in 0..50 { sched.step(); }
        // At halfway, should be ~0.05
        assert!(sched.get_lr() < 0.06 && sched.get_lr() > 0.04);
        for _ in 0..50 { sched.step(); }
        // At end, should be ~0.0
        assert!(sched.get_lr() < 0.01);
    }

    #[test]
    fn test_warmup() {
        let mut sched = WarmupLR::new(0.1, 10);
        assert!((sched.get_lr()).abs() < 1e-10); // epoch 0 = 0
        for _ in 0..5 { sched.step(); }
        assert!((sched.get_lr() - 0.05).abs() < 1e-10); // halfway
        for _ in 0..5 { sched.step(); }
        assert!((sched.get_lr() - 0.1).abs() < 1e-10); // fully warmed
    }

    #[test]
    fn test_reduce_on_plateau() {
        let mut sched = ReduceLROnPlateau::new(0.1, 0.5, 3);
        sched.step(1.0); // improvement
        sched.step(1.1); // no improve, count=1
        sched.step(1.2); // count=2
        sched.step(1.3); // count=3, reduce!
        assert!((sched.get_lr() - 0.05).abs() < 1e-10);
    }
}
