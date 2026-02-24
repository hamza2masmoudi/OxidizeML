use std::cell::Cell;

thread_local! {
    /// A thread-local flag that controls whether the Autodiff Engine records the computation graph tape.
    static IS_GRAD_ENABLED: Cell<bool> = Cell::new(true);
}

/// Temporarily disables gradient tracking within the executed closure.
/// Use this for Inference/Evaluation loops to drastically reduce memory usage by preventing graph allocation.
pub fn with_no_grad<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let prev = IS_GRAD_ENABLED.with(|flag| flag.replace(false));
    let result = f();
    IS_GRAD_ENABLED.with(|flag| flag.set(prev));
    result
}

/// Returns whether gradient tracking is currently globally active on this thread.
pub fn is_grad_enabled() -> bool {
    IS_GRAD_ENABLED.with(|flag| flag.get())
}
