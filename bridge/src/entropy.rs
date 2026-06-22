/// Per-token entropy monitor: tracks a sliding window of entropy values and fires
/// an interrupt when entropy crosses the threshold at a clause/sentence boundary.
///
/// Entropy is computed in Python (MLX is fast at it); this struct only manages
/// the state machine — window tracking, boundary detection, threshold comparison.
pub struct EntropyMonitor {
    threshold: f64,
    min_tokens_per_boundary: usize,
    token_count: usize,
    window_max_entropy: f64,
}

impl EntropyMonitor {
    pub fn new(threshold: f64, min_tokens_per_boundary: usize) -> Self {
        Self {
            threshold,
            min_tokens_per_boundary,
            token_count: 0,
            window_max_entropy: 0.0,
        }
    }

    /// Feed one token's entropy and whether this token is a clause/sentence boundary.
    /// Returns true when generation should pause and the current node should be escalated.
    pub fn update(&mut self, entropy: f64, is_boundary: bool) -> bool {
        self.token_count += 1;
        if entropy > self.window_max_entropy {
            self.window_max_entropy = entropy;
        }

        let should_interrupt = is_boundary
            && self.token_count >= self.min_tokens_per_boundary
            && self.window_max_entropy > self.threshold;

        if is_boundary {
            self.token_count = 0;
            self.window_max_entropy = 0.0;
        }

        should_interrupt
    }

    pub fn reset(&mut self) {
        self.token_count = 0;
        self.window_max_entropy = 0.0;
    }
}

// Utility: compute Shannon entropy from a log-probability vector (already softmaxed + log'd).
// Kept here for testing and for future Rust-level inference backends.
pub fn entropy_from_logprobs(logprobs: &[f32]) -> f64 {
    logprobs
        .iter()
        .map(|&lp| {
            let p = (lp as f64).exp();
            if p > 1e-12 { -p * (lp as f64) } else { 0.0 }
        })
        .sum()
}

// Utility: compute Shannon entropy from raw logits (unnormalized).
pub fn entropy_from_logits(logits: &[f32]) -> f64 {
    let max_l = logits
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max) as f64;
    let exps: Vec<f64> = logits.iter().map(|&l| (l as f64 - max_l).exp()).collect();
    let sum: f64 = exps.iter().sum();
    let log_sum = sum.ln() + max_l;

    logits
        .iter()
        .zip(exps.iter())
        .map(|(&l, &e)| {
            let p = e / sum;
            let lp = l as f64 - log_sum;
            if p > 1e-12 { -p * lp } else { 0.0 }
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_interrupt_before_min_tokens() {
        let mut m = EntropyMonitor::new(1.0, 5);
        for _ in 0..3 {
            assert!(!m.update(5.0, false));
        }
        assert!(!m.update(5.0, true)); // 4th token is boundary, count=4 < min=5
    }

    #[test]
    fn test_interrupt_at_boundary_with_high_entropy() {
        let mut m = EntropyMonitor::new(1.0, 3);
        m.update(2.0, false);
        m.update(2.0, false);
        assert!(m.update(2.0, true)); // crosses threshold at boundary
    }

    #[test]
    fn test_no_interrupt_below_threshold() {
        let mut m = EntropyMonitor::new(3.0, 2);
        m.update(1.0, false);
        assert!(!m.update(1.0, true));
    }

    #[test]
    fn test_entropy_from_uniform_logprobs() {
        // Uniform over 4 classes → entropy = ln(4) ≈ 1.386
        let logp = (0.25_f32).ln();
        let logprobs = vec![logp; 4];
        let h = entropy_from_logprobs(&logprobs);
        assert!((h - 4.0_f64.ln()).abs() < 1e-5, "got {h}");
    }
}
