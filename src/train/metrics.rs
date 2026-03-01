use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpochMetrics {
    pub epoch: usize,
    pub avg_loss: f64,
    pub samples_seen: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrainingHistory {
    pub epochs: Vec<EpochMetrics>,
}

impl TrainingHistory {
    pub fn push(&mut self, metrics: EpochMetrics) {
        self.epochs.push(metrics);
    }

    pub fn latest(&self) -> Option<&EpochMetrics> {
        self.epochs.last()
    }
}
