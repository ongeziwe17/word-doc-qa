use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub checkpoint_dir: String,
    pub logs_dir: String,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            epochs: 3,
            batch_size: 4,
            learning_rate: 1e-3,
            checkpoint_dir: "./checkpoints".to_string(),
            logs_dir: "./logs".to_string(),
        }
    }
}
