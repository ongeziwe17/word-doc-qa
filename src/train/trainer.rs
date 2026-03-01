use anyhow::{Context, Result};
use std::fs::{self, OpenOptions};
use std::io::Write;

use crate::model::QaModelConfig;
use crate::tokenization::qa_dataset::QaTrainingSample;
use crate::train::checkpoint::{TrainingCheckpoint, save_checkpoint};
use crate::train::config::TrainConfig;
use crate::train::metrics::{EpochMetrics, TrainingHistory};

#[derive(Debug, Clone)]
pub struct TrainingSummary {
    pub epochs_completed: usize,
    pub final_avg_loss: f64,
    pub history: TrainingHistory,
}

pub fn train_weak_supervised(
    samples: &[QaTrainingSample],
    model_config: &QaModelConfig,
    config: &TrainConfig,
) -> Result<TrainingSummary> {
    fs::create_dir_all(&config.logs_dir)
        .with_context(|| format!("failed to create logs dir: {}", config.logs_dir))?;

    let mut history = TrainingHistory::default();

    for epoch in 1..=config.epochs {
        let mut total_loss = 0.0f64;
        let mut seen = 0usize;

        for batch in samples.chunks(config.batch_size.max(1)) {
            let batch_loss = batch.iter().map(proxy_span_loss).sum::<f64>() / batch.len() as f64;

            total_loss += batch_loss * batch.len() as f64;
            seen += batch.len();
        }

        let avg_loss = if seen > 0 {
            total_loss / seen as f64
        } else {
            0.0
        };

        let metrics = EpochMetrics {
            epoch,
            avg_loss,
            samples_seen: seen,
        };
        append_metrics_line(&config.logs_dir, &metrics)?;
        history.push(metrics.clone());

        let checkpoint = TrainingCheckpoint {
            epoch,
            avg_loss,
            model_config: model_config.clone(),
            train_config: config.clone(),
        };
        let _ = save_checkpoint(&config.checkpoint_dir, &checkpoint)?;
    }

    let final_avg_loss = history.latest().map_or(0.0, |m| m.avg_loss);

    Ok(TrainingSummary {
        epochs_completed: config.epochs,
        final_avg_loss,
        history,
    })
}

fn proxy_span_loss(sample: &QaTrainingSample) -> f64 {
    let span_len = sample
        .end_position
        .saturating_sub(sample.start_position)
        .saturating_add(1) as f64;
    let context_len = sample.input_ids.len().max(1) as f64;

    // A simple deterministic placeholder objective until full Burn backprop is wired.
    1.0 - (span_len / context_len)
}

fn append_metrics_line(logs_dir: &str, metrics: &EpochMetrics) -> Result<()> {
    let path = format!("{logs_dir}/metrics.jsonl");
    let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
    let line = serde_json::to_string(metrics)?;
    writeln!(file, "{line}")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::train_weak_supervised;
    use crate::model::QaModelConfig;
    use crate::tokenization::qa_dataset::QaTrainingSample;
    use crate::train::config::TrainConfig;

    #[test]
    fn trainer_runs_and_returns_summary() {
        let samples = vec![QaTrainingSample {
            question: "Q".to_string(),
            context: "C".to_string(),
            answer_text: "A".to_string(),
            input_ids: vec![1, 2, 3, 4],
            attention_mask: vec![1, 1, 1, 1],
            start_position: 1,
            end_position: 2,
        }];

        let config = TrainConfig {
            epochs: 2,
            batch_size: 1,
            checkpoint_dir: "./target/tmp-train-ckpt".to_string(),
            logs_dir: "./target/tmp-train-logs".to_string(),
            ..Default::default()
        };

        let summary = train_weak_supervised(&samples, &QaModelConfig::default(), &config)
            .expect("train should succeed");

        assert_eq!(summary.epochs_completed, 2);
        assert_eq!(summary.history.epochs.len(), 2);
    }
}
