use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::Write;

use crate::model::QaModelConfig;
use crate::tokenization::qa_dataset::QaTrainingSample;
use crate::train::checkpoint::{TrainingCheckpoint, save_checkpoint};
use crate::train::config::TrainConfig;
use crate::train::eval::evaluate_on_samples;
use crate::train::metrics::{EpochMetrics, TrainingHistory};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearSpanModelState {
    pub start_bias: Vec<f32>,
    pub end_bias: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerState {
    pub step: usize,
    pub learning_rate: f32,
}

#[derive(Debug, Clone)]
pub struct TrainingSummary {
    pub epochs_completed: usize,
    pub final_avg_loss: f64,
    pub history: TrainingHistory,
    pub model_state: LinearSpanModelState,
    pub optimizer_state: OptimizerState,
}

#[derive(Debug, Clone)]
struct TrainBatchTensors {
    input_ids: Vec<Vec<f32>>,
    attention_mask: Vec<Vec<f32>>,
    start_positions: Vec<usize>,
    end_positions: Vec<usize>,
}

pub fn train_weak_supervised(
    samples: &[QaTrainingSample],
    model_config: &QaModelConfig,
    config: &TrainConfig,
) -> Result<TrainingSummary> {
    fs::create_dir_all(&config.logs_dir)
        .with_context(|| format!("failed to create logs dir: {}", config.logs_dir))?;

    let max_seq_len = samples.iter().map(|s| s.input_ids.len()).max().unwrap_or(1);
    let mut model_state = LinearSpanModelState {
        start_bias: vec![0.0; max_seq_len],
        end_bias: vec![0.0; max_seq_len],
    };
    let mut opt_state = OptimizerState {
        step: 0,
        learning_rate: config.learning_rate as f32,
    };

    let mut history = TrainingHistory::default();

    for epoch in 1..=config.epochs {
        let mut total_loss = 0.0f64;
        let mut seen = 0usize;

        for batch in samples.chunks(config.batch_size.max(1)) {
            let tensors = to_train_batch_tensors(batch, max_seq_len);
            let (batch_loss, start_grad, end_grad) =
                cross_entropy_and_grads(&tensors, &model_state);

            apply_sgd_step(
                &mut model_state,
                &mut opt_state,
                &start_grad,
                &end_grad,
                batch.len(),
            );

            total_loss += batch_loss as f64 * batch.len() as f64;
            seen += batch.len();
        }

        let avg_loss = if seen > 0 {
            total_loss / seen as f64
        } else {
            0.0
        };

        let eval = evaluate_on_samples(samples, Some(&model_state));
        println!(
            "epoch {epoch} => loss: {avg_loss:.4}, em: {:.3}, f1: {:.3}",
            eval.exact_match, eval.token_f1
        );

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
            model_state: model_state.clone(),
            optimizer_state: opt_state.clone(),
        };
        let _ = save_checkpoint(&config.checkpoint_dir, &checkpoint)?;
    }

    let final_avg_loss = history.latest().map_or(0.0, |m| m.avg_loss);

    Ok(TrainingSummary {
        epochs_completed: config.epochs,
        final_avg_loss,
        history,
        model_state,
        optimizer_state: opt_state,
    })
}

fn to_train_batch_tensors(batch: &[QaTrainingSample], max_seq_len: usize) -> TrainBatchTensors {
    let mut input_ids = Vec::with_capacity(batch.len());
    let mut attention_mask = Vec::with_capacity(batch.len());
    let mut start_positions = Vec::with_capacity(batch.len());
    let mut end_positions = Vec::with_capacity(batch.len());

    for sample in batch {
        let mut ids: Vec<f32> = sample.input_ids.iter().map(|v| *v as f32).collect();
        let mut mask: Vec<f32> = sample.attention_mask.iter().map(|v| *v as f32).collect();

        let pad = max_seq_len.saturating_sub(ids.len());
        ids.extend(std::iter::repeat_n(0.0, pad));
        mask.extend(std::iter::repeat_n(0.0, pad));

        input_ids.push(ids);
        attention_mask.push(mask);
        start_positions.push(sample.start_position.min(max_seq_len.saturating_sub(1)));
        end_positions.push(sample.end_position.min(max_seq_len.saturating_sub(1)));
    }

    TrainBatchTensors {
        input_ids,
        attention_mask,
        start_positions,
        end_positions,
    }
}

fn cross_entropy_and_grads(
    tensors: &TrainBatchTensors,
    state: &LinearSpanModelState,
) -> (f32, Vec<f32>, Vec<f32>) {
    let seq_len = state.start_bias.len();
    let mut loss = 0.0f32;
    let mut start_grad = vec![0.0f32; seq_len];
    let mut end_grad = vec![0.0f32; seq_len];

    for (i, _ids) in tensors.input_ids.iter().enumerate() {
        let mask = &tensors.attention_mask[i];
        let valid_len = mask.iter().filter(|m| **m > 0.0).count().max(1);

        let start_logits = &state.start_bias[..valid_len];
        let end_logits = &state.end_bias[..valid_len];

        let start_probs = softmax(start_logits);
        let end_probs = softmax(end_logits);

        let y_start = tensors.start_positions[i].min(valid_len - 1);
        let y_end = tensors.end_positions[i].min(valid_len - 1);

        loss += -start_probs[y_start].max(1e-8).ln();
        loss += -end_probs[y_end].max(1e-8).ln();

        for j in 0..valid_len {
            start_grad[j] += start_probs[j] - if j == y_start { 1.0 } else { 0.0 };
            end_grad[j] += end_probs[j] - if j == y_end { 1.0 } else { 0.0 };
        }
    }

    let denom = tensors.input_ids.len().max(1) as f32;
    (loss / denom, start_grad, end_grad)
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|x| (x - max_logit).exp()).collect();
    let sum: f32 = exps.iter().sum::<f32>().max(1e-8);
    exps.into_iter().map(|x| x / sum).collect()
}

fn apply_sgd_step(
    state: &mut LinearSpanModelState,
    opt: &mut OptimizerState,
    start_grad: &[f32],
    end_grad: &[f32],
    batch_size: usize,
) {
    let lr = opt.learning_rate / batch_size.max(1) as f32;
    for i in 0..state.start_bias.len() {
        state.start_bias[i] -= lr * start_grad[i];
        state.end_bias[i] -= lr * end_grad[i];
    }
    opt.step += 1;
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
        assert!(summary.optimizer_state.step > 0);
    }
}
