use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::Write;

use crate::model::{QaModel, QaModelConfig};
use crate::tokenization::qa_dataset::QaTrainingSample;
use crate::train::checkpoint::{TrainingCheckpoint, save_checkpoint};
use crate::train::config::TrainConfig;
use crate::train::eval::evaluate_on_samples;
use crate::train::metrics::{EpochMetrics, TrainingHistory};

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
    pub model: QaModel,
    pub optimizer_state: OptimizerState,
}

pub fn train_weak_supervised(
    samples: &[QaTrainingSample],
    model_config: &QaModelConfig,
    config: &TrainConfig,
) -> Result<TrainingSummary> {
    fs::create_dir_all(&config.logs_dir)
        .with_context(|| format!("failed to create logs dir: {}", config.logs_dir))?;

    let mut model = QaModel::new(model_config.clone());
    let mut opt_state = OptimizerState {
        step: 0,
        learning_rate: config.learning_rate as f32,
    };

    let (train_samples, val_samples) = split_train_val(samples, config.val_split);
    let mut history = TrainingHistory::default();
    let mut best_val_loss = f32::INFINITY;
    let mut stale_epochs = 0usize;

    for epoch in 1..=config.epochs {
        let mut total_loss = 0.0f64;
        let mut seen = 0usize;

        for batch in train_samples.chunks(config.batch_size.max(1)) {
            let (batch_loss, grads) = batch_loss_and_grads(&model, batch);
            apply_sgd_step(&mut model, &mut opt_state, grads);
            total_loss += batch_loss as f64 * batch.len() as f64;
            seen += batch.len();
        }

        let avg_loss = if seen > 0 {
            total_loss / seen as f64
        } else {
            0.0
        };

        let val_loss = if val_samples.is_empty() {
            avg_loss as f32
        } else {
            average_loss(&model, val_samples)
        };

        let eval = evaluate_on_samples(val_samples, Some(&model));
        println!(
            "epoch {epoch} => train_loss: {avg_loss:.4}, val_loss: {val_loss:.4}, em: {:.3}, f1: {:.3}",
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
            model: model.clone(),
            optimizer_state: opt_state.clone(),
        };
        let _ = save_checkpoint(&config.checkpoint_dir, &checkpoint)?;

        if val_loss + 1e-6 < best_val_loss {
            best_val_loss = val_loss;
            stale_epochs = 0;
        } else {
            stale_epochs += 1;
            if stale_epochs >= config.early_stopping_patience.max(1) {
                break;
            }
        }
    }

    let final_avg_loss = history.latest().map_or(0.0, |m| m.avg_loss);

    Ok(TrainingSummary {
        epochs_completed: history.epochs.len(),
        final_avg_loss,
        history,
        model,
        optimizer_state: opt_state,
    })
}

#[derive(Debug, Clone, Copy)]
struct ModelGrads {
    start_weight: f32,
    end_weight: f32,
    start_bias: f32,
    end_bias: f32,
}

fn split_train_val(
    samples: &[QaTrainingSample],
    val_split: f32,
) -> (&[QaTrainingSample], &[QaTrainingSample]) {
    if samples.len() <= 1 {
        return (samples, &[]);
    }

    let val_ratio = val_split.clamp(0.0, 0.8);
    let train_len = ((samples.len() as f32) * (1.0 - val_ratio)).round() as usize;
    let train_len = train_len.clamp(1, samples.len() - 1);
    (&samples[..train_len], &samples[train_len..])
}

fn average_loss(model: &QaModel, samples: &[QaTrainingSample]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut total = 0.0;
    for sample in samples {
        total += sample_loss(model, sample);
    }
    total / samples.len() as f32
}

fn sample_loss(model: &QaModel, sample: &QaTrainingSample) -> f32 {
    let output = model.forward(&sample.input_ids);
    let valid_len = sample
        .attention_mask
        .iter()
        .filter(|m| **m > 0)
        .count()
        .max(1);
    let y_start = sample.start_position.min(valid_len - 1);
    let y_end = sample.end_position.min(valid_len - 1);

    let start_probs = softmax(&output.start_logits[..valid_len]);
    let end_probs = softmax(&output.end_logits[..valid_len]);

    -start_probs[y_start].max(1e-8).ln() - end_probs[y_end].max(1e-8).ln()
}

fn batch_loss_and_grads(model: &QaModel, batch: &[QaTrainingSample]) -> (f32, ModelGrads) {
    let mut loss = 0.0f32;
    let mut grads = ModelGrads {
        start_weight: 0.0,
        end_weight: 0.0,
        start_bias: 0.0,
        end_bias: 0.0,
    };

    for sample in batch {
        let output = model.forward(&sample.input_ids);
        let valid_len = sample
            .attention_mask
            .iter()
            .filter(|m| **m > 0)
            .count()
            .max(1);
        let y_start = sample.start_position.min(valid_len - 1);
        let y_end = sample.end_position.min(valid_len - 1);

        let start_probs = softmax(&output.start_logits[..valid_len]);
        let end_probs = softmax(&output.end_logits[..valid_len]);

        loss += -start_probs[y_start].max(1e-8).ln();
        loss += -end_probs[y_end].max(1e-8).ln();

        for i in 0..valid_len {
            let x = sample.input_ids[i] as f32 / model.config.vocab_size.max(1) as f32;
            let d_start = start_probs[i] - if i == y_start { 1.0 } else { 0.0 };
            let d_end = end_probs[i] - if i == y_end { 1.0 } else { 0.0 };
            grads.start_weight += d_start * x;
            grads.start_bias += d_start;
            grads.end_weight += d_end * x;
            grads.end_bias += d_end;
        }
    }

    let denom = batch.len().max(1) as f32;
    grads.start_weight /= denom;
    grads.end_weight /= denom;
    grads.start_bias /= denom;
    grads.end_bias /= denom;

    (loss / denom, grads)
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|x| (x - max_logit).exp()).collect();
    let sum: f32 = exps.iter().sum::<f32>().max(1e-8);
    exps.into_iter().map(|x| x / sum).collect()
}

fn apply_sgd_step(model: &mut QaModel, opt: &mut OptimizerState, grads: ModelGrads) {
    let lr = opt.learning_rate;
    model.start_weight -= lr * grads.start_weight;
    model.end_weight -= lr * grads.end_weight;
    model.start_bias -= lr * grads.start_bias;
    model.end_bias -= lr * grads.end_bias;
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

        assert!(summary.epochs_completed >= 1);
        assert!(!summary.history.epochs.is_empty());
        assert!(summary.optimizer_state.step > 0);
    }
}
