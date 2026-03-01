use crate::data::corpus::Chunk;
use crate::inference::answer_question;
use crate::tokenization::qa_dataset::QaTrainingSample;
use crate::train::trainer::LinearSpanModelState;

#[derive(Debug, Clone, Default)]
pub struct EvalMetrics {
    pub exact_match: f64,
    pub token_f1: f64,
    pub total: usize,
}

pub fn evaluate_on_samples(
    samples: &[QaTrainingSample],
    model_state: Option<&LinearSpanModelState>,
) -> EvalMetrics {
    if samples.is_empty() {
        return EvalMetrics::default();
    }

    let mut em_sum = 0.0;
    let mut f1_sum = 0.0;

    for sample in samples {
        let pseudo_chunk = Chunk {
            doc_id: "eval".to_string(),
            chunk_id: 0,
            text: sample.context.clone(),
        };

        let predicted = answer_question(
            &[pseudo_chunk],
            &sample.question,
            1,
            sample.answer_text.split_whitespace().count().max(1),
            model_state,
        )
        .map(|p| p.answer)
        .unwrap_or_default();

        let gold = normalize(&sample.answer_text);
        let pred = normalize(&predicted);

        if pred == gold {
            em_sum += 1.0;
        }
        f1_sum += token_f1(&pred, &gold);
    }

    let total = samples.len() as f64;
    EvalMetrics {
        exact_match: em_sum / total,
        token_f1: f1_sum / total,
        total: samples.len(),
    }
}

fn normalize(text: &str) -> String {
    text.to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn token_f1(pred: &str, gold: &str) -> f64 {
    let pred_tokens: Vec<&str> = pred.split_whitespace().collect();
    let gold_tokens: Vec<&str> = gold.split_whitespace().collect();

    if pred_tokens.is_empty() || gold_tokens.is_empty() {
        return if pred_tokens == gold_tokens { 1.0 } else { 0.0 };
    }

    let mut common = 0usize;
    let mut used = vec![false; gold_tokens.len()];
    for pt in &pred_tokens {
        for (i, gt) in gold_tokens.iter().enumerate() {
            if !used[i] && pt == gt {
                used[i] = true;
                common += 1;
                break;
            }
        }
    }

    if common == 0 {
        return 0.0;
    }

    let precision = common as f64 / pred_tokens.len() as f64;
    let recall = common as f64 / gold_tokens.len() as f64;
    2.0 * precision * recall / (precision + recall)
}

#[cfg(test)]
mod tests {
    use super::evaluate_on_samples;
    use crate::tokenization::qa_dataset::QaTrainingSample;

    #[test]
    fn eval_returns_valid_range() {
        let samples = vec![QaTrainingSample {
            question: "What is this section about?".to_string(),
            context: "Rust ownership enables memory safety.".to_string(),
            answer_text: "Rust ownership enables memory safety.".to_string(),
            input_ids: vec![1, 2, 3],
            attention_mask: vec![1, 1, 1],
            start_position: 0,
            end_position: 2,
        }];

        let metrics = evaluate_on_samples(&samples, None);
        assert!(metrics.exact_match >= 0.0 && metrics.exact_match <= 1.0);
        assert!(metrics.token_f1 >= 0.0 && metrics.token_f1 <= 1.0);
        assert_eq!(metrics.total, 1);
    }
}
