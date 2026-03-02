#![allow(dead_code)]

use crate::tokenization::qa_dataset::QaTrainingSample;
use crate::tokenization::tokenizer::TokenizedText;

#[derive(Debug, Clone)]
pub struct Batch {
    pub input_ids: Vec<Vec<u32>>,
    pub attention_mask: Vec<Vec<u8>>,
}

#[derive(Debug, Clone)]
pub struct QaBatch {
    pub input_ids: Vec<Vec<u32>>,
    pub attention_mask: Vec<Vec<u8>>,
    pub start_positions: Vec<usize>,
    pub end_positions: Vec<usize>,
}

pub fn pad_and_create_batch(samples: &[TokenizedText], pad_id: u32) -> Batch {
    let max_len = samples.iter().map(|s| s.input_ids.len()).max().unwrap_or(0);

    let mut input_ids = Vec::with_capacity(samples.len());
    let mut attention_mask = Vec::with_capacity(samples.len());

    for sample in samples {
        let mut ids = sample.input_ids.clone();
        let mut mask = sample.attention_mask.clone();

        let pad_count = max_len.saturating_sub(ids.len());
        ids.extend(std::iter::repeat_n(pad_id, pad_count));
        mask.extend(std::iter::repeat_n(0, pad_count));

        input_ids.push(ids);
        attention_mask.push(mask);
    }

    Batch {
        input_ids,
        attention_mask,
    }
}

pub fn pad_and_create_qa_batch(samples: &[QaTrainingSample], pad_id: u32) -> QaBatch {
    let max_len = samples.iter().map(|s| s.input_ids.len()).max().unwrap_or(0);

    let mut input_ids = Vec::with_capacity(samples.len());
    let mut attention_mask = Vec::with_capacity(samples.len());
    let mut start_positions = Vec::with_capacity(samples.len());
    let mut end_positions = Vec::with_capacity(samples.len());

    for sample in samples {
        let mut ids = sample.input_ids.clone();
        let mut mask = sample.attention_mask.clone();

        let pad_count = max_len.saturating_sub(ids.len());
        ids.extend(std::iter::repeat_n(pad_id, pad_count));
        mask.extend(std::iter::repeat_n(0, pad_count));

        input_ids.push(ids);
        attention_mask.push(mask);
        start_positions.push(sample.start_position);
        end_positions.push(sample.end_position);
    }

    QaBatch {
        input_ids,
        attention_mask,
        start_positions,
        end_positions,
    }
}

#[cfg(test)]
mod tests {
    use super::{pad_and_create_batch, pad_and_create_qa_batch};
    use crate::tokenization::qa_dataset::QaTrainingSample;
    use crate::tokenization::tokenizer::TokenizedText;

    #[test]
    fn pads_sequences_to_max_len() {
        let samples = vec![
            TokenizedText {
                input_ids: vec![2, 3, 4],
                attention_mask: vec![1, 1, 1],
            },
            TokenizedText {
                input_ids: vec![5],
                attention_mask: vec![1],
            },
        ];

        let batch = pad_and_create_batch(&samples, 0);

        assert_eq!(batch.input_ids, vec![vec![2, 3, 4], vec![5, 0, 0]]);
        assert_eq!(batch.attention_mask, vec![vec![1, 1, 1], vec![1, 0, 0]]);
    }

    #[test]
    fn handles_empty_input() {
        let batch = pad_and_create_batch(&[], 0);
        assert!(batch.input_ids.is_empty());
        assert!(batch.attention_mask.is_empty());
    }

    #[test]
    fn qa_batch_keeps_labels() {
        let samples = vec![
            QaTrainingSample {
                question: "Q".to_string(),
                context: "C1".to_string(),
                answer_text: "A1".to_string(),
                input_ids: vec![1, 2, 3],
                attention_mask: vec![1, 1, 1],
                start_position: 1,
                end_position: 2,
            },
            QaTrainingSample {
                question: "Q".to_string(),
                context: "C2".to_string(),
                answer_text: "A2".to_string(),
                input_ids: vec![4],
                attention_mask: vec![1],
                start_position: 0,
                end_position: 0,
            },
        ];

        let batch = pad_and_create_qa_batch(&samples, 0);

        assert_eq!(batch.input_ids, vec![vec![1, 2, 3], vec![4, 0, 0]]);
        assert_eq!(batch.attention_mask, vec![vec![1, 1, 1], vec![1, 0, 0]]);
        assert_eq!(batch.start_positions, vec![1, 0]);
        assert_eq!(batch.end_positions, vec![2, 0]);
    }
}
