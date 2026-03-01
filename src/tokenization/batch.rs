use crate::tokenization::tokenizer::TokenizedText;

#[derive(Debug, Clone)]
pub struct Batch {
    pub input_ids: Vec<Vec<u32>>,
    pub attention_mask: Vec<Vec<u8>>,
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

#[cfg(test)]
mod tests {
    use super::pad_and_create_batch;
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
}
