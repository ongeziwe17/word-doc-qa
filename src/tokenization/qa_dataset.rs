use anyhow::Result;

use crate::tokenization::tokenizer::{QaTokenizer, TokenizedText};

const DEFAULT_QUESTION: &str = "What is this section about?";

#[derive(Debug, Clone)]
pub struct QaTrainingSample {
    pub question: String,
    pub context: String,
    pub answer_text: String,
    pub input_ids: Vec<u32>,
    pub attention_mask: Vec<u8>,
    pub start_position: usize,
    pub end_position: usize,
}

pub fn build_weak_supervised_samples(
    contexts: &[String],
    tokenizer: &QaTokenizer,
    max_len: usize,
) -> Result<Vec<QaTrainingSample>> {
    let mut samples = Vec::new();

    for context in contexts {
        let context = context.trim();
        if context.is_empty() {
            continue;
        }

        let answer_text = first_sentence(context).to_string();
        let encoded_context = tokenizer.encode(context, max_len)?;
        let encoded_answer = tokenizer.encode(&answer_text, max_len)?;

        let (start, end) = find_answer_span(&encoded_context, &encoded_answer);

        samples.push(QaTrainingSample {
            question: DEFAULT_QUESTION.to_string(),
            context: context.to_string(),
            answer_text,
            input_ids: encoded_context.input_ids,
            attention_mask: encoded_context.attention_mask,
            start_position: start,
            end_position: end,
        });
    }

    Ok(samples)
}

fn first_sentence(text: &str) -> &str {
    let mut end_index = text.len();

    for (idx, ch) in text.char_indices() {
        if matches!(ch, '.' | '!' | '?') {
            end_index = idx + ch.len_utf8();
            break;
        }
    }

    text[..end_index].trim()
}

fn find_answer_span(context: &TokenizedText, answer: &TokenizedText) -> (usize, usize) {
    if answer.input_ids.is_empty() || context.input_ids.is_empty() {
        return (0, 0);
    }

    if let Some(start) = find_subsequence(&context.input_ids, &answer.input_ids) {
        let end = start + answer.input_ids.len() - 1;
        return (start, end);
    }

    let fallback_end = usize::min(2, context.input_ids.len().saturating_sub(1));
    (0, fallback_end)
}

fn find_subsequence(haystack: &[u32], needle: &[u32]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }

    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

#[cfg(test)]
mod tests {
    use super::build_weak_supervised_samples;
    use crate::tokenization::tokenizer::build_tokenizer_from_texts;

    #[test]
    fn creates_sample_with_valid_span() {
        let contexts = vec!["Rust is fast. It is memory safe.".to_string()];
        let tokenizer = build_tokenizer_from_texts(&contexts, 100).expect("tokenizer");

        let samples =
            build_weak_supervised_samples(&contexts, &tokenizer, 64).expect("build samples");

        assert_eq!(samples.len(), 1);
        let sample = &samples[0];
        assert!(sample.start_position <= sample.end_position);
        assert!(sample.end_position < sample.input_ids.len());
    }

    #[test]
    fn handles_context_without_sentence_punctuation() {
        let contexts = vec!["Simple context without punctuation".to_string()];
        let tokenizer = build_tokenizer_from_texts(&contexts, 100).expect("tokenizer");

        let samples =
            build_weak_supervised_samples(&contexts, &tokenizer, 64).expect("build samples");

        assert_eq!(samples[0].answer_text, "Simple context without punctuation");
    }
}
