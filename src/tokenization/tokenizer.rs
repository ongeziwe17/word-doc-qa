use anyhow::{Context, Result};
use std::collections::HashMap;
use tokenizers::Tokenizer;
use tokenizers::models::wordlevel::WordLevelBuilder;
use tokenizers::normalizers::unicode::NFC;
use tokenizers::pre_tokenizers::whitespace::Whitespace;

const PAD_TOKEN: &str = "[PAD]";
const UNK_TOKEN: &str = "[UNK]";

#[derive(Debug, Clone)]
pub struct TokenizedText {
    pub input_ids: Vec<u32>,
    pub attention_mask: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct QaTokenizer {
    tokenizer: Tokenizer,
    pad_id: u32,
}

impl QaTokenizer {
    pub fn encode(&self, text: &str, max_len: usize) -> Result<TokenizedText> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(anyhow::Error::msg)
            .with_context(|| "failed to encode text")?;

        let mut input_ids = encoding.get_ids().to_vec();
        if input_ids.len() > max_len {
            input_ids.truncate(max_len);
        }

        let attention_mask = vec![1; input_ids.len()];

        Ok(TokenizedText {
            input_ids,
            attention_mask,
        })
    }

    pub fn pad_id(&self) -> u32 {
        self.pad_id
    }
}

pub fn build_tokenizer_from_texts(texts: &[String], vocab_limit: usize) -> Result<QaTokenizer> {
    let vocab = build_vocab(texts, vocab_limit);

    let model = WordLevelBuilder::default()
        .vocab(vocab)
        .unk_token(UNK_TOKEN.to_string())
        .build()
        .map_err(anyhow::Error::msg)
        .with_context(|| "failed to build word-level tokenizer model")?;

    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_normalizer(NFC);
    tokenizer.with_pre_tokenizer(Whitespace);

    Ok(QaTokenizer {
        tokenizer,
        pad_id: 0,
    })
}

fn build_vocab(texts: &[String], vocab_limit: usize) -> HashMap<String, u32> {
    let mut counts: HashMap<String, usize> = HashMap::new();

    for text in texts {
        for token in text.split_whitespace() {
            let token = token.to_lowercase();
            *counts.entry(token).or_default() += 1;
        }
    }

    let mut entries: Vec<(String, usize)> = counts.into_iter().collect();
    entries.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let keep = vocab_limit.saturating_sub(2);
    let mut vocab = HashMap::new();
    vocab.insert(PAD_TOKEN.to_string(), 0);
    vocab.insert(UNK_TOKEN.to_string(), 1);

    for (idx, (token, _)) in entries.into_iter().take(keep).enumerate() {
        vocab.insert(token, (idx + 2) as u32);
    }

    vocab
}

#[cfg(test)]
mod tests {
    use super::build_tokenizer_from_texts;

    #[test]
    fn encodes_known_and_unknown_tokens() {
        let texts = vec!["hello world".to_string(), "hello rust".to_string()];
        let tokenizer = build_tokenizer_from_texts(&texts, 100).expect("tokenizer");

        let known = tokenizer.encode("hello world", 16).expect("encode known");
        let unknown = tokenizer
            .encode("hello unknown", 16)
            .expect("encode unknown");

        assert_eq!(known.input_ids.len(), 2);
        assert_eq!(known.attention_mask, vec![1, 1]);
        assert_eq!(unknown.input_ids.len(), 2);
        assert_eq!(unknown.input_ids[1], 1);
    }

    #[test]
    fn truncates_to_max_len() {
        let texts = vec!["a b c d e".to_string()];
        let tokenizer = build_tokenizer_from_texts(&texts, 100).expect("tokenizer");

        let encoded = tokenizer.encode("a b c d e", 3).expect("encode");
        assert_eq!(encoded.input_ids.len(), 3);
        assert_eq!(encoded.attention_mask, vec![1, 1, 1]);
    }
}
