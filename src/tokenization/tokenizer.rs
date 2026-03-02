use anyhow::Result;
use std::collections::HashMap;

const PAD_TOKEN: &str = "[PAD]";
const UNK_TOKEN: &str = "[UNK]";

#[derive(Debug, Clone)]
pub struct TokenizedText {
    pub input_ids: Vec<u32>,
    pub attention_mask: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct QaTokenizer {
    vocab: HashMap<String, u32>,
    unk_id: u32,
}

impl QaTokenizer {
    pub fn from_vocab(vocab: HashMap<String, u32>) -> Self {
        Self { vocab, unk_id: 1 }
    }

    pub fn vocab(&self) -> &HashMap<String, u32> {
        &self.vocab
    }

    pub fn encode_with_tokens(
        &self,
        text: &str,
        max_len: usize,
    ) -> Result<(TokenizedText, Vec<String>)> {
        let mut tokens: Vec<String> = text
            .split_whitespace()
            .map(|t| {
                t.trim_matches(|c: char| !c.is_alphanumeric())
                    .to_lowercase()
            })
            .filter(|t| !t.is_empty())
            .collect();

        if tokens.len() > max_len {
            tokens.truncate(max_len);
        }

        let input_ids = tokens
            .iter()
            .map(|tok| self.vocab.get(tok).copied().unwrap_or(self.unk_id))
            .collect::<Vec<_>>();
        let attention_mask = vec![1; input_ids.len()];

        Ok((
            TokenizedText {
                input_ids,
                attention_mask,
            },
            tokens,
        ))
    }

    pub fn encode(&self, text: &str, max_len: usize) -> Result<TokenizedText> {
        Ok(self.encode_with_tokens(text, max_len)?.0)
    }
}

pub fn build_tokenizer_from_texts(texts: &[String], vocab_limit: usize) -> Result<QaTokenizer> {
    Ok(QaTokenizer::from_vocab(build_vocab(texts, vocab_limit)))
}

fn build_vocab(texts: &[String], vocab_limit: usize) -> HashMap<String, u32> {
    let mut counts: HashMap<String, usize> = HashMap::new();

    for text in texts {
        for token in text.split_whitespace() {
            let token = token
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase();
            if !token.is_empty() {
                *counts.entry(token).or_default() += 1;
            }
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
    use super::{QaTokenizer, build_tokenizer_from_texts};
    use std::collections::HashMap;

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

    #[test]
    fn can_rebuild_from_vocab() {
        let mut vocab = HashMap::new();
        vocab.insert("[PAD]".to_string(), 0);
        vocab.insert("[UNK]".to_string(), 1);
        vocab.insert("rust".to_string(), 2);
        let tok = QaTokenizer::from_vocab(vocab);
        let enc = tok.encode("rust rocks", 10).expect("enc");
        assert_eq!(enc.input_ids, vec![2, 1]);
    }
}
