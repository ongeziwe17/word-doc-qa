use std::collections::HashSet;

use crate::data::corpus::Chunk;

#[derive(Debug, Clone)]
pub struct RetrievedChunk {
    pub chunk: Chunk,
    pub score: f64,
}

pub fn retrieve_top_k(chunks: &[Chunk], question: &str, k: usize) -> Vec<RetrievedChunk> {
    if k == 0 {
        return Vec::new();
    }

    let query_tokens = tokenize_set(question);

    let mut scored: Vec<RetrievedChunk> = chunks
        .iter()
        .cloned()
        .map(|chunk| {
            let chunk_tokens = tokenize_set(&chunk.text);
            let overlap = query_tokens.intersection(&chunk_tokens).count() as f64;
            let denom = (query_tokens.len() + chunk_tokens.len()).max(1) as f64;
            let score = (2.0 * overlap) / denom; // Dice-style overlap
            RetrievedChunk { chunk, score }
        })
        .collect();

    scored.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.chunk.doc_id.cmp(&b.chunk.doc_id))
            .then_with(|| a.chunk.chunk_id.cmp(&b.chunk.chunk_id))
    });

    scored.into_iter().take(k).collect()
}

fn tokenize_set(text: &str) -> HashSet<String> {
    text.split_whitespace()
        .map(|t| {
            t.chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
                .to_lowercase()
        })
        .filter(|t| !t.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::retrieve_top_k;
    use crate::data::corpus::Chunk;

    #[test]
    fn ranks_relevant_chunk_higher() {
        let chunks = vec![
            Chunk {
                doc_id: "a".to_string(),
                chunk_id: 0,
                text: "Rust ownership and borrowing".to_string(),
            },
            Chunk {
                doc_id: "b".to_string(),
                chunk_id: 0,
                text: "Cooking recipes and ingredients".to_string(),
            },
        ];

        let top = retrieve_top_k(&chunks, "What is Rust ownership?", 1);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].chunk.doc_id, "a");
    }

    #[test]
    fn returns_empty_when_k_zero() {
        let chunks = vec![Chunk {
            doc_id: "a".to_string(),
            chunk_id: 0,
            text: "text".to_string(),
        }];
        assert!(retrieve_top_k(&chunks, "text", 0).is_empty());
    }
}
