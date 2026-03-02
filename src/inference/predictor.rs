use crate::data::corpus::Chunk;
use crate::inference::decode::{best_span_from_logits, compute_token_logits, span_text};
use crate::inference::retrieval::retrieve_top_k;
use crate::train::trainer::LinearSpanModelState;

#[derive(Debug, Clone)]
pub struct AnswerPrediction {
    pub answer: String,
    pub source_doc: String,
    pub source_chunk_id: usize,
    pub retrieval_score: f64,
    pub span_score: f32,
}

pub fn answer_question(
    chunks: &[Chunk],
    question: &str,
    top_k: usize,
    max_answer_words: usize,
    model_state: Option<&LinearSpanModelState>,
) -> Option<AnswerPrediction> {
    let top = retrieve_top_k(chunks, question, top_k.max(1));

    let mut best_pred: Option<AnswerPrediction> = None;

    for candidate in top {
        let start_prior = model_state.map(|m| m.start_bias.as_slice());
        let end_prior = model_state.map(|m| m.end_bias.as_slice());
        let (start_logits, end_logits, tokens) =
            compute_token_logits(&candidate.chunk.text, question, start_prior, end_prior);
        let (start, end, span_score) =
            best_span_from_logits(&start_logits, &end_logits, max_answer_words)?;
        let answer = span_text(&tokens, start, end, max_answer_words);

        let pred = AnswerPrediction {
            answer,
            source_doc: candidate.chunk.doc_id.clone(),
            source_chunk_id: candidate.chunk.chunk_id,
            retrieval_score: candidate.score,
            span_score,
        };

        let is_better = best_pred
            .as_ref()
            .map(|b| pred.span_score > b.span_score)
            .unwrap_or(true);
        if is_better {
            best_pred = Some(pred);
        }
    }

    best_pred
}

#[cfg(test)]
mod tests {
    use super::answer_question;
    use crate::data::corpus::Chunk;

    #[test]
    fn predicts_from_best_retrieved_chunk() {
        let chunks = vec![
            Chunk {
                doc_id: "doc_rust".to_string(),
                chunk_id: 0,
                text: "Rust uses ownership for memory safety without garbage collection."
                    .to_string(),
            },
            Chunk {
                doc_id: "doc_other".to_string(),
                chunk_id: 0,
                text: "Basketball has five players per team on the court.".to_string(),
            },
        ];

        let pred = answer_question(&chunks, "How does Rust handle memory?", 3, 24, None)
            .expect("prediction");
        assert_eq!(pred.source_doc, "doc_rust");
        assert!(pred.answer.to_lowercase().contains("rust"));
    }
}
