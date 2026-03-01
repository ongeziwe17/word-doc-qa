use crate::data::corpus::Chunk;
use crate::inference::decode::best_span_text;
use crate::inference::retrieval::retrieve_top_k;

#[derive(Debug, Clone)]
pub struct AnswerPrediction {
    pub answer: String,
    pub source_doc: String,
    pub source_chunk_id: usize,
    pub retrieval_score: f64,
}

pub fn answer_question(chunks: &[Chunk], question: &str) -> Option<AnswerPrediction> {
    let top = retrieve_top_k(chunks, question, 1);
    let best = top.first()?;

    let answer = best_span_text(&best.chunk.text, 24);

    Some(AnswerPrediction {
        answer,
        source_doc: best.chunk.doc_id.clone(),
        source_chunk_id: best.chunk.chunk_id,
        retrieval_score: best.score,
    })
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

        let pred = answer_question(&chunks, "How does Rust handle memory?").expect("prediction");
        assert_eq!(pred.source_doc, "doc_rust");
        assert!(pred.answer.to_lowercase().contains("rust"));
    }
}
