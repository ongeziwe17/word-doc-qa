use crate::data::corpus::Chunk;
use crate::inference::decode::{best_span_from_logits, compute_token_logits, span_text};
use crate::inference::retrieval::retrieve_top_k;
use crate::model::QaModel;
use crate::tokenization::tokenizer::QaTokenizer;

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
    max_answer_len: usize,
    model: Option<&QaModel>,
    tokenizer: Option<&QaTokenizer>,
) -> Option<AnswerPrediction> {
    let top = retrieve_top_k(chunks, question, top_k.max(1));

    let mut best_pred: Option<AnswerPrediction> = None;

    for candidate in top {
        let (start_logits, end_logits, tokens) =
            if let (Some(model), Some(tokenizer)) = (model, tokenizer) {
                let (encoded, tokens) = tokenizer
                    .encode_with_tokens(&candidate.chunk.text, 256)
                    .ok()?;
                let out = model.forward(&encoded.input_ids);
                (out.start_logits, out.end_logits, tokens)
            } else {
                compute_token_logits(&candidate.chunk.text, question, None, None)
            };

        let (start, end, span_score) =
            best_span_from_logits(&start_logits, &end_logits, max_answer_len)?;
        let answer = span_text(&tokens, start, end, max_answer_len);

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
    use crate::model::{QaModel, QaModelConfig};
    use crate::tokenization::tokenizer::build_tokenizer_from_texts;

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

        let model = QaModel::new(QaModelConfig::default());
        let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
        let tokenizer = build_tokenizer_from_texts(&texts, 100).expect("tok");
        let pred = answer_question(
            &chunks,
            "How does Rust handle memory?",
            3,
            24,
            Some(&model),
            Some(&tokenizer),
        )
        .expect("prediction");
        assert_eq!(pred.source_doc, "doc_rust");
        assert!(!pred.answer.is_empty());
    }
}
