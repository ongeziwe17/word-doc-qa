mod data;
mod inference;
mod model;
mod tokenization;
mod train;

use anyhow::Result;
use data::load_corpus_from_dir;
use inference::answer_question;
use model::QaModelConfig;
use tokenization::{
    build_tokenizer_from_texts, build_weak_supervised_samples, pad_and_create_batch,
    pad_and_create_qa_batch,
};
use train::{TrainConfig, load_latest_checkpoint, train_weak_supervised};

fn main() -> Result<()> {
    env_logger::init();

    let chunks = load_corpus_from_dir("./data", 1_000)?;
    println!("Loaded {} chunks from ./data", chunks.len());

    if let Some(first) = chunks.first() {
        let preview: String = first.text.chars().take(120).collect();
        println!(
            "First chunk => doc: {}, chunk: {}, preview: {}",
            first.doc_id, first.chunk_id, preview
        );

        let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
        let tokenizer = build_tokenizer_from_texts(&texts, 8_000)?;

        let take_n = usize::min(2, chunks.len());
        let tokenized: Vec<_> = chunks
            .iter()
            .take(take_n)
            .map(|chunk| tokenizer.encode(&chunk.text, 64))
            .collect::<Result<Vec<_>>>()?;

        let batch = pad_and_create_batch(&tokenized, tokenizer.pad_id());
        let seq_len = batch.input_ids.first().map_or(0, Vec::len);

        println!(
            "Tokenization => batch_size: {}, seq_len: {}",
            batch.input_ids.len(),
            seq_len
        );

        let qa_samples = build_weak_supervised_samples(&texts, &tokenizer, 64)?;
        let qa_take_n = usize::min(2, qa_samples.len());
        let qa_batch = pad_and_create_qa_batch(&qa_samples[..qa_take_n], tokenizer.pad_id());

        println!(
            "Weak QA dataset => samples: {}, batch_size: {}, first_span: ({}, {})",
            qa_samples.len(),
            qa_batch.input_ids.len(),
            qa_batch.start_positions.first().copied().unwrap_or(0),
            qa_batch.end_positions.first().copied().unwrap_or(0)
        );

        let model_config = QaModelConfig::default();
        println!(
            "Model config => d_model: {}, ff_dim: {}, encoder_layers: {}",
            model_config.d_model,
            model_config.ff_dim,
            model_config.effective_num_layers()
        );

        let train_config = TrainConfig::default();
        let summary = train_weak_supervised(&qa_samples, &model_config, &train_config)?;
        println!(
            "Training => epochs: {}, final_avg_loss: {:.4}",
            summary.epochs_completed, summary.final_avg_loss
        );

        if let Some(last_ckpt) = load_latest_checkpoint(&train_config.checkpoint_dir)? {
            println!(
                "Checkpoint => epoch: {}, avg_loss: {:.4}",
                last_ckpt.epoch, last_ckpt.avg_loss
            );
        }
        let question = "What is this document about?";
        if let Some(prediction) = answer_question(&chunks, question) {
            println!(
                "Ask => q: {} | answer: {} | source: {}#{} | score: {:.3}",
                question,
                prediction.answer,
                prediction.source_doc,
                prediction.source_chunk_id,
                prediction.retrieval_score
            );
        }
    } else {
        println!("No chunks found. Add .docx files to ./data");
    }

    Ok(())
}
