mod data;
mod tokenization;

use anyhow::Result;
use data::load_corpus_from_dir;
use tokenization::{build_tokenizer_from_texts, pad_and_create_batch};

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
    } else {
        println!("No chunks found. Add .docx files to ./data");
    }

    Ok(())
}
