mod data;

use anyhow::Result;
use data::load_corpus_from_dir;

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
    } else {
        println!("No chunks found. Add .docx files to ./data");
    }

    Ok(())
}
