use crate::data::chunker::chunk_text;
use crate::data::cleaner::clean_text;
use crate::data::docx_loader::load_docx_text;
use anyhow::{Context, Result};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Chunk {
    pub doc_id: String,
    pub chunk_id: usize,
    pub text: String,
}

pub fn load_corpus_from_dir(data_dir: &str, max_chars: usize) -> Result<Vec<Chunk>> {
    let dir = Path::new(data_dir);
    if !dir.exists() {
        return Ok(Vec::new());
    }

    let mut chunks = Vec::new();

    for entry in fs::read_dir(dir).with_context(|| format!("failed to read {data_dir}"))? {
        let entry = entry?;
        let path = entry.path();

        if !is_docx(&path) {
            continue;
        }

        let doc_id = path
            .file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or("unknown")
            .to_string();

        let raw_text = load_docx_text(&path)?;
        let cleaned = clean_text(&raw_text);

        for (chunk_id, text) in chunk_text(&cleaned, max_chars).into_iter().enumerate() {
            chunks.push(Chunk {
                doc_id: doc_id.clone(),
                chunk_id,
                text,
            });
        }
    }

    Ok(chunks)
}

fn is_docx(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("docx"))
        .unwrap_or(false)
}
