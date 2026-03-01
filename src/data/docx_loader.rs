use anyhow::{Context, Result};
use docx_rs::{DocumentChild, ParagraphChild, RunChild, read_docx};
use std::fs;
use std::path::Path;

pub fn load_docx_text(path: &Path) -> Result<String> {
    let bytes =
        fs::read(path).with_context(|| format!("failed to read DOCX file: {}", path.display()))?;

    let docx = read_docx(&bytes)
        .with_context(|| format!("failed to parse DOCX content: {}", path.display()))?;

    let mut paragraphs = Vec::new();

    for child in &docx.document.children {
        if let DocumentChild::Paragraph(paragraph) = child {
            let mut paragraph_text = String::new();
            for run_child in &paragraph.children {
                if let ParagraphChild::Run(run) = run_child {
                    for part in &run.children {
                        match part {
                            RunChild::Text(text) => paragraph_text.push_str(&text.text),
                            RunChild::Tab(_) => paragraph_text.push('\t'),
                            RunChild::Break(_) => paragraph_text.push('\n'),
                            _ => {}
                        }
                    }
                }
            }

            if !paragraph_text.trim().is_empty() {
                paragraphs.push(paragraph_text);
            }
        }
    }

    Ok(paragraphs.join("\n"))
}
