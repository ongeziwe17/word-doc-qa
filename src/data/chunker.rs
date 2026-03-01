pub fn chunk_text(text: &str, max_chars: usize) -> Vec<String> {
    if max_chars == 0 {
        return Vec::new();
    }

    let mut chunks = Vec::new();
    let mut current = String::new();

    for paragraph in text.split("\n\n") {
        let paragraph = paragraph.trim();
        if paragraph.is_empty() {
            continue;
        }

        if paragraph.len() > max_chars {
            if !current.is_empty() {
                chunks.push(current);
                current = String::new();
            }
            chunks.extend(split_long_text(paragraph, max_chars));
            continue;
        }

        let separator = if current.is_empty() { "" } else { "\n\n" };
        if current.len() + separator.len() + paragraph.len() <= max_chars {
            current.push_str(separator);
            current.push_str(paragraph);
        } else {
            if !current.is_empty() {
                chunks.push(current);
            }
            current = paragraph.to_string();
        }
    }

    if !current.is_empty() {
        chunks.push(current);
    }

    chunks
}

fn split_long_text(text: &str, max_chars: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();

    for word in text.split_whitespace() {
        if word.len() > max_chars {
            if !current.is_empty() {
                out.push(current);
                current = String::new();
            }
            let mut start = 0;
            while start < word.len() {
                let end = usize::min(start + max_chars, word.len());
                out.push(word[start..end].to_string());
                start = end;
            }
            continue;
        }

        let next_len = if current.is_empty() {
            word.len()
        } else {
            current.len() + 1 + word.len()
        };

        if next_len <= max_chars {
            if !current.is_empty() {
                current.push(' ');
            }
            current.push_str(word);
        } else {
            out.push(current);
            current = word.to_string();
        }
    }

    if !current.is_empty() {
        out.push(current);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::chunk_text;

    #[test]
    fn splits_into_multiple_chunks() {
        let text =
            "Paragraph one with enough words.\n\nParagraph two continues.\n\nParagraph three ends.";
        let chunks = chunk_text(text, 45);

        assert_eq!(chunks.len(), 3);
        assert!(chunks.iter().all(|c| c.len() <= 45));
    }

    #[test]
    fn handles_overlong_words() {
        let text = "supercalifragilisticexpialidocious";
        let chunks = chunk_text(text, 10);

        assert_eq!(
            chunks,
            vec!["supercalif", "ragilistic", "expialidoc", "ious"]
        );
    }

    #[test]
    fn returns_empty_for_zero_max_chars() {
        assert!(chunk_text("hello", 0).is_empty());
    }
}
