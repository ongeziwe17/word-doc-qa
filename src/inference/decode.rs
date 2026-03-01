use std::collections::HashSet;

pub fn best_span_text(context: &str, question: &str, max_words: usize) -> String {
    let query_tokens = tokenize(question);

    let mut best_sentence = String::new();
    let mut best_score = f64::MIN;

    for sentence in split_sentences(context) {
        let sent_tokens = tokenize(sentence);
        let overlap = query_tokens.intersection(&sent_tokens).count() as f64;
        let len_penalty = (sent_tokens.len().max(1) as f64).sqrt();
        let score = overlap / len_penalty;

        if score > best_score {
            best_score = score;
            best_sentence = sentence.to_string();
        }
    }

    if best_sentence.is_empty() {
        best_sentence = context.to_string();
    }

    best_sentence
        .split_whitespace()
        .take(max_words)
        .collect::<Vec<_>>()
        .join(" ")
}

fn split_sentences(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0usize;

    for (idx, ch) in text.char_indices() {
        if matches!(ch, '.' | '!' | '?') {
            let end = idx + ch.len_utf8();
            let sentence = text[start..end].trim();
            if !sentence.is_empty() {
                out.push(sentence);
            }
            start = end;
        }
    }

    let tail = text[start..].trim();
    if !tail.is_empty() {
        out.push(tail);
    }

    out
}

fn tokenize(text: &str) -> HashSet<String> {
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
    use super::best_span_text;

    #[test]
    fn picks_sentence_with_highest_overlap() {
        let context = "Rust uses ownership for memory safety. Soccer has 11 players.";
        let answer = best_span_text(context, "How does rust ensure memory safety?", 12);
        assert!(answer.to_lowercase().contains("ownership"));
    }
}
