use std::collections::HashSet;

pub fn compute_token_logits(
    context: &str,
    question: &str,
    start_prior: Option<&[f32]>,
    end_prior: Option<&[f32]>,
) -> (Vec<f32>, Vec<f32>, Vec<String>) {
    let query = tokenize(question);
    let tokens: Vec<String> = context
        .split_whitespace()
        .map(|t| {
            t.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|t| !t.is_empty())
        .collect();

    let mut start_logits = vec![0.0; tokens.len()];
    let mut end_logits = vec![0.0; tokens.len()];

    for (idx, tok) in tokens.iter().enumerate() {
        let overlap_score = if query.contains(tok) { 1.0 } else { 0.0 };
        start_logits[idx] = overlap_score;
        end_logits[idx] = overlap_score;

        if let Some(sp) = start_prior {
            if idx < sp.len() {
                start_logits[idx] += sp[idx];
            }
        }
        if let Some(ep) = end_prior {
            if idx < ep.len() {
                end_logits[idx] += ep[idx];
            }
        }
    }

    (start_logits, end_logits, tokens)
}

pub fn best_span_from_logits(
    start_logits: &[f32],
    end_logits: &[f32],
    max_answer_len: usize,
) -> Option<(usize, usize)> {
    if start_logits.is_empty() || end_logits.is_empty() {
        return None;
    }

    let mut best = None;
    let mut best_score = f32::NEG_INFINITY;
    let max_len = max_answer_len.max(1);

    for s in 0..start_logits.len() {
        let end_limit = usize::min(end_logits.len() - 1, s + max_len - 1);
        for e in s..=end_limit {
            let score = start_logits[s] + end_logits[e];
            if score > best_score {
                best_score = score;
                best = Some((s, e));
            }
        }
    }

    best
}

pub fn span_text(tokens: &[String], start: usize, end: usize, max_words: usize) -> String {
    if tokens.is_empty() {
        return String::new();
    }

    let end = usize::min(end, tokens.len() - 1);
    let start = usize::min(start, end);
    tokens[start..=end]
        .iter()
        .take(max_words.max(1))
        .cloned()
        .collect::<Vec<_>>()
        .join(" ")
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
    use super::{best_span_from_logits, compute_token_logits, span_text};

    #[test]
    fn decodes_span_from_logits() {
        let (s, e, toks) = compute_token_logits(
            "Rust uses ownership for memory safety.",
            "How does rust memory work?",
            None,
            None,
        );
        let (bs, be) = best_span_from_logits(&s, &e, 4).expect("span");
        let answer = span_text(&toks, bs, be, 4);
        assert!(!answer.is_empty());
    }
}
