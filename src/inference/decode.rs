pub fn best_span_text(context: &str, max_words: usize) -> String {
    context
        .split_whitespace()
        .take(max_words)
        .collect::<Vec<_>>()
        .join(" ")
}
