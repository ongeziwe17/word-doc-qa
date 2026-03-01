pub fn clean_text(text: &str) -> String {
    let normalized = text.replace("\r\n", "\n").replace('\r', "\n");

    let mut cleaned_lines = Vec::new();
    let mut previous_blank = false;

    for line in normalized.lines() {
        let compact = line.split_whitespace().collect::<Vec<_>>().join(" ");

        if compact.is_empty() {
            if !previous_blank {
                cleaned_lines.push(String::new());
            }
            previous_blank = true;
        } else {
            cleaned_lines.push(compact);
            previous_blank = false;
        }
    }

    cleaned_lines.join("\n").trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::clean_text;

    #[test]
    fn normalizes_whitespace_and_blank_lines() {
        let raw = "  Hello   world\r\n\r\n\r\nThis\t is   a test.  \n   \nAnother line  ";
        let cleaned = clean_text(raw);

        assert_eq!(cleaned, "Hello world\n\nThis is a test.\n\nAnother line");
    }

    #[test]
    fn trims_outer_whitespace() {
        let raw = "\n\n   Keep me   \n\n";
        let cleaned = clean_text(raw);
        assert_eq!(cleaned, "Keep me");
    }
}
