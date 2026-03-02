# INSTRUCTIONS

This document explains what you need to run `word-doc-qa`, including environment setup and common commands.

## 1) Environment requirements

### OS
- Linux, macOS, or Windows (with WSL recommended for parity with Linux shell commands).

### Toolchain
- Rust toolchain (stable) installed via `rustup`.
- `cargo` available in `PATH`.

### Runtime/project dependencies
- The project is a Rust binary using crates like `clap`, `serde`, `anyhow`, `docx-rs`, and `burn`.
- If running with `--offline`, these dependencies must already exist in your local Cargo cache.

## 2) Install Rust (if needed)

### Linux/macOS
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
rustc --version
cargo --version
```

### Windows
- Install from: https://rustup.rs/
- Re-open terminal and verify:
```powershell
rustc --version
cargo --version
```

## 3) Get the project ready

```bash
git clone <your-repo-url>
cd word-doc-qa
```

## 4) Data layout expected

Place `.docx` files under a data directory (default `./data`). Example:

```text
word-doc-qa/
  data/
    handbook.docx
    schedule.docx
```

## 5) Build / test commands

### Online mode (recommended first-time)
```bash
cargo build
cargo test
```

### Offline mode (only if dependencies are already cached)
```bash
cargo build --offline
cargo test --offline
```

If offline fails with dependency resolution errors, run without `--offline` once to populate cache.

## 6) Run commands

## Train

User-provided example command:
```bash
cargo run --offline -- train --data-dir data/
```

Equivalent online command:
```bash
cargo run -- train --data-dir data/
```

Common train flags:
- `--data-dir <path>` (default `./data`)
- `--max-chars <n>`
- `--epochs <n>`
- `--batch-size <n>`
- `--lr <float>`
- `--checkpoint-dir <path>`

## Ask

User-provided example command:
```bash
cargo run --offline -- ask --question "What month and date will the 2024 End of year Graduation Ceremony be held?" --data-dir data/
```

Equivalent online command:
```bash
cargo run -- ask --question "What month and date will the 2024 End of year Graduation Ceremony be held?" --data-dir data/
```

Common ask flags:
- `--question <text>`
- `--data-dir <path>`
- `--top-k <n>`
- `--max-answer-len <n>`
- `--checkpoint-dir <path>`
- `--checkpoint-path <file>`

Behavior note:
- If `--checkpoint-path` is set, it is used directly.
- Otherwise, the app prefers `best` checkpoint and falls back to latest checkpoint.

## 7) Troubleshooting

### `--offline` fails with missing crate/index
Cause: local Cargo cache is incomplete.
Fix: run once online:
```bash
cargo build
```
Then retry offline.

### No chunks found
Cause: no `.docx` files in `--data-dir`.
Fix: add Word documents to the configured directory.

### Network restriction errors (crates.io 403 / tunnel)
Cause: restricted environment.
Fix: run where crates.io is reachable, or pre-cache dependencies and use `--offline`.

## 8) Suggested submission run checklist

```bash
cargo fmt
cargo run -- train --data-dir data/
cargo run -- ask --question "Your question here" --data-dir data/
```

If your environment requires no internet access and dependencies are already cached, use `--offline` variants.
