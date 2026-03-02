# word-doc-qa

A Rust command-line prototype for Question Answering over `.docx` files.

It provides two commands:
- `train`: ingest DOCX files, build tokenizer + weak-supervised samples, and train/save checkpoints.
- `ask`: load checkpoints and answer questions over retrieved chunks.

## Next milestone focus

Current code has end-to-end train/ask functionality with tokenizer/model/checkpoint integration.

Recommended next technical milestone is:
1. **Backend-native training path** (autograd + backend optimizer integration),
2. **Keep/extend best-checkpoint default loading**,
3. **Refine model quality and reproducibility metrics**.

## Quick start

See [INSTRUCTIONS.md](./INSTRUCTIONS.md) for full environment, installation, and run commands.

### Typical run commands

```bash
cargo run --offline -- train --data-dir data/
cargo run --offline -- ask --question "What month and date will the 2024 End of year Graduation Ceremony be held?" --data-dir data/
```

> Note: `--offline` only works if dependencies are already cached locally.