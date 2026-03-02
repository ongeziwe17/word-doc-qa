use clap::{Args, Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "word-doc-qa", about = "Word Document Q&A System")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    Train(TrainArgs),
    Ask(AskArgs),
}

#[derive(Debug, Args)]
pub struct TrainArgs {
    #[arg(long, default_value = "./data")]
    pub data_dir: String,
    #[arg(long, default_value_t = 1000)]
    pub max_chars: usize,
    #[arg(long, default_value_t = 3)]
    pub epochs: usize,
    #[arg(long, default_value_t = 4)]
    pub batch_size: usize,
    #[arg(long, default_value_t = 1e-3)]
    pub lr: f64,
    #[arg(long, default_value = "./checkpoints")]
    pub checkpoint_dir: String,
}

#[derive(Debug, Args)]
pub struct AskArgs {
    #[arg(long)]
    pub question: String,
    #[arg(long, default_value = "./data")]
    pub data_dir: String,
    #[arg(long, default_value_t = 1000)]
    pub max_chars: usize,
    #[arg(long, default_value_t = 3)]
    pub top_k: usize,
    #[arg(long, default_value_t = 24, alias = "max-answer-words")]
    pub max_answer_len: usize,
    #[arg(long, default_value = "./checkpoints")]
    pub checkpoint_dir: String,
    #[arg(long)]
    pub checkpoint_path: Option<String>,
}
