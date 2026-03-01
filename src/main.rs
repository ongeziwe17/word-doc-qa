mod cli;
mod data;
mod inference;
mod model;
mod tokenization;
mod train;

use anyhow::Result;
use clap::Parser;
use cli::{Cli, Commands, run_ask, run_train};

fn main() -> Result<()> {
    env_logger::init();

    let cli = Cli::parse();
    match cli.command {
        Commands::Train(args) => run_train(args),
        Commands::Ask(args) => run_ask(args),
    }
}
