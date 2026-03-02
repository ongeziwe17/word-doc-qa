pub mod args;
pub mod commands;

pub use args::{Cli, Commands};
pub use commands::{run_ask, run_train};
