use anyhow::Result;

use crate::cli::args::{AskArgs, TrainArgs};
use crate::data::load_corpus_from_dir;
use crate::inference::answer_question;
use crate::model::QaModelConfig;
use crate::tokenization::{build_tokenizer_from_texts, build_weak_supervised_samples};
use crate::train::{
    TrainConfig, evaluate_on_samples, load_checkpoint, load_latest_checkpoint,
    train_weak_supervised,
};

pub fn run_train(args: TrainArgs) -> Result<()> {
    let chunks = load_corpus_from_dir(&args.data_dir, args.max_chars)?;
    println!("Loaded {} chunks from {}", chunks.len(), args.data_dir);

    if chunks.is_empty() {
        println!("No chunks found. Add .docx files to {}", args.data_dir);
        return Ok(());
    }

    let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let tokenizer = build_tokenizer_from_texts(&texts, 8_000)?;
    let qa_samples = build_weak_supervised_samples(&texts, &tokenizer, 64)?;

    let model_config = QaModelConfig::default();
    let train_config = TrainConfig {
        epochs: args.epochs,
        batch_size: args.batch_size,
        learning_rate: args.lr,
        checkpoint_dir: args.checkpoint_dir,
        ..TrainConfig::default()
    };

    let summary = train_weak_supervised(&qa_samples, &model_config, &train_config)?;
    println!(
        "Training complete => epochs: {}, final_avg_loss: {:.4}",
        summary.epochs_completed, summary.final_avg_loss
    );
    if let Some(last_epoch) = summary.history.latest() {
        println!(
            "Last epoch stats => epoch: {}, samples_seen: {}, optimizer_steps: {}",
            last_epoch.epoch, last_epoch.samples_seen, summary.optimizer_state.step
        );
    }

    if let Some(last_ckpt) = load_latest_checkpoint(&train_config.checkpoint_dir)? {
        println!(
            "Latest checkpoint => epoch: {}, avg_loss: {:.4}, opt_step: {}",
            last_ckpt.epoch, last_ckpt.avg_loss, last_ckpt.optimizer_state.step
        );
    }

    let eval = evaluate_on_samples(&qa_samples, Some(&summary.model));
    println!(
        "Eval => EM: {:.3}, F1: {:.3}, samples: {}",
        eval.exact_match, eval.token_f1, eval.total
    );

    Ok(())
}

pub fn run_ask(args: AskArgs) -> Result<()> {
    let chunks = load_corpus_from_dir(&args.data_dir, args.max_chars)?;
    println!("Loaded {} chunks from {}", chunks.len(), args.data_dir);

    if chunks.is_empty() {
        println!("No chunks found. Add .docx files to {}", args.data_dir);
        return Ok(());
    }

    let checkpoint = if let Some(path) = args.checkpoint_path.as_ref() {
        Some(load_checkpoint(path)?)
    } else {
        load_latest_checkpoint(&args.checkpoint_dir)?
    };

    let model = checkpoint.as_ref().map(|c| &c.model);

    match answer_question(
        &chunks,
        &args.question,
        args.top_k,
        args.max_answer_len,
        model,
    ) {
        Some(pred) => {
            println!(
                "Answer => {}\nSource => {}#{}\nRetrieval score => {:.3}\nSpan score => {:.3}",
                pred.answer,
                pred.source_doc,
                pred.source_chunk_id,
                pred.retrieval_score,
                pred.span_score
            );
        }
        None => {
            println!("No answer could be produced.");
        }
    }

    Ok(())
}
