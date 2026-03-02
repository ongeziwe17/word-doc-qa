use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::model::{QaModel, QaModelConfig};
use crate::train::config::TrainConfig;
use crate::train::trainer::OptimizerState;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingCheckpoint {
    pub epoch: usize,
    pub avg_loss: f64,
    pub model_config: QaModelConfig,
    pub train_config: TrainConfig,
    pub model: QaModel,
    pub optimizer_state: OptimizerState,
    pub tokenizer_vocab: HashMap<String, u32>,
}

pub fn save_checkpoint(dir: &str, ckpt: &TrainingCheckpoint) -> Result<PathBuf> {
    fs::create_dir_all(dir).with_context(|| format!("failed to create checkpoint dir: {dir}"))?;
    let path = Path::new(dir).join(format!("epoch_{:03}.json", ckpt.epoch));
    write_checkpoint_file(&path, ckpt)?;
    Ok(path)
}

pub fn save_best_checkpoint(dir: &str, ckpt: &TrainingCheckpoint) -> Result<PathBuf> {
    fs::create_dir_all(dir).with_context(|| format!("failed to create checkpoint dir: {dir}"))?;
    let path = Path::new(dir).join("best.json");
    write_checkpoint_file(&path, ckpt)?;
    Ok(path)
}

fn write_checkpoint_file(path: &Path, ckpt: &TrainingCheckpoint) -> Result<()> {
    let json = serde_json::to_string_pretty(ckpt)?;
    fs::write(path, json).with_context(|| format!("failed to write checkpoint: {}", path.display()))
}

pub fn load_checkpoint(path: &str) -> Result<TrainingCheckpoint> {
    let checkpoint_path = Path::new(path);
    let content = fs::read_to_string(checkpoint_path)
        .with_context(|| format!("failed reading checkpoint: {}", checkpoint_path.display()))?;
    serde_json::from_str::<TrainingCheckpoint>(&content)
        .with_context(|| format!("failed parsing checkpoint: {}", checkpoint_path.display()))
}

pub fn load_best_checkpoint(dir: &str) -> Result<Option<TrainingCheckpoint>> {
    let path = Path::new(dir).join("best.json");
    if !path.exists() {
        return Ok(None);
    }
    let ckpt = load_checkpoint(
        path.to_str()
            .ok_or_else(|| anyhow::anyhow!("non-utf8 checkpoint path"))?,
    )?;
    Ok(Some(ckpt))
}

pub fn load_latest_checkpoint(dir: &str) -> Result<Option<TrainingCheckpoint>> {
    let base = Path::new(dir);
    if !base.exists() {
        return Ok(None);
    }

    let mut files: Vec<PathBuf> = fs::read_dir(base)?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension().and_then(|e| e.to_str()) == Some("json")
                && p.file_name().and_then(|n| n.to_str()) != Some("best.json")
        })
        .collect();

    files.sort();
    let Some(last) = files.last() else {
        return Ok(None);
    };

    let ckpt = load_checkpoint(
        last.to_str()
            .ok_or_else(|| anyhow::anyhow!("non-utf8 checkpoint path"))?,
    )?;
    Ok(Some(ckpt))
}

#[cfg(test)]
mod tests {
    use super::{
        TrainingCheckpoint, load_best_checkpoint, load_checkpoint, load_latest_checkpoint,
        save_best_checkpoint, save_checkpoint,
    };
    use crate::model::{QaModel, QaModelConfig};
    use crate::train::config::TrainConfig;
    use crate::train::trainer::OptimizerState;
    use std::collections::HashMap;

    #[test]
    fn saves_and_loads_checkpoint() {
        let dir = "./target/tmp-checkpoints";
        let ckpt = TrainingCheckpoint {
            epoch: 1,
            avg_loss: 0.5,
            model_config: QaModelConfig::default(),
            train_config: TrainConfig::default(),
            model: QaModel::new(QaModelConfig::default()),
            optimizer_state: OptimizerState::new(0.001, 128),
            tokenizer_vocab: HashMap::new(),
        };

        let path = save_checkpoint(dir, &ckpt).expect("save checkpoint");
        let _ = save_best_checkpoint(dir, &ckpt).expect("save best");
        let loaded = load_latest_checkpoint(dir).expect("load").expect("exists");
        let best = load_best_checkpoint(dir)
            .expect("load best")
            .expect("exists");
        let direct = load_checkpoint(path.to_str().expect("utf8 path")).expect("direct load");

        assert_eq!(loaded.epoch, 1);
        assert_eq!(best.epoch, 1);
        assert_eq!(direct.epoch, loaded.epoch);
    }
}
