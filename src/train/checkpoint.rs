use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

use crate::model::QaModelConfig;
use crate::train::config::TrainConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingCheckpoint {
    pub epoch: usize,
    pub avg_loss: f64,
    pub model_config: QaModelConfig,
    pub train_config: TrainConfig,
}

pub fn save_checkpoint(dir: &str, ckpt: &TrainingCheckpoint) -> Result<PathBuf> {
    fs::create_dir_all(dir).with_context(|| format!("failed to create checkpoint dir: {dir}"))?;
    let path = Path::new(dir).join(format!("epoch_{:03}.json", ckpt.epoch));
    let json = serde_json::to_string_pretty(ckpt)?;
    fs::write(&path, json)
        .with_context(|| format!("failed to write checkpoint: {}", path.display()))?;
    Ok(path)
}

pub fn load_latest_checkpoint(dir: &str) -> Result<Option<TrainingCheckpoint>> {
    let base = Path::new(dir);
    if !base.exists() {
        return Ok(None);
    }

    let mut files: Vec<PathBuf> = fs::read_dir(base)?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("json"))
        .collect();

    files.sort();
    let Some(last) = files.last() else {
        return Ok(None);
    };

    let content = fs::read_to_string(last)
        .with_context(|| format!("failed reading checkpoint: {}", last.display()))?;
    let ckpt = serde_json::from_str::<TrainingCheckpoint>(&content)
        .with_context(|| format!("failed parsing checkpoint: {}", last.display()))?;
    Ok(Some(ckpt))
}

#[cfg(test)]
mod tests {
    use super::{TrainingCheckpoint, load_latest_checkpoint, save_checkpoint};
    use crate::model::QaModelConfig;
    use crate::train::config::TrainConfig;

    #[test]
    fn saves_and_loads_checkpoint() {
        let dir = "./target/tmp-checkpoints";
        let ckpt = TrainingCheckpoint {
            epoch: 1,
            avg_loss: 0.5,
            model_config: QaModelConfig::default(),
            train_config: TrainConfig::default(),
        };

        let _ = save_checkpoint(dir, &ckpt).expect("save checkpoint");
        let loaded = load_latest_checkpoint(dir).expect("load").expect("exists");

        assert_eq!(loaded.epoch, 1);
        assert_eq!(loaded.avg_loss, 0.5);
    }
}
