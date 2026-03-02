use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QaModelConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub ff_dim: usize,
    pub num_layers: usize,
}

impl QaModelConfig {
    pub fn effective_num_layers(&self) -> usize {
        usize::max(6, self.num_layers)
    }
}

impl Default for QaModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 8_000,
            d_model: 128,
            ff_dim: 256,
            num_layers: 6,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QaModel {
    pub config: QaModelConfig,
    pub start_weight: f32,
    pub end_weight: f32,
    pub start_bias: f32,
    pub end_bias: f32,
}

#[derive(Debug, Clone)]
pub struct QaModelOutput {
    pub start_logits: Vec<f32>,
    pub end_logits: Vec<f32>,
}

impl QaModel {
    pub fn new(config: QaModelConfig) -> Self {
        Self {
            config,
            start_weight: 0.01,
            end_weight: -0.01,
            start_bias: 0.0,
            end_bias: 0.0,
        }
    }

    pub fn forward(&self, input_ids: &[u32]) -> QaModelOutput {
        let mut start_logits = Vec::with_capacity(input_ids.len());
        let mut end_logits = Vec::with_capacity(input_ids.len());

        for token_id in input_ids {
            let x = *token_id as f32 / self.config.vocab_size.max(1) as f32;
            start_logits.push(self.start_weight * x + self.start_bias);
            end_logits.push(self.end_weight * x + self.end_bias);
        }

        QaModelOutput {
            start_logits,
            end_logits,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{QaModel, QaModelConfig};

    #[test]
    fn enforces_minimum_encoder_depth() {
        let config = QaModelConfig {
            num_layers: 2,
            ..Default::default()
        };

        assert_eq!(config.effective_num_layers(), 6);
    }

    #[test]
    fn forward_outputs_match_input_length() {
        let model = QaModel::new(QaModelConfig::default());
        let out = model.forward(&[1, 2, 3]);
        assert_eq!(out.start_logits.len(), 3);
        assert_eq!(out.end_logits.len(), 3);
    }
}
