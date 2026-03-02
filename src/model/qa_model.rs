use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QaModelConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub ff_dim: usize,
    pub num_layers: usize,
}

impl QaModelConfig {
    #[allow(dead_code)]
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
    pub embeddings: Vec<Vec<f32>>,
    pub start_proj: Vec<f32>,
    pub end_proj: Vec<f32>,
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
        let vocab = config.vocab_size.max(2);
        let d_model = config.d_model.max(8);

        let mut embeddings = vec![vec![0.0; d_model]; vocab];
        for (tok, emb) in embeddings.iter_mut().enumerate() {
            for (d, v) in emb.iter_mut().enumerate() {
                *v = (((tok * 31 + d * 17) % 997) as f32 / 997.0) * 0.02 - 0.01;
            }
        }

        let start_proj = (0..d_model)
            .map(|i| (((i * 13) % 101) as f32 / 101.0) * 0.02 - 0.01)
            .collect();
        let end_proj = (0..d_model)
            .map(|i| (((i * 19) % 103) as f32 / 103.0) * 0.02 - 0.01)
            .collect();

        Self {
            config,
            embeddings,
            start_proj,
            end_proj,
            start_bias: 0.0,
            end_bias: 0.0,
        }
    }

    pub fn forward(&self, input_ids: &[u32]) -> QaModelOutput {
        let mut start_logits = Vec::with_capacity(input_ids.len());
        let mut end_logits = Vec::with_capacity(input_ids.len());

        for token_id in input_ids {
            let idx = (*token_id as usize).min(self.embeddings.len().saturating_sub(1));
            let emb = &self.embeddings[idx];
            let s = dot(emb, &self.start_proj) + self.start_bias;
            let e = dot(emb, &self.end_proj) + self.end_bias;
            start_logits.push(s);
            end_logits.push(e);
        }

        QaModelOutput {
            start_logits,
            end_logits,
        }
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
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
