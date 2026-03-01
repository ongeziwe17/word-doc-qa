use burn::module::Module;
use burn::tensor::{Int, Tensor, backend::Backend};
use serde::{Deserialize, Serialize};

use crate::model::embeddings::TokenEmbeddings;
use crate::model::qa_head::QaHead;
use crate::model::transformer::TransformerEncoder;

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

#[derive(Debug)]
pub struct QaModelOutput<B: Backend> {
    pub start_logits: Tensor<B, 2>,
    pub end_logits: Tensor<B, 2>,
}

#[derive(Module, Debug)]
pub struct QaModel<B: Backend> {
    embeddings: TokenEmbeddings<B>,
    encoder: TransformerEncoder<B>,
    qa_head: QaHead<B>,
}

impl<B: Backend> QaModel<B> {
    pub fn new(config: QaModelConfig, device: &B::Device) -> Self {
        Self {
            embeddings: TokenEmbeddings::new(config.vocab_size, config.d_model, device),
            encoder: TransformerEncoder::new(
                config.effective_num_layers(),
                config.d_model,
                config.ff_dim,
                device,
            ),
            qa_head: QaHead::new(config.d_model, device),
        }
    }

    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> QaModelOutput<B> {
        let embedded = self.embeddings.forward(input_ids);
        let encoded = self.encoder.forward(embedded);
        let (start_logits, end_logits) = self.qa_head.forward(encoded);

        QaModelOutput {
            start_logits,
            end_logits,
        }
    }

    pub fn num_layers(&self) -> usize {
        self.encoder.num_layers()
    }
}

#[cfg(test)]
mod tests {
    use super::QaModelConfig;

    #[test]
    fn enforces_minimum_encoder_depth() {
        let config = QaModelConfig {
            num_layers: 2,
            ..Default::default()
        };

        assert_eq!(config.effective_num_layers(), 6);
    }
}
