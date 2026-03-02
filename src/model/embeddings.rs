use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig};
use burn::tensor::{Int, Tensor, backend::Backend};

#[derive(Module, Debug)]
pub struct TokenEmbeddings<B: Backend> {
    token_embedding: Embedding<B>,
}

impl<B: Backend> TokenEmbeddings<B> {
    pub fn new(vocab_size: usize, d_model: usize, device: &B::Device) -> Self {
        Self {
            token_embedding: EmbeddingConfig::new(vocab_size, d_model).init(device),
        }
    }

    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.token_embedding.forward(input_ids)
    }
}
