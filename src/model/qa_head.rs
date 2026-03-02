use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{Tensor, backend::Backend};

#[derive(Module, Debug)]
pub struct QaHead<B: Backend> {
    start_linear: Linear<B>,
    end_linear: Linear<B>,
}

impl<B: Backend> QaHead<B> {
    pub fn new(d_model: usize, device: &B::Device) -> Self {
        Self {
            start_linear: LinearConfig::new(d_model, 1).init(device),
            end_linear: LinearConfig::new(d_model, 1).init(device),
        }
    }

    pub fn forward(&self, hidden_states: Tensor<B, 3>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let [batch_size, seq_len, _] = hidden_states.dims();

        let start_logits = self
            .start_linear
            .forward(hidden_states.clone())
            .reshape([batch_size, seq_len]);
        let end_logits = self
            .end_linear
            .forward(hidden_states)
            .reshape([batch_size, seq_len]);

        (start_logits, end_logits)
    }
}
