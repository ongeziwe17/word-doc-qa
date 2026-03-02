use burn::module::Module;
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::activation::relu;
use burn::tensor::{Tensor, backend::Backend};

#[derive(Module, Debug)]
pub struct EncoderBlock<B: Backend> {
    norm: LayerNorm<B>,
    ff1: Linear<B>,
    ff2: Linear<B>,
}

impl<B: Backend> EncoderBlock<B> {
    pub fn new(d_model: usize, ff_dim: usize, device: &B::Device) -> Self {
        Self {
            norm: LayerNormConfig::new(d_model).init(device),
            ff1: LinearConfig::new(d_model, ff_dim).init(device),
            ff2: LinearConfig::new(ff_dim, d_model).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = x.clone();
        let h = self.norm.forward(x);
        let h = self.ff1.forward(h);
        let h = relu(h);
        let h = self.ff2.forward(h);
        residual + h
    }
}

#[derive(Module, Debug)]
pub struct TransformerEncoder<B: Backend> {
    layers: Vec<EncoderBlock<B>>,
}

impl<B: Backend> TransformerEncoder<B> {
    pub fn new(num_layers: usize, d_model: usize, ff_dim: usize, device: &B::Device) -> Self {
        let enforced_layers = usize::max(6, num_layers);
        let layers = (0..enforced_layers)
            .map(|_| EncoderBlock::new(d_model, ff_dim, device))
            .collect();

        Self { layers }
    }

    pub fn forward(&self, mut x: Tensor<B, 3>) -> Tensor<B, 3> {
        for layer in &self.layers {
            x = layer.forward(x);
        }
        x
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}
