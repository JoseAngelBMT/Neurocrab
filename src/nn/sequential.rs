use crate::autograd::tape::Tape;
use crate::autograd::variable::Variable;
use crate::nn::module::Module;
use crate::nn::parameter::Parameter;

pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new(layers: Vec<Box<dyn Module>>) -> Self {
        Self { layers }
    }
}

impl Module for Sequential {
    fn forward(&self, input: &Variable<f32>, tape: &mut Tape<f32>) -> Variable<f32> {
        let mut x = self.layers[0].forward(input, tape);
        for layer in &self.layers[1..] {
            x = layer.forward(&x, tape);
        }
        x
    }

    fn parameters(&self) -> Vec<&Parameter<f32>> {
        self.layers
            .iter()
            .flat_map(|x| x.parameters())
            .collect()
    }
}
