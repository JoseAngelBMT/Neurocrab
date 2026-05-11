use crate::autograd::tape::Tape;
use crate::autograd::variable::Variable;
use crate::nn::module::Module;
use crate::nn::parameter::Parameter;

pub struct ReLU;

impl Module for ReLU {
    fn forward(&self, input: &Variable<f32>, tape: &mut Tape<f32>) -> Variable<f32> {
        input.relu(tape)
    }

    fn parameters(&self) -> Vec<&Parameter<f32>> {
        vec![]
    }
}