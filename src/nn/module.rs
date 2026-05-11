use crate::autograd::tape::Tape;
use crate::autograd::variable::Variable;
use crate::nn::parameter::Parameter;

pub trait Module {
    fn forward(&self, input: &Variable<f32>, tape: &mut Tape<f32>) -> Variable<f32>;
    fn parameters(&self) -> Vec<&Parameter<f32>>;
}