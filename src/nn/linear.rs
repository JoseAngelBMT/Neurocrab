use crate::autograd::tape::Tape;
use crate::autograd::variable::Variable;
use crate::nn::parameter::Parameter;
use crate::tensor::Tensor;
use rand::distributions::Distribution;
use rand::distributions::Uniform;
use rand::thread_rng;
use crate::nn::module::Module;

pub struct Linear {
    pub weight: Parameter<f32>,
    pub bias: Parameter<f32>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let limit = (6.0f32 / (in_features + out_features) as f32).sqrt(); // Xavier initialization
        let dist = Uniform::new(-limit, limit);
        let data: Vec<f32> = (0..(in_features * out_features))
            .map(|_| dist.sample(&mut thread_rng()))
            .collect();

        let weight_tensor = Tensor::from_vec(data, vec![out_features, in_features]).unwrap();
        let bias_tensor: Tensor<f32> = Tensor::zeros(vec![1, out_features]);

        Self {
            weight: Parameter::new(weight_tensor),
            bias: Parameter::new(bias_tensor),
        }
    }

}

impl Module for Linear {
    fn forward(&self, input: &Variable<f32>, tape: &mut Tape<f32>) -> Variable<f32> {
        let weight_t = self.weight.value.borrow().clone();
        let w_var = tape.variable(weight_t);
        self.weight.id.set(Some(w_var.id));

        let bias_t = self.bias.value.borrow().clone();
        let b_var = tape.variable(bias_t);
        self.bias.id.set(Some(b_var.id));

        input.matmul(&w_var.transpose(tape), tape).add(&b_var, tape)

    }

    fn parameters(&self) -> Vec<&Parameter<f32>> {
        vec![&self.weight, &self.bias]
    }
}
