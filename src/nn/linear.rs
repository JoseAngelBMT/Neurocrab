use crate::autograd::tape::Tape;
use crate::autograd::variable::Variable;
use crate::tensor::Tensor;
use rand::distributions::Uniform;
use rand::distributions::Distribution;
use rand::thread_rng;

pub struct Linear {
    pub weight: Variable<f32>,
    pub bias: Variable<f32>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, tape: &mut Tape<f32>) -> Self {
        let limit = (6.0f32 / (in_features + out_features) as f32).sqrt(); // Xavier initialization
        let dist = Uniform::new(-limit, limit);
        let data: Vec<f32> = (0..(in_features * out_features))
            .map(|_| dist.sample(&mut thread_rng()))
            .collect();

        let weight_tensor = Tensor::from_vec(data, vec![out_features, in_features]).unwrap();
        let bias_tensor = Tensor::zeros(vec![out_features]);

        let weight = tape.variable(weight_tensor);
        let bias = tape.variable(bias_tensor);
        Self { weight, bias }
    }

    pub fn forward(&self, input: &Variable<f32>, tape: &mut Tape<f32>) -> Variable<f32> {
        let weight_t = self.weight.transpose(tape);
        input.matmul(&weight_t, tape).add(&self.bias, tape)
    }
}
