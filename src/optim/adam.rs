use crate::autograd::grad_store::GradStore;
use crate::autograd::variable::ValueId;
use crate::nn::parameter::Parameter;
use crate::tensor::Tensor;
use std::collections::HashMap;
use num_traits::real::Real;

pub struct Adam {
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    t: usize,
    m: HashMap<ValueId, Tensor<f32>>,
    v: HashMap<ValueId, Tensor<f32>>,
}

impl Adam {
    pub fn new(lr: f32) -> Self {
        Adam {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            t: 0,
            m: HashMap::<ValueId, Tensor<f32>>::new(),
            v: HashMap::<ValueId, Tensor<f32>>::new(),
        }
    }

    pub fn step(&mut self, params: Vec<&Parameter<f32>>, grads: &GradStore<f32>) {
        self.t += 1;
        for param in params {
            let Some(id) = param.id.get() else { continue };
            let Some(grad) = grads.get(&id) else { continue };

            let mut tensor = param.value.borrow_mut();
            let m_arr = self
                .m
                .entry(id)
                .or_insert_with(|| Tensor::zeros(tensor.shape().clone()));
            let v_arr = self
                .v
                .entry(id)
                .or_insert_with(|| Tensor::zeros(tensor.shape().clone()));

            let bc1 = 1.0 - self.beta1.powi(self.t as i32);
            let bc2 = 1.0 - self.beta2.powi(self.t as i32);

            for i in 0..tensor.num_elements() {
                let g = grad.data()[i];
                m_arr.data_mut()[i] = self.beta1 * m_arr.data()[i] + (1.0 - self.beta1) * g;
                v_arr.data_mut()[i] = self.beta2 * v_arr.data()[i] + (1.0 - self.beta2) * g * g;

                let m_corr = m_arr.data()[i] / bc1;
                let v_corr = v_arr.data()[i] / bc2;
                tensor.data_mut()[i] -= self.lr * m_corr / (v_corr.sqrt() + self.epsilon);
            }
        }
    }
}
