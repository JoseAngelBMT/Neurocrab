use crate::autograd::variable::Variable;
use crate::autograd::grad_store::GradStore;
pub struct SGD {
    lr: f32
}

impl SGD {
    pub fn new(lr: f32) -> SGD {
        SGD {lr}
    }

    pub fn step(&self, params: &mut Vec<&mut Variable<f32>>, grads: &GradStore<f32>) {
        for var in params.iter_mut() {
            if let Some(grad) = grads.get(&var.id) {
                for (w, g) in var.tensor.data_mut().iter_mut().zip(grad.data()) {
                    *w -= self.lr * g;
                }
            }
        }
    }
}