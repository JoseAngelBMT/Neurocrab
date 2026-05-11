use crate::autograd::grad_store::GradStore;
use crate::nn::parameter::Parameter;
use crate::tensor::Tensor;

pub struct SGD {
    lr: f32,
}

impl SGD {
    pub fn new(lr: f32) -> SGD {
        SGD { lr }
    }

    pub fn step(&self, params: Vec<&Parameter<f32>>, grads: &GradStore<f32>) {
        for param in params {
            let mut value = param.value.borrow_mut();
            let Some(id) = param.id.get() else {continue};
            if let Some(grad) = grads.get(&id) {
                for (w, g) in value.data_mut().iter_mut().zip(grad.data()) {
                    *w -= self.lr * g;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::variable::ValueId;

    #[test]
    fn sgd_updates_parameter() {
        let mut grad_store = GradStore::new();

        let tensor = Tensor::from_vec(vec![1.0f32, 2.0f32, 3.0f32], vec![3]).unwrap();
        let param = Parameter::new(tensor);

        // Simulate what forward does: set ID + put gradient in store
        let id = ValueId(5);
        param.id.set(Some(id));
        let grad = Tensor::from_vec(vec![0.1f32, 0.2f32, 0.3f32], vec![3]).unwrap();
        grad_store.set(&id, grad);

        let sgd = SGD::new(0.5);
        sgd.step(vec![&param], &grad_store);

        // w -= lr * grad: 1.0 - 0.5*0.1=0.95, 2.0-0.5*0.2=1.9, 3.0-0.5*0.3=2.85
        assert_eq!(param.value.borrow().data(), &vec![0.95, 1.9, 2.85]);
    }

    #[test]
    fn sgd_skips_missing_gradient() {
        let tensor = Tensor::from_vec(vec![5.0f32], vec![1]).unwrap();
        let param = Parameter::new(tensor);

        param.id.set(Some(ValueId(99)));
        let grad_store = GradStore::new(); // empty, no grad for ID 99

        let sgd = SGD::new(0.1);
        sgd.step(vec![&param], &grad_store);

        assert_eq!(param.value.borrow().data(), &vec![5.0]); // unchanged
    }

    #[test]
    fn sgd_skips_null_id() {
        let tensor = Tensor::from_vec(vec![7.0f32], vec![1]).unwrap();
        let param = Parameter::new(tensor);
        // id is None by default, no set called

        let grad_store = GradStore::new();
        let sgd = SGD::new(0.1);
        sgd.step(vec![&param], &grad_store);

        assert_eq!(param.value.borrow().data(), &vec![7.0]); // unchanged
    }
}
