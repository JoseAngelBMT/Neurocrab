use std::ops::Add;
use crate::autograd::variable::ValueId;
use crate::tensor::Tensor;
use std::collections::HashMap;

pub(crate) struct GradStore<T> {
    pub(crate) grads: HashMap<ValueId, Tensor<T>>,
}

impl<T> GradStore<T> {

    pub fn new() -> Self {
        Self {
            grads: HashMap::new(),
        }
    }

    pub fn get(&self, grad_id: &ValueId) -> Option<&Tensor<T>> {
        self.grads.get(grad_id)
    }

    pub fn set(&mut self, grad_id: &ValueId, tensor: Tensor<T>) {
        self.grads.insert(grad_id.clone(), tensor);
    }

    pub fn remove(&mut self, grad_id: &ValueId) -> Option<Tensor<T>> {
        self.grads.remove(grad_id)
    }

    pub fn accumulate(&mut self, grad_id: &ValueId, tensor: Tensor<T>)
    where
        T: Add<Output=T> + Clone
    {
        if let Some(t) = self.grads.remove(grad_id) {
            let new_tensor = tensor.add(&t).unwrap();
            self.set(grad_id, new_tensor);
        }else {
            self.set(grad_id, tensor);
        }
    }
}
