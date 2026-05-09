use crate::autograd::saved::SavedContext;
use crate::autograd::tape::{OpKind, Tape};
use crate::tensor::Tensor;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(pub usize);

pub struct Variable<T> {
    pub id: ValueId,
    pub tensor: Tensor<T>,
    pub requires_grad: bool,
}

impl<T> Variable<T> {
    pub fn add(&self, other: &Self, tape: &mut Tape<T>) -> Variable<T>
    where
        T: std::ops::Add<Output = T> + Clone,
    {
        let result = crate::ops::add(&self.tensor, &other.tensor).unwrap();
        let id = tape.register(
            OpKind::Add,
            vec![self.id, other.id],
            SavedContext::None,
            result.shape().clone(),
        );
        Variable {
            id,
            tensor: result,
            requires_grad: false,
        }
    }

    pub fn sub(&self, other: &Self, tape: &mut Tape<T>) -> Variable<T>
    where
        T: std::ops::Sub<Output = T> + Clone,
    {
        let result = crate::ops::sub(&self.tensor, &other.tensor).unwrap();
        let id = tape.register(
            OpKind::Sub,
            vec![self.id, other.id],
            SavedContext::None,
            result.shape().clone(),
        );
        Variable {
            id,
            tensor: result,
            requires_grad: false,
        }
    }

    pub fn mul(&self, other: &Self, tape: &mut Tape<T>) -> Variable<T>
    where
        T: std::ops::Mul<Output = T> + Clone,
    {
        let result = crate::ops::mul(&self.tensor, &other.tensor).unwrap();
        let id = tape.register(
            OpKind::Mul,
            vec![self.id, other.id],
            SavedContext::Tensors(vec![self.tensor.clone(), other.tensor.clone()]),
            result.shape().clone(),
        );
        Variable {
            id,
            tensor: result,
            requires_grad: false,
        }
    }

    pub fn div(&self, other: &Self, tape: &mut Tape<T>) -> Variable<T>
    where
        T: std::ops::Div<Output = T> + Clone,
    {
        let result = crate::ops::div(&self.tensor, &other.tensor).unwrap();
        let id = tape.register(
            OpKind::Div,
            vec![self.id, other.id],
            SavedContext::Tensors(vec![self.tensor.clone(), other.tensor.clone()]),
            result.shape().clone(),
        );
        Variable {
            id,
            tensor: result,
            requires_grad: false,
        }
    }
}
