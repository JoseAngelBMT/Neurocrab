use std::cell::{Cell, RefCell};
use crate::autograd::variable::ValueId;
use crate::tensor::Tensor;

pub struct Parameter<T> {
    pub value: RefCell<Tensor<T>>,
    pub id: Cell<Option<ValueId>>,
}

impl<T> Parameter<T> {
    pub fn new(tensor: Tensor<T>) -> Parameter<T> {
        Parameter {
            value: RefCell::new(tensor),
            id: Cell::new(None),
        }
    }
}
