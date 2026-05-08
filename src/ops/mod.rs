use crate::tensor::{Tensor, TensorError};

macro_rules! impl_binary_op {
    ($fn_name:ident, $trait:ident, $method:ident) => {
        pub fn $fn_name<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>, TensorError>
        where
            T: std::ops::$trait<Output=T> + Clone,
        {
            if a.shape() != b.shape() {
                return Err(TensorError::ShapeMismatchBinaryOp {
                    lhs: a.shape().clone(),
                    rhs: b.shape().clone(),
                })
            }

            let data = a
                .data()
                .iter()
                .zip(b.data().iter())
                .map(|(x, y)| x.clone().$method(y.clone()))
                .collect();

            Tensor::from_vec(data, a.shape().clone())
        }
    };
}

impl_binary_op!(add, Add, add);
impl_binary_op!(sub, Sub, sub);
impl_binary_op!(mul, Mul, mul);
impl_binary_op!(div, Div, div);