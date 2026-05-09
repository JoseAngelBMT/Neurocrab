mod error;

use crate::ops::error::OperationError;
use crate::tensor::{Tensor, TensorError};

macro_rules! impl_binary_op {
    ($fn_name:ident, $trait:ident, $method:ident) => {
        pub fn $fn_name<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>, TensorError>
        where
            T: std::ops::$trait<Output = T> + Clone,
        {
            if a.shape() != b.shape() {
                return Err(TensorError::ShapeMismatchBinaryOp {
                    lhs: a.shape().clone(),
                    rhs: b.shape().clone(),
                });
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

// 2D
pub fn transpose<T>(a: &Tensor<T>) -> Tensor<T>
where
    T: Clone,
{
    let mut new_shape = a.shape().to_vec();
    let mut new_strides = a.strides().to_vec();
    new_shape.reverse();
    new_strides.reverse();
    Tensor::from_raw(a.data().clone(), new_shape, new_strides).unwrap()
}

pub fn sum<T>(a: &Tensor<T>) -> Tensor<T>
where
    T: std::iter::Sum<T> + Clone,
{
    let result = a.data().iter().cloned().sum();
    Tensor::from_vec(vec![result], vec![]).unwrap()
}

pub fn matmul<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>, OperationError>
where
    T: std::ops::Mul<Output = T> + std::ops::Add<Output = T> + Default + Clone,
{
    if a.shape()[1] != b.shape()[0] {
        return Err(OperationError::InvalidMatMulShape {
            a: a.shape().clone(),
            b: b.shape().clone(),
        });
    }

    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[1];

    let mut data: Vec<T> = Vec::with_capacity(m * n);

    for x in 0..m {
        for y in 0..n {
            let mut acc = T::default();
            for w in 0..k {
                let axw = a.get(&[x, w]).unwrap();
                let bwy = b.get(&[w, y]).unwrap();
                acc = acc.add(axw.clone().mul(bwy.clone()))
            }
            data.push(acc);
        }
    }
    Ok(Tensor::from_vec(data, vec![m, n]).unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn transpose_2d() {
        let a = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let t = transpose(&a);
        assert_eq!(t.shape(), &[3, 2]);
        assert_eq!(t.strides(), &[1, 3]);
        assert_eq!(*t.get(&[0, 0]).unwrap(), 1);
        assert_eq!(*t.get(&[0, 1]).unwrap(), 4);
        assert_eq!(*t.get(&[1, 0]).unwrap(), 2);
        assert_eq!(*t.get(&[1, 1]).unwrap(), 5);
        assert_eq!(*t.get(&[2, 0]).unwrap(), 3);
        assert_eq!(*t.get(&[2, 1]).unwrap(), 6);
    }

    #[test]
    fn sum_2d() {
        let a = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let result = sum(&a);
        assert_eq!(result.data(), &[21]);
    }

    #[test]
    fn matmul_2d() {
        let a = Tensor::from_vec(vec![2, 1, 1, 4], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![1, 2, 0, 0, 1, 2], vec![2, 3]).unwrap();

        let expected = Tensor::from_vec(vec![2, 5, 2, 1, 6, 8], vec![2, 3]).unwrap();
        let result = matmul(&a, &b).unwrap();
        assert_eq!(expected, result);
    }

    #[test]
    fn wrong_matmul_shape() {
        let a = Tensor::from_vec(vec![1, 2, 0, 0, 1, 2], vec![2, 3]).unwrap();
        let b = Tensor::from_vec(vec![1, 2, 0, 0, 1, 2], vec![2, 3]).unwrap();
        let result = matmul(&a, &b);
        assert!(result.is_err());
    }
}
