use crate::tensor::error::TensorError;
use crate::ops;
use num_traits::One;

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor<T> {
    data: Vec<T>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl<T> Tensor<T> {
    pub fn from_vec(data: Vec<T>, shape: Vec<usize>) -> Result<Self, TensorError> {
        let expected_shape = num_elements_from_shape(&shape);

        if data.len() != expected_shape {
            return Err(TensorError::ShapeMismatch {
                expected: expected_shape,
                actual: data.len(),
            });
        }
        let strides = compute_contiguous_strides(&shape);

        Ok(Self {
            data,
            shape,
            strides,
        })
    }

    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    pub fn strides(&self) -> &Vec<usize> {
        &self.strides
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn num_elements(&self) -> usize {
        self.data.len()
    }

    pub fn is_contiguous(&self) -> bool {
        self.strides == compute_contiguous_strides(&self.shape)
    }

    pub fn index_to_offset(&self, indices: &[usize]) -> Result<usize, TensorError> {
        if indices.len() != self.rank() {
            return Err(TensorError::InvalidIndexRank {
                expected: self.rank(),
                actual: indices.len(),
            });
        }

        let mut offset = 0;

        for (dim, &idx) in indices.iter().enumerate() {
            let dim_size = self.shape[dim];

            if idx >= dim_size {
                return Err(TensorError::IndexOutOfBounds {
                    dim: dim_size,
                    index: idx,
                    size: dim_size,
                });
            }
            offset += idx * self.strides[dim];
        }
        Ok(offset)
    }

    pub fn get(&self, indices: &[usize]) -> Result<&T, TensorError> {
        let offset = self.index_to_offset(indices)?;
        Ok(&self.data[offset])
    }

    pub fn get_mut(&mut self, indices: &[usize]) -> Result<&mut T, TensorError> {
        let offset = self.index_to_offset(indices)?;
        Ok(&mut self.data[offset])
    }

    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self, TensorError>
    where
        T: Clone,
    {
        let new_num_elements = num_elements_from_shape(&new_shape);

        if new_num_elements != self.num_elements() {
            return Err(TensorError::InvalidReshape {
                from: self.shape.clone(),
                to: new_shape,
            });
        }

        Ok(Self {
            data: self.data.clone(),
            shape: new_shape.clone(),
            strides: compute_contiguous_strides(&new_shape),
        })
    }

}

// Tensor operations
impl<T> Tensor<T> {
    pub fn add(&self, other: &Self) -> Result<Self, TensorError>
    where
        T: std::ops::Add<Output = T> + Clone,
    {
        ops::add(self, other)
    }

    pub fn sub(&self, other: &Self) -> Result<Self, TensorError>
    where
        T: std::ops::Sub<Output = T> + Clone,
    {
        ops::sub(self, other)
    }

    pub fn mul(&self, other: &Self) -> Result<Self, TensorError>
    where
        T: std::ops::Mul<Output = T> + Clone,
    {
        ops::mul(self, other)
    }

    pub fn div(&self, other: &Self) -> Result<Self, TensorError>
    where
        T: std::ops::Div<Output = T> + Clone,
    {
        ops::div(self, other)
    }

    pub fn negate(&self) -> Self
    where
        T: std::ops::Neg<Output = T> + Clone,
    {
        let data: Vec<T> = self.data.iter().map(|x| -x.clone()).collect();
        Tensor::from_vec(data, self.shape.clone()).unwrap()
    }
}


impl<T: Clone + Default + One> Tensor<T> {
    pub fn zeros(shape: Vec<usize>) -> Self {
        let num_elements = num_elements_from_shape(&shape);
        let data = vec![T::default(); num_elements];
        let strides = compute_contiguous_strides(&shape);

        Self {
            data,
            shape,
            strides,
        }
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        let num_elements = num_elements_from_shape(&shape);
        let data = vec![T::one(); num_elements];
        let strides = compute_contiguous_strides(&shape);

        Self {
            data,
            shape,
            strides,
        }
    }
}

pub fn num_elements_from_shape(shape: &Vec<usize>) -> usize {
    shape.iter().product()
}

pub fn compute_contiguous_strides(shape: &Vec<usize>) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }

    let mut strides = vec![0; shape.len()];
    let mut acc = 1;

    for i in (0..shape.len()).rev() {
        strides[i] = acc;
        acc *= shape[i];
    }
    strides
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_tensor() {
        let t = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();

        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.strides(), &[3, 1]);
        assert_eq!(t.rank(), 2);
        assert_eq!(t.num_elements(), 6);
        assert!(t.is_contiguous());
    }

    #[test]
    fn gets_value() {
        let t = Tensor::from_vec(vec![10, 20, 30, 40], vec![2, 2]).unwrap();

        assert_eq!(*t.get(&[0, 0]).unwrap(), 10);
        assert_eq!(*t.get(&[0, 1]).unwrap(), 20);
        assert_eq!(*t.get(&[1, 0]).unwrap(), 30);
        assert_eq!(*t.get(&[1, 1]).unwrap(), 40);
    }

    #[test]
    fn zeros() {
        let t = Tensor::from_vec(vec![0, 0, 0, 0], vec![2, 2]).unwrap();
        let t_zeros = Tensor::<i32>::zeros(vec![2, 2]);

        assert_eq!(t, t_zeros);
    }

    #[test]
    fn reshapes_tensor() {
        let t = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let reshaped = t.reshape(vec![3, 2]).unwrap();

        assert_eq!(reshaped.shape(), &[3, 2]);
        assert_eq!(reshaped.strides(), &[2, 1]);
        assert_eq!(reshaped.num_elements(), 6);

        assert_eq!(*reshaped.get(&[0, 0]).unwrap(), 1);
        assert_eq!(*reshaped.get(&[0, 1]).unwrap(), 2);
        assert_eq!(*reshaped.get(&[1, 0]).unwrap(), 3);
        assert_eq!(*reshaped.get(&[2, 1]).unwrap(), 6);
    }

    #[test]
    fn reshape_fails_when_num_elements_do_not_match() {
        let t = Tensor::from_vec(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let result = t.reshape(vec![3, 2]);

        assert_eq!(
            result.unwrap_err(),
            TensorError::InvalidReshape {
                from: vec![2, 2],
                to: vec![3, 2],
            }
        );
    }

    #[test]
    fn adds_tensors() {
        let a = Tensor::from_vec(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![4, 3, 2, 1], vec![2, 2]).unwrap();

        let result = a.add(&b).unwrap();
        assert_eq!(
            result,
            Tensor::from_vec(vec![5, 5, 5, 5], vec![2, 2]).unwrap()
        );
    }

    #[test]
    fn subs_tensors() {
        let a = Tensor::from_vec(vec![5, 6, 7, 8], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![1, 2, 3, 4], vec![2, 2]).unwrap();

        let result = a.sub(&b).unwrap();
        assert_eq!(
            result,
            Tensor::from_vec(vec![4, 4, 4, 4], vec![2, 2]).unwrap()
        );
    }

    #[test]
    fn muls_tensors() {
        let a = Tensor::from_vec(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![4, 3, 2, 1], vec![2, 2]).unwrap();

        let result = a.mul(&b).unwrap();
        assert_eq!(
            result,
            Tensor::from_vec(vec![4, 6, 6, 4], vec![2, 2]).unwrap()
        );
    }

    #[test]
    fn divs_tensors() {
        let a = Tensor::from_vec(vec![8, 6, 4, 2], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![4, 3, 2, 1], vec![2, 2]).unwrap();

        let result = a.div(&b).unwrap();
        assert_eq!(
            result,
            Tensor::from_vec(vec![2, 2, 2, 2], vec![2, 2]).unwrap()
        );
    }
}
