use crate::tensor::Tensor;

#[derive(Debug, Clone)]
pub(crate) enum SavedContext<T> {
    None,
    Tensors(Vec<Tensor<T>>),
}
