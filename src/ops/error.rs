#[derive(Debug, Clone, PartialEq)]
pub enum OperationError {
    InvalidMatMulShape { a: Vec<usize>, b: Vec<usize> },
}
