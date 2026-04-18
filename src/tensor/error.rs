#[derive(Debug, Clone, PartialEq)]
pub enum TensorError {
    ShapeMismatch {
        expected: usize,
        actual: usize,
    },
    InvalidIndexRank {
        expected: usize,
        actual: usize,
    },
    IndexOutOfBounds {
        dim: usize,
        index: usize,
        size: usize,
    },
    InvalidReshape {
        from: Vec<usize>,
        to: Vec<usize>,
    },
    ShapeMismatchBinaryOp {
        lhs: Vec<usize>,
        rhs: Vec<usize>,
    },
}