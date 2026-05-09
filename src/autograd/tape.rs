use crate::autograd::saved::SavedContext;
use crate::autograd::variable::{ValueId, Variable};
use crate::tensor::Tensor;

#[derive(Debug, Clone, Copy)]
pub(crate) enum OpKind {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone)]
pub(crate) struct Node<T> {
    pub(crate) op: OpKind,
    pub(crate) inputs: Vec<ValueId>,
    pub(crate) output: ValueId,
    pub(crate) saved: SavedContext<T>,
    pub(crate) output_shape: Vec<usize>,
}

pub struct Tape<T> {
    // Is equal to graph
    pub(crate) nodes: Vec<Node<T>>,
    pub(crate) next_id: usize,
}

impl<T> Tape<T> {
    pub fn new() -> Tape<T> {
        Self {
            nodes: vec![],
            next_id: 0,
        }
    }

    pub fn register(
        &mut self,
        op_kind: OpKind,
        inputs: Vec<ValueId>,
        saved_context: SavedContext<T>,
        output_shape: Vec<usize>,
    ) -> ValueId {
        self.next_id +=  1;

        let node = Node {
            op: op_kind,
            inputs,
            output: ValueId(self.next_id),
            saved: saved_context,
            output_shape
        };
        self.nodes.push(node);
        ValueId(self.next_id)
    }

    pub fn variable(&mut self, tensor: Tensor<T>) -> Variable<T> {
        let id = ValueId(self.next_id);
        self.next_id += 1;
        Variable {
            id,
            tensor,
            requires_grad: true,
        }
    }
}
