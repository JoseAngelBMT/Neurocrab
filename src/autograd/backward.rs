use crate::autograd::grad_store::GradStore;
use crate::autograd::saved::SavedContext;
use crate::autograd::tape::{OpKind, Tape};
use crate::autograd::variable::ValueId;
use crate::tensor::Tensor;
use std::ops::{Add, Div, Mul, Neg, Sub};

pub fn backward<T>(tape: &Tape<T>, loss_id: ValueId, initial_gradient: Tensor<T>) -> GradStore<T>
where
    T: Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Neg<Output = T>
        + Clone,
{
    let mut grad_store = GradStore::new();
    grad_store.set(&loss_id, initial_gradient);

    for node in tape.nodes.iter().rev() {
        if let Some(grad_out) = grad_store.remove(&node.output) {
            let (grad_a, grad_b) = match node.op {
                OpKind::Add => (grad_out.clone(), grad_out.clone()),
                OpKind::Sub => (grad_out.clone(), grad_out.clone().negate()),
                OpKind::Mul => {
                    let a: Tensor<T> = get_saved_tensor(&node.saved, 0);
                    let b: Tensor<T> = get_saved_tensor(&node.saved, 1);
                    (grad_out.mul(&b).unwrap(), grad_out.mul(&a).unwrap())
                }
                OpKind::Div => {
                    let a: Tensor<T> = get_saved_tensor(&node.saved, 0);
                    let b: Tensor<T> = get_saved_tensor(&node.saved, 1);
                    let grad_a = grad_out.div(&b).unwrap();
                    let grad_b = grad_out
                        .mul(&a)
                        .unwrap()
                        .div(&b.mul(&b).unwrap())
                        .unwrap()
                        .negate();
                    (grad_a, grad_b)
                }
            };
            grad_store.accumulate(&node.inputs[0], grad_a);
            grad_store.accumulate(&node.inputs[1], grad_b);
        } else {
            continue;
        };
    }
    grad_store
}

fn get_saved_tensor<T>(saved_context: &SavedContext<T>, id: usize) -> Tensor<T>
where
    T: Clone,
{
    match saved_context {
        SavedContext::None => panic!("Requires saved context"),
        SavedContext::Tensors(tensors) => tensors[id].clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mul_autograd() {
        /*
        x = 3.0, w = 2.0
        y = x * w = 6.0
        ∂y/∂x = w = 2.0
        ∂y/∂w = x = 3.0
         */
        let mut tape = Tape::new();
        let x = Tensor::from_vec(vec![3.0], vec![1]).unwrap();
        let w = Tensor::from_vec(vec![2.0], vec![1]).unwrap();

        let x_v = tape.variable(x);
        let w_v = tape.variable(w);
        let y = x_v.mul(&w_v, &mut tape);

        let initial_grad = Tensor::ones(vec![1]); // ∂loss/∂loss
        let grad_store = backward(&tape, y.id, initial_grad);
        assert_eq!(grad_store.get(&x_v.id).unwrap().data(), &vec![2.0]);
        assert_eq!(grad_store.get(&w_v.id).unwrap().data(), &vec![3.0]);
    }

    #[test]
    fn add_autograd() {
        let mut tape = Tape::new();
        let x = Tensor::from_vec(vec![3.0], vec![1]).unwrap();
        let w = Tensor::from_vec(vec![2.0], vec![1]).unwrap();
        let x_v = tape.variable(x);
        let w_v = tape.variable(w);
        let y = x_v.add(&w_v, &mut tape);

        let initial_grad = Tensor::ones(vec![1]);
        let grad_store = backward(&tape, y.id, initial_grad);
        assert_eq!(grad_store.get(&x_v.id).unwrap().data(), &vec![1.0]);
        assert_eq!(grad_store.get(&w_v.id).unwrap().data(), &vec![1.0]);
    }

    #[test]
    fn sub_autograd() {
        let mut tape = Tape::new();
        let x = Tensor::from_vec(vec![3.0], vec![1]).unwrap();
        let w = Tensor::from_vec(vec![2.0], vec![1]).unwrap();
        let x_v = tape.variable(x);
        let w_v = tape.variable(w);
        let y = x_v.sub(&w_v, &mut tape);

        let initial_grad = Tensor::ones(vec![1]);
        let grad_store = backward(&tape, y.id, initial_grad);
        assert_eq!(grad_store.get(&x_v.id).unwrap().data(), &vec![1.0]);
        assert_eq!(grad_store.get(&w_v.id).unwrap().data(), &vec![-1.0]);
    }


    #[test]
    fn div_autograd() {
        let mut tape = Tape::new();
        let x = Tensor::from_vec(vec![6.0], vec![1]).unwrap();
        let w = Tensor::from_vec(vec![2.0], vec![1]).unwrap();
        let x_v = tape.variable(x);
        let w_v = tape.variable(w);
        let y = x_v.div(&w_v, &mut tape);

        let initial_grad = Tensor::ones(vec![1]);
        let grad_store = backward(&tape, y.id, initial_grad);
        assert_eq!(grad_store.get(&x_v.id).unwrap().data(), &vec![0.5]);
        assert_eq!(grad_store.get(&w_v.id).unwrap().data(), &vec![-1.5]);
    }

    #[test]
    fn multi_autograd() {
        let mut tape = Tape::new();
        let x = Tensor::from_vec(vec![3.0], vec![1]).unwrap();
        let w = Tensor::from_vec(vec![2.0], vec![1]).unwrap();
        let b = Tensor::from_vec(vec![1.0], vec![1]).unwrap();

        let x_v = tape.variable(x);
        let w_v = tape.variable(w);
        let b_v = tape.variable(b);
        let y = x_v.mul(&w_v, &mut tape);
        let z = y.add(&b_v, &mut tape);

        let initial_grad = Tensor::ones(vec![1]);
        let grad_store = backward(&tape, z.id, initial_grad);

        assert_eq!(grad_store.get(&x_v.id).unwrap().data(), &vec![2.0]);
        assert_eq!(grad_store.get(&w_v.id).unwrap().data(), &vec![3.0]);
        assert_eq!(grad_store.get(&b_v.id).unwrap().data(), &vec![1.0]);
        assert!(grad_store.get(&y.id).is_none());
    }
}
