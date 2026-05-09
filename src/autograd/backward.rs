use crate::autograd::grad_store::GradStore;
use crate::autograd::saved::SavedContext;
use crate::autograd::saved::SavedContext::Tensors;
use crate::autograd::tape::{OpKind, Tape};
use crate::autograd::variable::ValueId;
use crate::ops::{matmul, transpose};
use crate::tensor::Tensor;
use std::ops::{Add, Div, Mul, Neg, Sub};

pub fn backward<T>(tape: &Tape<T>, loss_id: ValueId, initial_gradient: Tensor<T>) -> GradStore<T>
where
    T: Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Neg<Output = T>
        + std::iter::Sum<T>
        + Default
        + Clone,
{
    let mut grad_store = GradStore::new();
    grad_store.set(&loss_id, initial_gradient);

    for node in tape.nodes.iter().rev() {
        if let Some(grad_out) = grad_store.remove(&node.output) {
            let grads: Vec<Tensor<T>> = match node.op {
                OpKind::Add => vec![grad_out.clone(), grad_out.clone()],
                OpKind::Sub => vec![grad_out.clone(), grad_out.clone().negate()],
                OpKind::Sum => {
                    let shape = get_saved_shape(&node.saved);
                    let n = shape.iter().product();
                    let value = grad_out.data()[0].clone();
                    vec![Tensor::from_vec(vec![value; n], shape).unwrap()]
                }
                OpKind::Mul => {
                    let a: Tensor<T> = get_saved_tensor(&node.saved, 0);
                    let b: Tensor<T> = get_saved_tensor(&node.saved, 1);
                    vec![grad_out.mul(&b).unwrap(), grad_out.mul(&a).unwrap()]
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
                    vec![grad_a, grad_b]
                }
                OpKind::MatMul => {
                    let a: Tensor<T> = get_saved_tensor(&node.saved, 0);
                    let b: Tensor<T> = get_saved_tensor(&node.saved, 1);
                    let b_t = transpose(&b);
                    let a_t = transpose(&a);
                    vec![
                        matmul(&grad_out, &b_t).unwrap(), // dA = dC @ B^T
                        matmul(&a_t, &grad_out).unwrap(), // dB = A^T @ dC
                    ]
                }
            };
            for (input, grad) in node.inputs.iter().zip(grads.into_iter()) {
                grad_store.accumulate(input, grad);
            }
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
        SavedContext::Tensors(tensors) => tensors[id].clone(),
        _ => panic!("Requires saved tensors."),
    }
}

fn get_saved_shape<T>(saved_context: &SavedContext<T>) -> Vec<usize> {
    match saved_context {
        SavedContext::InputShape(saved_shape) => saved_shape.clone(),
        _ => panic!("Requires saved shape."),
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

    #[test]
    fn matmul_autograd() {
        let mut tape = Tape::new();
        let a = Tensor::from_vec(vec![1, 2, 3, 4], vec![2,2]).unwrap();
        let b = Tensor::from_vec(vec![5, 6, 7, 8], vec![2,2]).unwrap();

        let a_v = tape.variable(a);
        let b_v = tape.variable(b);
        let c = a_v.matmul(&b_v, &mut tape);

        let initial_grad = Tensor::ones(vec![2,2]);
        let grad_store = backward(&tape, c.id, initial_grad);
        assert_eq!(grad_store.get(&a_v.id).unwrap().data(), &vec![11, 15, 11, 15]);
        assert_eq!(grad_store.get(&b_v.id).unwrap().data(), &vec![4, 4, 6, 6]);

    }
}
