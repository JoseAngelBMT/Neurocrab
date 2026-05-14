use crate::autograd::saved::SavedContext;
use crate::autograd::tape::{OpKind, Tape};
use crate::autograd::variable::Variable;
use crate::ops;
use crate::tensor::Tensor;

pub fn mse_loss(
    pred: &Variable<f32>,
    target: &Variable<f32>,
    tape: &mut Tape<f32>,
) -> Variable<f32> {
    let diff = pred.sub(&target, tape);
    let sq = diff.mul(&diff, tape);
    sq.mean(tape)
}

pub fn cross_entropy_loss(
    logits: &Variable<f32>,
    targets: &Tensor<usize>,
    tape: &mut Tape<f32>,
) -> Variable<f32> {
    let probs = ops::softmax(&logits.tensor);
    let c = *probs.shape().last().unwrap();
    let batch = targets.num_elements();
    let mut loss_val = 0.0f32;

    for r in 0..batch {
        let t = targets.data()[r] as usize;
        let p = probs.data()[r * c + t];
        loss_val -= (p + 1e-10).ln();
    }
    loss_val /= batch as f32;

    let target_f32: Vec<f32> = targets.data().iter().map(|&t| t as f32).collect();
    let targets_tensor = Tensor::from_vec(target_f32, vec![batch]).unwrap();

    let id = tape.register(
        OpKind::CrossEntropy,
        vec![logits.id],
        SavedContext::Tensors(vec![probs, targets_tensor]),
        vec![],
    );

    Variable {
        id,
        tensor: Tensor::from_vec(vec![loss_val], vec![]).unwrap(),
        requires_grad: false,
    }
}

#[cfg(test)]
mod tests {
    use crate::autograd::backward::backward;
    use crate::autograd::tape::Tape;
    use crate::nn::loss::mse_loss;
    use crate::tensor::Tensor;

    #[test]
    fn mse_loss_test() {
        let mut tape = Tape::<f32>::new();
        let pred = Tensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap();
        let target = Tensor::from_vec(vec![2.0, 4.0], vec![2]).unwrap();

        let pred_v = tape.variable(pred);
        let target_v = tape.variable(target);

        let loss = mse_loss(&pred_v, &target_v, &mut tape);
        assert_eq!(loss.tensor.data(), &vec![2.5f32]);
        let grads = backward(&tape, loss.id, Tensor::ones(vec![]));
        assert_eq!(grads.get(&pred_v.id).unwrap().data(), &vec![-1.0, -2.0]);
    }
}
