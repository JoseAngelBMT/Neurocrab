use crate::autograd::tape::Tape;
use crate::autograd::variable::Variable;

pub fn mse_loss(
    pred: &Variable<f32>,
    target: &Variable<f32>,
    tape: &mut Tape<f32>,
) -> Variable<f32> {
    let diff = pred.sub(&target, tape);
    let sq = diff.mul(&diff, tape);
    sq.mean(tape)
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