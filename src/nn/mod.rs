pub mod linear;
pub mod loss;
mod module;
pub mod parameter;
mod sequential;
mod relu;

#[cfg(test)]
mod test {
    use crate::autograd::backward::backward;
    use crate::autograd::tape::Tape;
    use crate::nn::linear::Linear;
    use crate::nn::loss::mse_loss;
    use crate::nn::module::Module;
    use crate::nn::relu::ReLU;
    use crate::nn::sequential::Sequential;
    use crate::optim::sgd::SGD;
    use crate::tensor::Tensor;

    #[test]
    fn xor() {
        let xor_data = [
            (vec![0.0f32, 0.0f32], 0.0f32),
            (vec![0.0f32, 1.0f32], 1.0f32),
            (vec![1.0f32, 0.0f32], 1.0f32),
            (vec![1.0f32, 1.0f32], 0.0f32),
        ];

        let tape = Tape::<f32>::new();
        let linear1 = Linear::new(2, 16);
        let linear2 = Linear::new(16, 1);
        let sgd = SGD::new(0.01);

        let model = Sequential::new(vec![
            Box::new(linear1),
            Box::new(ReLU),
            Box::new(linear2),
        ]);

        // Train
        for _ in 0..5000 {
            for (x_row, y_val) in xor_data.iter() {
                let mut tape = Tape::<f32>::new();
                let x =
                    tape.variable(Tensor::from_vec(vec![x_row[0], x_row[1]], vec![1, 2]).unwrap());
                let y = tape.variable(Tensor::<f32>::from_vec(vec![
                    *y_val], vec![1, 1]).unwrap());

                let pred = model.forward(&x, &mut tape);
                let loss = mse_loss(&pred, &y, &mut tape);

                let grads = backward(&tape, loss.id, Tensor::ones(vec![]));
                let params = model.parameters();
                sgd.step(params, &grads);
            }
        }

        // Test
        let mut tape = Tape::<f32>::new();
        for (x_row, expected) in xor_data.iter() {
            tape.clear();
            let x = tape.variable(Tensor::from_vec(x_row.clone(), vec![1, 2]).unwrap());
            let pred = model.forward(&x, &mut tape);

            let output = pred.tensor.data()[0];
            println!("Input {:?} → {:.4} (expected {:.0})", x_row, output, expected);
            assert!((output - expected).abs() < 0.2);
        }
    }
}
