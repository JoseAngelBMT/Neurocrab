pub mod linear;
mod loss;
mod module;

#[cfg(test)]
mod test {
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use crate::autograd::backward::backward;
    use crate::autograd::tape::Tape;
    use crate::nn::linear::Linear;
    use crate::nn::loss::mse_loss;
    use crate::optim::sgd::SGD;
    use crate::tensor::Tensor;

    #[test]
    fn xor() {
        let mut rng = StdRng::seed_from_u64(33);
        let xor_data = [
            (vec![0.0f32, 0.0f32], 0.0f32),
            (vec![0.0f32, 1.0f32], 1.0f32),
            (vec![1.0f32, 0.0f32], 1.0f32),
            (vec![1.0f32, 1.0f32], 0.0f32),
        ];

        let mut tape = Tape::<f32>::new();
        let mut linear1 = Linear::new(2, 8, &mut tape);
        let mut linear2 = Linear::new(8, 1, &mut tape);
        let sgd = SGD::new(0.05);

        // Train
        for epoch in 0..5000 {
            for (x_row, y_val) in xor_data.iter() {
                let mut tape = Tape::<f32>::new();
                linear1.register(&mut tape);
                linear2.register(&mut tape);
                let x =
                    tape.variable(Tensor::from_vec(vec![x_row[0], x_row[1]], vec![1, 2]).unwrap());
                let y = tape.variable(Tensor::<f32>::from_vec(vec![
                    *y_val], vec![1, 1]).unwrap());

                let h = linear1.forward(&x, &mut tape).relu(&mut tape);
                let pred = linear2.forward(&h, &mut tape); // Remove sigmoid for testing
                let loss = mse_loss(&pred, &y, &mut tape);

                let grads = backward(&tape, loss.id, Tensor::ones(vec![]));
                sgd.step(&mut vec![&mut linear1.weight, &mut linear1.bias,
                                   &mut linear2.weight, &mut linear2.bias], &grads);
            }
        }

        // Test
        let mut tape = Tape::<f32>::new();
        for (x_row, expected) in xor_data.iter() {
            tape.clear();
            linear1.register(&mut tape);
            linear2.register(&mut tape);
            let x = tape.variable(Tensor::from_vec(x_row.clone(), vec![1, 2]).unwrap());
            let pred = linear2.forward(
                &linear1.forward(&x, &mut tape).relu(&mut tape),
                &mut tape
            );

            let output = pred.tensor.data()[0];
            println!("Input {:?} → {:.4} (expected {:.0})", x_row, output, expected);
            assert!((output - expected).abs() < 0.2);
        }
    }
}
