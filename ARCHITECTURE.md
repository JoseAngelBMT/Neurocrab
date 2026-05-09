# Neurocrab — Architecture & Implementation Guide

## Objective

Build a PyTorch clone in Rust from scratch (zero dependencies, std only). Learn:
- Neural network internals: autograd, backpropagation, layers, optimizers
- PyTorch internals: how it works under the hood
- Idiomatic Rust: ownership, traits, generics, zero-cost abstractions

## Architecture Principle

**Pure tensor math is separate from autograd graph management.**

```
┌─────────────────────────────────────────────┐
│  nn/          Layers, losses, activations   │
├─────────────────────────────────────────────┤
│  optim/       SGD, Adam, parameter updates  │
├─────────────────────────────────────────────┤
│  autograd/    Variable, Tape, GradStore     │
│               backward engine               │
├─────────────────────────────────────────────┤
│  ops/         Pure math: add, mul, matmul   │
│               sum, relu, transpose, etc.    │
├─────────────────────────────────────────────┤
│  tensor/      Tensor<T>: data, shape, strides│
│               indexing, reshape, validation │
└─────────────────────────────────────────────┘
```

### Rules
- `tensor/` knows nothing about autograd
- `ops/` knows nothing about graphs or models
- `autograd/` tracks operations using IDs, not Rust references
- `nn/` consumes lower layers, exposes `Module` trait
- Use `f32` for all neural network code (keep tensor generic for foundation)

### Codebase map

```
src/
├── lib.rs                    → mod tensor; mod ops; mod autograd; mod nn; mod optim;
├── tensor/                   → Pure tensor (current code, cleaned up)
│   ├── mod.rs
│   ├── tensor.rs             → Tensor<T>, constructors, indexing, reshape
│   └── error.rs              → TensorError enum
├── ops/                      → Pure math kernels
│   ├── mod.rs                → add, sub, mul, div, sum, mean, exp, log, pow
│   └── matmul.rs             → matrix multiplication
├── autograd/                 → Automatic differentiation
│   ├── mod.rs
│   ├── variable.rs           → Variable { id: ValueId, tensor, requires_grad }
│   ├── tape.rs               → Tape { nodes: Vec<Node> }, register()
│   ├── grad_store.rs         → GradStore { grads: Vec<Option<Tensor>> }
│   ├── saved.rs              → SavedContext enum (minimal data for backward)
│   └── backward.rs           → backward(tape, loss_id) -> GradStore
├── nn/                       → Neural network layers
│   ├── mod.rs
│   ├── parameter.rs          → Parameter { value: Variable }
│   ├── module.rs             → Module trait
│   ├── linear.rs             → Linear { weight, bias }
│   ├── activation.rs         → ReLU, Sigmoid, Tanh
│   └── loss.rs               → MSELoss, CrossEntropyLoss
└── optim/                    → Optimizers
    ├── mod.rs
    ├── sgd.rs                → SGD { lr }
    └── adam.rs               → Adam (future)
```

---

## Implementation Checklist

### MILESTONE 1: Pure Tensor Foundation ✅ (current state)

- [x] `Tensor<T>` struct: `data`, `shape`, `strides`
- [x] `from_vec(data, shape)` — with validation
- [x] `zeros(shape)` — all-zero tensor
- [x] `shape()`, `strides()`, `rank()`, `num_elements()`
- [x] `index_to_offset(indices)` — stride-based offset
- [x] `get(indices)`, `get_mut(indices)` — multi-dimensional access
- [x] `is_contiguous()` — stride check
- [x] `reshape(new_shape)` — with element-count validation
- [x] Element-wise ops: `add`, `sub`, `mul`, `div`
- [x] `TensorError` enum with structured variants
- [x] Tests: creation, indexing, reshape, all binary ops
- [x] `num_elements_from_shape`, `compute_contiguous_strides` helpers

### MILESTONE 2: Clean Up & Extract Ops

- [ ] ~~Add `pub fn data(&self) -> &[T]` accessor to Tensor~~ DONE
- [ ] ~~Move binary ops (`add`, `sub`, `mul`, `div`) to `src/ops/mod.rs` as free functions~~ KEPT on Tensor
  - [ ] Pure `fn add(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>, TensorError>`
  - [ ] Pure `fn sub(...)`, `fn mul(...)`, `fn div(...)`
  - [ ] All existing tests still pass

### MILESTONE 3: Missing Tensor Operations

- [ ] Block 1 — Transpose
  - [ ] `transpose(dim0, dim1)` — stride-swap, zero-copy
  - [ ] Tests: 2D matrix transpose, verify via `get()`
- [ ] Block 2 — Reduction ops
  - [ ] `sum_all()` — reduce entire tensor to scalar
  - [ ] `sum_dim(dim: usize)` — reduce along one axis
  - [ ] `mean()` — same as sum divided by n
  - [ ] Tests: 2D sum_all, sum_dim(0), sum_dim(1)
- [ ] Block 3 — Matrix multiplication
  - [ ] `matmul(a, b)` — 2D: [M,K] @ [K,N] = [M,N]
  - [ ] Inner dimension validation
  - [ ] Tests: known 2×3 @ 3×2 = 2×2, error on bad shapes
- [ ] Block 4 — Scalar math
  - [ ] `exp(tensor)` — element-wise e^x
  - [ ] `log(tensor)` — element-wise ln(x)
  - [ ] `pow(tensor, n)` — element-wise x^n
  - [ ] Tests: verify on known values

### MILESTONE 4: Autograd Infrastructure

- [ ] Block 5 — `ValueId`
  - [ ] `pub struct ValueId(pub usize);`
  - [ ] `#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]`
- [ ] Block 6 — `Variable`
  - [ ] `Variable { id: ValueId, tensor: Tensor<f32>, requires_grad: bool }`
  - [ ] `fn leaf(tensor, tape) -> Variable` — creates leaf, sets `requires_grad = true`
  - [ ] `fn parameter(tensor, tape) -> Variable` — same as leaf (parameter = leaf)
- [ ] Block 7 — `Tape`
  - [ ] `Tape { nodes: Vec<Node>, next_id: usize }`
  - [ ] `Node { op: Op, inputs: Vec<ValueId>, output: ValueId, saved: SavedContext }`
  - [ ] `Op enum { Add, Sub, Mul, Div, MatMul, Sum, Relu, ... }`
  - [ ] `fn register(&mut self, op, inputs, saved) -> ValueId`
- [ ] Block 8 — `SavedContext`
  - [ ] `enum SavedContext { None, Binary { lhs: ValueId, rhs: ValueId }, ... }`
  - [ ] Minimal data — only what backward NEEDS
- [ ] Block 9 — `GradStore`
  - [ ] `GradStore { grads: Vec<Option<Tensor<f32>>> }`
  - [ ] `fn new(size) -> Self`
  - [ ] `fn seed(&mut self, id: ValueId, gradient: Tensor<f32>)`
  - [ ] `fn get(&self, id: ValueId) -> &Tensor<f32>`
  - [ ] `fn accumulate(&mut self, id: ValueId, gradient: &Tensor<f32>)`
- [ ] Block 10 — Tracked ops on `Variable`
  - [ ] `Variable::add(&self, other, tape) -> Variable`
  - [ ] `Variable::sub(&self, other, tape) -> Variable`
  - [ ] `Variable::mul(&self, other, tape) -> Variable`
  - [ ] `Variable::div(&self, other, tape) -> Variable`
  - [ ] Each calls pure `ops::` function internally, then registers in tape

### MILESTONE 5: Backward Pass (THE ENGINE)

- [ ] Block 11 — `backward.rs`
  - [ ] `fn backward(tape: &Tape, loss: ValueId) -> GradStore`
  - [ ] Algorithm:
    1. Create `GradStore` with capacity = tape node count
    2. Seed loss gradient = `Tensor::ones(loss_shape)` (scalar `[1.0]`)
    3. Walk tape nodes in reverse order
    4. For each node, match `op`:
       - `Add`: grad_a = grad_out, grad_b = grad_out
       - `Sub`: grad_a = grad_out, grad_b = -grad_out
       - `Mul`: grad_a = grad_out * b, grad_b = grad_out * a
       - `Div`: grad_a = grad_out / b, grad_b = -grad_out * a / b²
    5. Accumulate into `GradStore` for each input
  - [ ] Gradient check EVERY op: `(f(x+ε) - f(x-ε)) / 2ε` with `ε=1e-5`
- [ ] Block 12 — Backward for matmul
  - [ ] `dL/dA = dL/dC @ B^T`
  - [ ] `dL/dB = A^T @ dL/dC`
  - [ ] Needs transpose and matmul ops
  - [ ] Gradient check
- [ ] Block 13 — Backward for reductions
  - [ ] `sum` backward: broadcast scalar grad to input shape
  - [ ] `mean` backward: same, divided by n
  - [ ] Gradient check
- [ ] Block 14 — Activations with backward
  - [ ] `relu(x)` forward: `max(0, x)`, backward: `1 if x > 0 else 0`
  - [ ] `sigmoid(x)` forward: `1/(1+e^-x)`, backward: `y*(1-y)`
  - [ ] Both registered in tape
  - [ ] Gradient check both

### MILESTONE 6: Neural Network Building Blocks

- [ ] Block 15 — `Parameter`
  - [ ] `Parameter { value: Variable }`
  - [ ] Stores a leaf variable (weight or bias)
- [ ] Block 16 — `Module` trait
  - [ ] `fn forward(&self, x: &Variable, tape: &mut Tape) -> Variable`
  - [ ] `fn parameters(&self) -> Vec<&Parameter>`
- [ ] Block 17 — `Linear` layer
  - [ ] `weight: Parameter` shape [out_features, in_features]
  - [ ] `bias: Option<Parameter>` shape [out_features]
  - [ ] Forward: `output = input @ weight^T + bias`
  - [ ] Xavier/Glorot initialization
- [ ] Block 18 — Activations
  - [ ] `ReLU::forward(x, tape) -> Variable`
  - [ ] `Sigmoid::forward(x, tape) -> Variable`
  - [ ] Both implement `Module`
- [ ] Block 19 — `MSE` loss
  - [ ] Forward: `mean((pred - target)^2)`
  - [ ] Built from tracked ops — backward is automatic
  - [ ] Returns scalar Variable
- [ ] Block 20 — `Sequential` container
  - [ ] Holds `Vec<Box<dyn Module>>`
  - [ ] `forward` chains all layers
  - [ ] `parameters` collects from all layers

### MILESTONE 7: Training

- [ ] Block 21 — `SGD` optimizer
  - [ ] `SGD { lr: f32 }`
  - [ ] `fn step(params, grads)` — `param -= lr * grad`
  - [ ] `fn zero_grad(grads)`
- [ ] Block 22 — XOR training
  - [ ] Model: Linear(2→4) + ReLU + Linear(4→1) + Sigmoid
  - [ ] Loss: MSE
  - [ ] 5000 epochs, lr=0.1
  - [ ] Verify outputs: [0,0]→~0, [0,1]→~1, [1,0]→~1, [1,1]→~0

### MILESTONE 8: Production Features

- [ ] Block 23 — Batched matmul: [B, M, K] @ [B, K, N] = [B, M, N]
- [ ] Block 24 — Broadcasting (stride-0 trick)
- [ ] Block 25 — `CrossEntropyLoss` + `Softmax`
- [ ] Block 26 — MNIST training (28×28 → 10 classes)
- [ ] Block 27 — `Adam` optimizer
- [ ] Block 28 — Parallel kernels (Rayon / par_iter)
- [ ] Block 29 — `Conv2D` layer
- [ ] Block 30 — `Dropout` + `BatchNorm`

---

## Gradient Check Template

For every operation with a backward pass, write this test:

```rust
fn gradient_check_add() {
    let a = Tensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap();
    let b = Tensor::from_vec(vec![3.0, 4.0], vec![2]).unwrap();
    let epsilon = 1e-4f32;

    // Perturb a[0] up
    let mut a_plus = a.clone(); a_plus.get_mut(&[0]).unwrap() += epsilon;
    let c_plus = ops::add(&a_plus, &b).unwrap();

    // Perturb a[0] down
    let mut a_minus = a.clone(); a_minus.get_mut(&[0]).unwrap() -= epsilon;
    let c_minus = ops::add(&a_minus, &b).unwrap();

    let numerical_grad = (c_plus.data()[0] - c_minus.data()[0]) / (2.0 * epsilon);

    // Now compute with autograd
    let tape = ...;
    let analytical_grad = backward(tape, loss_id).get(a_id)[0];

    assert!((analytical_grad - numerical_grad).abs() < 1e-3);
}
```

## Ground Rules

1. **Paper first, code second.** Do the math by hand before touching the keyboard.
2. **Gradient check everything.** Every backward op gets `(f(x+ε)-f(x-ε))/2ε`.
3. **Test tiny, then scale.** Scalar [1] → vector [n] → matrix [m,n] → batch [b,m,n].
4. **f32 for NN code.** Keep `Tensor<T>` for foundation blocks, use `Tensor<f32>` for autograd+.
5. **One block at a time.** Test it. Only move on when solid.
6. **Let the borrow checker teach you.** If Rust fights, your ownership model is probably wrong.
