import json
import statistics
import time
from pathlib import Path

import numpy as np
import sklearn
import torch
import torch.nn as nn
from sklearn.datasets import make_moons

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

torch.set_num_threads(1)
torch.manual_seed(42)
np.random.seed(42)


# ============================================================
# TASKS 1-3: SCALAR AUTOMATIC DIFFERENTIATION ENGINE
# ============================================================

class Value:
    """A scalar value and its dynamically constructed computation graph."""

    def __init__(self, data, parents=(), backward=None, label=""):
        self.data = float(data)
        self.grad = 0.0
        self.parents = tuple(parents)
        self._backward = backward or (lambda: None)
        self.label = label

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))

        def backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = backward
        return out

    __radd__ = __add__

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))

        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = backward
        return out

    __rmul__ = __mul__

    def __pow__(self, power):
        if not isinstance(power, (int, float)):
            raise TypeError("Power must be a number")

        out = Value(self.data ** power, (self,))

        def backward():
            self.grad += power * (self.data ** (power - 1)) * out.grad

        out._backward = backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,))

        def backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = backward
        return out

    def sum(self):
        """Scalar sum: returns an equivalent graph-connected scalar."""
        out = Value(self.data, (self,))
        out._backward = lambda: setattr(self, "grad", self.grad + out.grad)
        return out

    def backward(self):
        """Topological reverse-mode backpropagation."""
        topo = []
        visited = set()

        def build(node):
            if node not in visited:
                visited.add(node)
                for parent in node.parents:
                    build(parent)
                topo.append(node)

        build(self)

        # Reset all gradients so repeated backward calls are independent.
        for node in topo:
            node.grad = 0.0

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Value(data={self.data:.6f}, grad={self.grad:.6f})"


def custom_expression(a, b, c):
    return ((a * b + c) ** 2).relu().sum()


def verify_gradients():
    # Float64 reference values.
    values = [1.5, 2.0, 0.5]

    a, b, c = [Value(v) for v in values]
    output = custom_expression(a, b, c)
    output.backward()
    custom_grads = np.array([a.grad, b.grad, c.grad], dtype=np.float64)

    # PyTorch autograd reference.
    ta, tb, tc = [
        torch.tensor(v, dtype=torch.float64, requires_grad=True)
        for v in values
    ]
    tout = ((ta * tb + tc) ** 2).relu()
    tout.backward()
    torch_grads = np.array(
        [ta.grad.item(), tb.grad.item(), tc.grad.item()],
        dtype=np.float64,
    )

    # Central finite differences.
    def function(x):
        aa, bb, cc = x
        return max(0.0, (aa * bb + cc) ** 2)

    eps = 1e-6
    numerical_grads = []
    for i in range(3):
        plus = values.copy()
        minus = values.copy()
        plus[i] += eps
        minus[i] -= eps
        numerical_grads.append(
            (function(plus) - function(minus)) / (2 * eps)
        )
    numerical_grads = np.array(numerical_grads, dtype=np.float64)

    diff_torch = float(np.max(np.abs(custom_grads - torch_grads)))
    diff_numeric = float(np.max(np.abs(custom_grads - numerical_grads)))

    # Explicit shared-node gradient accumulation test.
    x = Value(3.0)
    shared_output = x * x + x
    shared_output.backward()
    accumulation_pass = abs(x.grad - 7.0) < 1e-12

    result = {
        "expression": "((a*b+c)**2).relu().sum()",
        "input_values": values,
        "output": output.data,
        "custom_gradients": custom_grads.tolist(),
        "pytorch_gradients": torch_grads.tolist(),
        "finite_difference_gradients": numerical_grads.tolist(),
        "max_abs_difference_vs_pytorch": diff_torch,
        "max_abs_difference_vs_finite_difference": diff_numeric,
        "tolerance": 1e-6,
        "shared_node_accumulation": {
            "expected_gradient": 7.0,
            "actual_gradient": x.grad,
            "passed": accumulation_pass,
        },
        "passed": (
            diff_torch < 1e-6
            and diff_numeric < 1e-6
            and accumulation_pass
        ),
    }

    save_json("gradient_verification.json", result)
    print("\n=== TASK 4: GRADIENT VERIFICATION ===")
    print(json.dumps(result, indent=2))
    print("GRADIENT VERIFICATION:", "PASS" if result["passed"] else "FAIL")
    return result


# ============================================================
# TASK 5: make_moons TRAINING AND BENCHMARK
# ============================================================

class NumpyNetwork:
    """2-8-1 neural network with manual vectorized backpropagation."""

    def __init__(self, seed=42):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, 0.5, (2, 8))
        self.b1 = np.zeros((1, 8))
        self.w2 = rng.normal(0, 0.5, (8, 1))
        self.b2 = np.zeros((1, 1))

    @staticmethod
    def sigmoid(x):
        x = np.clip(x, -40, 40)
        return 1.0 / (1.0 + np.exp(-x))

    def train(self, X, y, epochs=120, lr=0.1):
        n = len(X)

        for _ in range(epochs):
            z1 = X @ self.w1 + self.b1
            h = np.maximum(0, z1)
            z2 = h @ self.w2 + self.b2
            p = self.sigmoid(z2)

            loss = -np.mean(
                y * np.log(p + 1e-12)
                + (1 - y) * np.log(1 - p + 1e-12)
            )

            dz2 = (p - y) / n
            dw2 = h.T @ dz2
            db2 = np.sum(dz2, axis=0, keepdims=True)

            dh = dz2 @ self.w2.T
            dz1 = dh * (z1 > 0)
            dw1 = X.T @ dz1
            db1 = np.sum(dz1, axis=0, keepdims=True)

            self.w1 -= lr * dw1
            self.b1 -= lr * db1
            self.w2 -= lr * dw2
            self.b2 -= lr * db2

        return float(loss)


class TorchNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc2(torch.relu(self.fc1(x))))


def benchmark_training():
    X, y = make_moons(n_samples=256, noise=0.2, random_state=42)
    X = X.astype(np.float64)
    y = y.reshape(-1, 1).astype(np.float64)

    TX = torch.tensor(X, dtype=torch.float64)
    TY = torch.tensor(y, dtype=torch.float64)

    epochs = 120
    runs = 5

    def numpy_run(seed):
        model = NumpyNetwork(seed)
        return model.train(X, y, epochs=epochs)

    def torch_run(seed):
        torch.manual_seed(seed)
        model = TorchNetwork().double()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = nn.BCELoss()

        model.train()
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            predictions = model(TX)
            loss = criterion(predictions, TY)
            loss.backward()
            optimizer.step()

        return float(loss.item())

    # Warm-up is excluded from all reported timings.
    numpy_run(100)
    torch_run(100)

    numpy_times, torch_times = [], []
    numpy_losses, torch_losses = [], []

    for seed in range(runs):
        start = time.perf_counter()
        nl = numpy_run(seed)
        numpy_times.append(time.perf_counter() - start)
        numpy_losses.append(nl)

        start = time.perf_counter()
        tl = torch_run(seed)
        torch_times.append(time.perf_counter() - start)
        torch_losses.append(tl)

    result = {
        "dataset": "make_moons",
        "samples": len(X),
        "input_features": 2,
        "architecture": "2-8-1",
        "activation": "ReLU hidden, sigmoid output",
        "loss": "Binary cross-entropy",
        "epochs": epochs,
        "runs": runs,
        "timing": "Full-batch training including forward, loss, backward and optimizer update; warm-up excluded",
        "numpy_median_seconds": statistics.median(numpy_times),
        "pytorch_median_seconds": statistics.median(torch_times),
        "numpy_run_seconds": numpy_times,
        "pytorch_run_seconds": torch_times,
        "numpy_final_losses": numpy_losses,
        "pytorch_final_losses": torch_losses,
        "numpy_median_final_loss": statistics.median(numpy_losses),
        "pytorch_median_final_loss": statistics.median(torch_losses),
        "device": "CPU",
        "note": "Timings depend on hardware, software versions, threading and workload.",
    }

    save_json("benchmark_results.json", result)
    print("\n=== TASK 5: TRAINING BENCHMARK ===")
    print(json.dumps(result, indent=2))
    return result


# ============================================================
# TASK 6A: COMPUTATION GRAPH INSPECTION
# ============================================================

def inspect_graph():
    model = TorchNetwork().double()
    x = torch.tensor([[0.2, 0.7]], dtype=torch.float64)
    output = model(x)

    nodes = []
    seen = set()

    def visit(fn):
        if fn is None or fn in seen:
            return
        seen.add(fn)
        nodes.append(type(fn).__name__)
        for next_fn, _ in fn.next_functions:
            visit(next_fn)

    visit(output.grad_fn)

    result = {
        "input_shape": list(x.shape),
        "output_shape": list(output.shape),
        "output_grad_fn": type(output.grad_fn).__name__,
        "graph_nodes": nodes,
        "graph_node_count": len(nodes),
        "description": "Autograd nodes represent differentiable operations and their connections.",
    }

    print("\n=== TASK 6A: PYTORCH AUTOGRAD GRAPH ===")
    print(json.dumps(result, indent=2))
    return result


# ============================================================
# TASK 6B: EAGER VERSUS TRACED EXECUTION
# ============================================================

def benchmark_execution_modes():
    torch.manual_seed(42)
    model = TorchNetwork().eval()
    x = torch.randn(256, 2)

    # Tracing setup occurs outside the measured intervals.
    traced = torch.jit.trace(model, x)

    def measure(fn, warmup=30, loops=200):
        with torch.no_grad():
            for _ in range(warmup):
                fn()
            timings = []
            for _ in range(5):
                start = time.perf_counter()
                for _ in range(loops):
                    fn()
                timings.append((time.perf_counter() - start) / loops)
        return statistics.median(timings)

    eager_time = measure(lambda: model(x))
    traced_time = measure(lambda: traced(x))

    result = {
        "method": "torch.jit.trace",
        "fallback_reason": "Tracing provides a practical graph-based alternative without relying on torch.compile/Inductor support on this Windows CPU setup.",
        "eager_median_seconds_per_call": eager_time,
        "traced_median_seconds_per_call": traced_time,
        "eager_runs": 5,
        "traced_runs": 5,
        "warmup_iterations": 30,
        "measured_iterations_per_run": 200,
        "trace_setup_excluded": True,
        "device": "CPU",
    }

    save_json("execution_results.json", result)
    print("\n=== TASK 6B: EXECUTION MODES ===")
    print(json.dumps(result, indent=2))
    return result


# ============================================================
# TASK 6C: TENSOR CONTIGUITY
# ============================================================

def benchmark_contiguity():
    base = torch.randn(256, 512)
    non_contiguous = base.transpose(0, 1)
    contiguous = non_contiguous.contiguous()

    assert contiguous.shape == non_contiguous.shape
    assert contiguous.is_contiguous()
    assert not non_contiguous.is_contiguous()

    def operation(tensor):
        return tensor * 1.5 + 0.25

    def measure(tensor, loops=200):
        for _ in range(30):
            operation(tensor)

        timings = []
        for _ in range(5):
            start = time.perf_counter()
            for _ in range(loops):
                operation(tensor)
            timings.append((time.perf_counter() - start) / loops)
        return statistics.median(timings)

    contiguous_time = measure(contiguous)
    non_contiguous_time = measure(non_contiguous)

    start = time.perf_counter()
    copied = non_contiguous.contiguous()
    copy_time = time.perf_counter() - start

    result = {
        "logical_shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "contiguous_tensor_is_contiguous": contiguous.is_contiguous(),
        "transposed_tensor_is_contiguous": non_contiguous.is_contiguous(),
        "contiguous_stride": list(contiguous.stride()),
        "non_contiguous_stride": list(non_contiguous.stride()),
        "contiguous_median_seconds": contiguous_time,
        "non_contiguous_median_seconds": non_contiguous_time,
        "one_contiguous_copy_seconds": copy_time,
        "copy_result_is_contiguous": copied.is_contiguous(),
        "runs": 5,
        "warmup_iterations": 30,
        "iterations_per_run": 200,
        "operation": "elementwise multiply by 1.5 then add 0.25",
        "note": "Copy cost must be considered; contiguous conversion is worthwhile when later operations save more than the copy cost.",
    }

    save_json("contiguity_results.json", result)
    print("\n=== TASK 6C: TENSOR CONTIGUITY ===")
    print(json.dumps(result, indent=2))
    return result


# ============================================================
# RESULTS AND DOCUMENTATION
# ============================================================

def save_json(filename, data):
    with (RESULTS / filename).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_readme(gradient, training, graph, execution, contiguity):
    passed = gradient["passed"]

    content = f"""# Lab 5 — Frameworks and Automatic Differentiation

## Overview

This project implements a scalar reverse-mode automatic differentiation engine,
validates its gradients, trains a small neural network on make_moons, and
investigates PyTorch computation graphs, execution modes and tensor contiguity.

## Environment

- Python 3.12
- NumPy {np.__version__}
- Scikit-learn {sklearn.__version__}
- PyTorch {torch.__version__}
- Device: CPU

## Installation and execution

Create the environment and install requirements:

```powershell
py -3.12 -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
```

Run the complete lab:

```powershell
.\\.venv\\Scripts\\python.exe lab5.py
```

## Project structure

- `lab5.py`: implementation and experiments.
- `requirements.txt`: dependencies.
- `results/gradient_verification.json`: gradient comparisons.
- `results/benchmark_results.json`: training benchmark.
- `results/execution_results.json`: eager/traced execution.
- `results/contiguity_results.json`: tensor layout benchmark.
- `results/lab5_summary.json`: overall result summary.

## Custom Value engine

Each Value stores scalar data, a gradient, parent nodes and a local backward
function. Arithmetic operations create new nodes and connect the graph.
Backpropagation first constructs a topological ordering, then visits it in
reverse. Gradients are accumulated with `+=`, which is essential when a node
feeds multiple downstream operations.

Implemented operations: addition, multiplication, numeric powers, ReLU and
scalar sum.

## Gradient verification

Expression: `((a*b+c)**2).relu().sum()`

Inputs: {gradient["input_values"]}

Custom gradients: {gradient["custom_gradients"]}

PyTorch gradients: {gradient["pytorch_gradients"]}

Finite-difference gradients: {gradient["finite_difference_gradients"]}

Maximum absolute difference versus PyTorch:
{gradient["max_abs_difference_vs_pytorch"]:.12g}

Maximum absolute difference versus finite differences:
{gradient["max_abs_difference_vs_finite_difference"]:.12g}

Shared-node accumulation test: {gradient["shared_node_accumulation"]["passed"]}

Gradient verification: **{"PASS" if passed else "FAIL"}**

Required tolerance: strictly less than 1e-6.

## make_moons training benchmark

Dataset: {training["samples"]} samples, 2 input features.

Architecture: 2-8-1, ReLU hidden activation, sigmoid output and binary
cross-entropy. Both implementations use full-batch training, SGD-style
updates, 120 epochs and five timed runs. One warm-up run is excluded.

| Implementation | Median training time | Median final loss |
|---|---:|---:|
| NumPy manual backpropagation | {training["numpy_median_seconds"]:.6f} s | {training["numpy_median_final_loss"]:.8f} |
| PyTorch autograd | {training["pytorch_median_seconds"]:.6f} s | {training["pytorch_median_final_loss"]:.8f} |

This comparison measures complete training steps, including forward pass,
loss calculation, backward pass and parameter updates. It is not a universal
framework ranking. Results depend on hardware, libraries, threading and
workload size.

## PyTorch computation graph

The model's output `grad_fn` is `{graph["output_grad_fn"]}`.
The inspected graph contains {graph["graph_node_count"]} reachable autograd
nodes. The graph node types are saved in the console output.

## Eager versus traced execution

Method: `{execution["method"]}`.

Eager median: {execution["eager_median_seconds_per_call"]:.9f} seconds/call.

Traced median: {execution["traced_median_seconds_per_call"]:.9f} seconds/call.

Tracing setup is excluded from steady-state measurements. Tracing can reduce
Python overhead for a stable model, but may not improve every workload.
Compilation is most useful when its one-time setup cost is amortized across
many executions and the workload is sufficiently large.

## Tensor contiguity

Logical shape: {contiguity["logical_shape"]}

Dtype: {contiguity["dtype"]}

Contiguous view: {contiguity["contiguous_tensor_is_contiguous"]}

Transposed view: {contiguity["transposed_tensor_is_contiguous"]}

Contiguous operation median: {contiguity["contiguous_median_seconds"]:.9f} s.

Non-contiguous operation median: {contiguity["non_contiguous_median_seconds"]:.9f} s.

One contiguous copy: {contiguity["one_contiguous_copy_seconds"]:.9f} s.

A transpose can create a non-contiguous view without copying the underlying
storage. Some operations handle strides efficiently; others may be slower or
require internal copies. Calling `.contiguous()` explicitly allocates and
copies data. It is worthwhile when the cost is recovered by repeated
downstream operations or when an operation requires contiguous memory.

## Limitations and caveats

- The custom engine supports scalar values and the required operations, not
  general tensor broadcasting.
- Numerical timings are machine- and workload-dependent.
- CPU-only PyTorch was used; GPU performance is not measured.
- Tracing setup is excluded from steady-state timing.
- Microbenchmarks are sensitive to operating-system scheduling and background
  activity.

## Result files

All JSON files are generated by actual execution of `lab5.py`.
Do not treat timing values as transferable to other machines.
"""
    (ROOT / "README.md").write_text(content, encoding="utf-8")


def main():
    print("=" * 65)
    print("LAB 5: FRAMEWORKS AND AUTOMATIC DIFFERENTIATION")
    print("=" * 65)
    print("Python-compatible environment:", torch.__version__)
    print("Device: CPU")

    gradient = verify_gradients()
    training = benchmark_training()
    graph = inspect_graph()
    execution = benchmark_execution_modes()
    contiguity = benchmark_contiguity()

    summary = {
        "lab": "Lab 5 - Frameworks and Automatic Differentiation",
        "gradient_verification_passed": gradient["passed"],
        "gradient_max_difference_pytorch": gradient["max_abs_difference_vs_pytorch"],
        "gradient_max_difference_numerical": gradient["max_abs_difference_vs_finite_difference"],
        "shared_gradient_accumulation_passed": gradient["shared_node_accumulation"]["passed"],
        "training_benchmark_completed": True,
        "graph_inspection_completed": bool(graph["graph_node_count"]),
        "execution_comparison_completed": True,
        "contiguity_comparison_completed": True,
        "all_required_experiments_completed": True,
    }

    save_json("lab5_summary.json", summary)
    write_readme(gradient, training, graph, execution, contiguity)

    print("\n" + "=" * 65)
    print("FINAL VALIDATION")
    print("=" * 65)
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nREADME and JSON results saved successfully.")
    if not gradient["passed"]:
        raise RuntimeError("Gradient verification failed.")
    print("\nLAB 5 EXECUTION COMPLETED.")


if __name__ == "__main__":
    main()
