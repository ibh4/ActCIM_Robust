from __future__ import annotations

import time
import numpy as np
import torch


def measure_throughput(model, input_shape, device, num_batches=100):
    model = model.to(device)
    model.eval()

    inputs = torch.randn(*input_shape, device=device)

    with torch.no_grad():
        for _ in range(10):
            _ = model(inputs)

    if device.type == "cuda":
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_batches):
                _ = model(inputs)
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    else:
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_batches):
                _ = model(inputs)
        elapsed = time.perf_counter() - start

    total_samples = input_shape[0] * num_batches
    throughput = total_samples / elapsed
    return throughput


def measure_latency(model, input_shape, device, num_warmup=10, num_iterations=100):
    model = model.to(device)
    model.eval()

    inputs = torch.randn(*input_shape, device=device)

    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(inputs)

    latencies = []
    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        for _ in range(num_iterations):
            starter.record()
            with torch.no_grad():
                _ = model(inputs)
            ender.record()
            torch.cuda.synchronize()
            latencies.append(starter.elapsed_time(ender))
        avg_latency = np.mean(latencies)
    else:
        for _ in range(num_iterations):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(inputs)
            latencies.append((time.perf_counter() - start) * 1000)
        avg_latency = np.mean(latencies)

    return {
        "mean_latency_ms": float(avg_latency),
        "std_latency_ms": float(np.std(latencies)),
        "min_latency_ms": float(np.min(latencies)),
        "max_latency_ms": float(np.max(latencies)),
    }


def get_model_size_mb(model):
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size
    return total_size / (1024 ** 2)


def count_parameters(model, trainable_only=True):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_peak_gpu_memory():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)
