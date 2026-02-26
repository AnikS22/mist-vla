#!/usr/bin/env python3
"""Jetson bring-up smoke test for latent safety steering runtime."""

import time
import numpy as np
import torch
import torch.nn as nn


class EEFCorrectionMLP(nn.Module):
    HIDDEN_DIM = 256

    def __init__(self, input_dim=256):
        super().__init__()
        h = self.HIDDEN_DIM
        self.input_norm = nn.LayerNorm(input_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(h, h // 2),
            nn.LayerNorm(h // 2),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(h // 2, h // 4),
            nn.LayerNorm(h // 4),
            nn.GELU(),
            nn.Dropout(0.0),
        )
        feat = h // 4
        self.fail_head = nn.Linear(feat, 1)
        self.ttf_head = nn.Linear(feat, 1)
        self.correction_head = nn.Linear(feat, 3)

    def forward(self, x):
        z = self.encoder(self.input_norm(x))
        return {
            "will_fail": self.fail_head(z).squeeze(-1),
            "ttf": self.ttf_head(z).squeeze(-1),
            "correction": self.correction_head(z),
        }


def main():
    print("=== Jetson Latent Steering Smoke Test ===")
    print("torch_version:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    print("cuda_device_count:", torch.cuda.device_count())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("cuda_device_0:", torch.cuda.get_device_name(0))

    np.random.seed(42)
    torch.manual_seed(42)

    # ACT-like latent dimensionality smoke path.
    input_dim = 256
    model = EEFCorrectionMLP(input_dim=input_dim).to(device).eval()
    x = torch.randn(1, input_dim, device=device)

    # Warmup
    for _ in range(100):
        with torch.no_grad():
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times_ms = []
    for _ in range(500):
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(x)
            _ = torch.sigmoid(out["will_fail"])
        if device.type == "cuda":
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    p50 = float(np.percentile(times_ms, 50))
    p95 = float(np.percentile(times_ms, 95))
    mean = float(np.mean(times_ms))

    print("forward_mean_ms:", round(mean, 4))
    print("forward_p50_ms:", round(p50, 4))
    print("forward_p95_ms:", round(p95, 4))
    print("sample_fail_prob:", round(float(torch.sigmoid(out["will_fail"]).item()), 4))
    print("sample_ttf:", round(float(out["ttf"].item()), 4))
    print("sample_corr_xyz:", [round(v, 5) for v in out["correction"].detach().cpu().numpy()[0].tolist()])
    print("SMOKE_TEST_PASS")


if __name__ == "__main__":
    main()
