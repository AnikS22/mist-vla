#!/usr/bin/env python3
"""
Train a simple MLP to predict time-to-failure (and fail-within-k).
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score


class TimeFailureDataset(Dataset):
    def __init__(self, samples):
        self.hidden = np.array([s["hidden_state"] for s in samples])
        self.time_to_failure = np.array([s.get("time_to_failure", -1) for s in samples])
        self.fail_within_k = np.array([s.get("fail_within_k", 0) for s in samples])

        self.hidden_mean = self.hidden.mean(axis=0)
        self.hidden_std = self.hidden.std(axis=0) + 1e-8

    def __len__(self):
        return len(self.hidden)

    def __getitem__(self, idx):
        h = (self.hidden[idx] - self.hidden_mean) / self.hidden_std
        y = self.fail_within_k[idx]
        return torch.FloatTensor(h), torch.FloatTensor([y])


class TimeFailureMLP(nn.Module):
    def __init__(self, input_dim=4096, hidden_dims=(512, 256)):
        super().__init__()
        layers = []
        dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(dim, h), nn.ReLU(), nn.Dropout(0.1)])
            dim = h
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    with open(args.data, "rb") as f:
        data = pickle.load(f)
    samples = data["dataset"]

    dataset = TimeFailureDataset(samples)
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = TimeFailureMLP(input_dim=dataset.hidden.shape[1]).to(device)
    opt = AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        all_logits, all_y = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                logits = model(x).cpu().numpy().flatten()
                all_logits.append(logits)
                all_y.append(y.numpy().flatten())
        preds = np.concatenate(all_logits)
        targets = np.concatenate(all_y)
        if len(np.unique(targets)) > 1:
            auc = roc_auc_score(targets, preds)
        else:
            auc = 0.5
        best_auc = max(best_auc, auc)
        print(f"Epoch {epoch}: val AUC={auc:.4f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "best_auc": best_auc}, out_path)
    print(f"Saved model -> {out_path} (best AUC={best_auc:.4f})")


if __name__ == "__main__":
    main()
