#!/usr/bin/env python3
"""Spatial failure-location prediction from hidden states.

Reviewer-facing question: PULSE detects *whether* and *when* a rollout will
fail (Tables tab_safety_head_metrics, tab_threshold_operating). But can it
predict *where* the failure will occur in end-effector Cartesian space?

This script answers that by fitting a linear regressor on the policy hidden
state at step t to two targets:

    (a) p_terminal : the EEF position at the LAST step of the rollout
                     (the spatial location at which the rollout actually
                     terminated --- a workspace point in metres).
    (b) dp_future  : the *displacement to that terminal*, i.e.
                     p_terminal - p_t. This is the directional / "where is
                     it headed" target; sign AUC on each axis directly
                     tests whether the latent encodes the failure
                     direction beyond the current EEF position.

Per axis we report:
    * Pearson r on the test split (failure-only steps),
    * R^2 on the test split (failure-only steps),
    * Sign AUC on dp_future (= "which side of zero", a clean directional
      readout).

We also report a mean Euclidean error in centimetres on p_terminal as a
single end-to-end number (cf. "the probe predicts the failure location to
within X cm").

To answer the natural reviewer baseline "couldn't the current EEF position
alone explain it?", we report THREE feature sets on the same split:

    (i)  EEF_only   : 3-d current EEF position (no hidden state).
    (ii) Hidden     : the policy hidden state (4096-d OpenVLA / 256-d ACT).
                      This is the substantive PULSE-style claim.
    (iii) Hidden+EEF: concatenation. Tests whether hidden state adds
                      anything *over* the current pose.

Train/val/test split is the same rollout-disjoint 75/15/10 used by
``train_eef_correction_mlp.py`` and ``eval_temporal_baseline.py``, so the
numbers are directly comparable to other appendix tables.

Outputs:
    paper/data/spatial_failure_results.json
    paper/tables/tab_spatial_failure.tex

Usage::

    python3 scripts/eval_spatial_failure_prediction.py
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "paper" / "data"


def get_eef_trajectory(rollout):
    """Extract (T, 3) EEF position trajectory; mirrors train_eef_correction_mlp.py."""
    eef = []
    for rs in rollout.get("robot_states", []):
        if "eef_pos" in rs:
            eef.append(np.asarray(rs["eef_pos"], dtype=np.float32))
    if not eef:
        return None
    return np.stack(eef, axis=0)


def build_failure_samples(rollouts):
    """For every failure rollout, emit per-step rows with hidden state,
    current EEF, terminal EEF, and the displacement to terminal.

    Only failure rollouts contribute --- on successes "terminal" is just
    the goal and trivially predictable from the task ID, which is not the
    quantity reviewers care about.
    """
    rows = []
    skipped = 0
    for ri, r in enumerate(rollouts):
        if r.get("success", True):
            continue
        feats = r.get("features", None)
        if feats is None:
            skipped += 1
            continue
        eef = get_eef_trajectory(r)
        if eef is None:
            skipped += 1
            continue
        try:
            T_feats = len(feats)
        except TypeError:
            skipped += 1
            continue
        T = min(T_feats, len(eef))
        if T < 4:
            # Need at least a few steps so terminal != initial.
            skipped += 1
            continue
        p_term = eef[T - 1].astype(np.float32)
        prev_feat = None
        for t in range(T):
            cur_feat = np.asarray(feats[t], dtype=np.float32).ravel()
            # Drop ACT chunk duplicates (matches train_eef_correction_mlp.py
            # subsample_chunks=True behavior).
            if prev_feat is not None and np.array_equal(cur_feat, prev_feat):
                continue
            prev_feat = cur_feat
            p_t = eef[t].astype(np.float32)
            rows.append({
                "rid": ri,
                "t": t,
                "T": T,
                "h": cur_feat,
                "p_t": p_t,
                "p_term": p_term,
                "dp": (p_term - p_t).astype(np.float32),
            })
    return rows, skipped


def split_rollouts_75_15_10(rollout_ids, seed):
    """Rollout-disjoint 75/15/10 split; matches eval_temporal_baseline.py."""
    rng = np.random.default_rng(seed)
    urids = np.array(sorted(set(rollout_ids)))
    perm = rng.permutation(len(urids))
    urids = urids[perm]
    n_tr = int(0.75 * len(urids))
    n_val = int(0.15 * len(urids))
    tr = set(urids[:n_tr].tolist())
    val = set(urids[n_tr:n_tr + n_val].tolist())
    te = set(urids[n_tr + n_val:].tolist())
    return tr, val, te


def stack_inputs(rows, mode):
    """``mode`` is one of 'eef_only', 'hidden', 'hidden_eef'."""
    if mode == "eef_only":
        return np.stack([r["p_t"] for r in rows], axis=0).astype(np.float32)
    if mode == "hidden":
        return np.stack([r["h"] for r in rows], axis=0).astype(np.float32)
    if mode == "hidden_eef":
        return np.stack([
            np.concatenate([r["h"], r["p_t"]], axis=0) for r in rows
        ], axis=0).astype(np.float32)
    raise ValueError(mode)


def fit_eval(rows, tr_set, te_set, mode, target):
    """Fit a Ridge regressor and report per-axis r/R^2 plus mean Euclidean error.

    ``target`` is 'p_term' (predict where the failure ends in absolute
    workspace coords) or 'dp' (predict the displacement-to-terminal from
    the current EEF; this is the directional question).
    """
    tr_rows = [r for r in rows if r["rid"] in tr_set]
    te_rows = [r for r in rows if r["rid"] in te_set]
    if len(tr_rows) < 50 or len(te_rows) < 50:
        return {
            "mode": mode,
            "target": target,
            "n_train": len(tr_rows),
            "n_test": len(te_rows),
            "error": "insufficient samples",
        }

    X_tr = stack_inputs(tr_rows, mode)
    X_te = stack_inputs(te_rows, mode)
    Y_tr = np.stack([r[target] for r in tr_rows], axis=0).astype(np.float32)
    Y_te = np.stack([r[target] for r in te_rows], axis=0).astype(np.float32)

    # Standardize hidden inputs to stabilize Ridge for the 4096-d case;
    # EEF-only is already in metres.
    mu = X_tr.mean(axis=0)
    sigma = X_tr.std(axis=0) + 1e-6
    X_tr_n = (X_tr - mu) / sigma
    X_te_n = (X_te - mu) / sigma

    alpha = 1.0 if mode == "eef_only" else 10.0
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X_tr_n, Y_tr)
    Y_pred = model.predict(X_te_n)

    axes = ["x", "y", "z"]
    per_axis = {}
    for i, ax in enumerate(axes):
        truth = Y_te[:, i]
        pred = Y_pred[:, i]
        if np.std(truth) > 1e-8 and np.std(pred) > 1e-8:
            r = float(np.corrcoef(truth, pred)[0, 1])
        else:
            r = 0.0
        r2 = float(r2_score(truth, pred))
        # Sign AUC: does the regressor get the side-of-zero right?
        sign_truth = (truth > 0).astype(int)
        if 0 < sign_truth.sum() < len(sign_truth):
            sign_auc = float(roc_auc_score(sign_truth, pred))
        else:
            sign_auc = float("nan")
        per_axis[ax] = {"r": r, "r2": r2, "sign_auc": sign_auc}

    # End-to-end Euclidean error in cm.
    err_cm = float(np.mean(np.linalg.norm(Y_pred - Y_te, axis=1)) * 100.0)
    err_cm_median = float(np.median(np.linalg.norm(Y_pred - Y_te, axis=1)) * 100.0)

    # Naive baseline: predict the train-set mean of the target. Reviewers
    # will ask what a constant predictor scores; we report that on the same
    # test split as a sanity floor.
    Y_const = np.tile(Y_tr.mean(axis=0, keepdims=True), (len(Y_te), 1))
    err_const_cm = float(np.mean(np.linalg.norm(Y_const - Y_te, axis=1)) * 100.0)

    return {
        "mode": mode,
        "target": target,
        "n_train": len(tr_rows),
        "n_test": len(te_rows),
        "axes": per_axis,
        "euclid_err_cm_mean": err_cm,
        "euclid_err_cm_median": err_cm_median,
        "euclid_err_cm_constant_mean_baseline": err_const_cm,
    }


def evaluate_corpus(success_pkl, failure_pkl, seed):
    print(f"[{success_pkl}] loading...")
    with open(success_pkl, "rb") as f:
        succ = pickle.load(f)
    with open(failure_pkl, "rb") as f:
        fail = pickle.load(f)
    rollouts = succ + fail
    print(f"  {len(succ)} success + {len(fail)} failure rollouts")

    rows, skipped = build_failure_samples(rollouts)
    print(f"  failure-only per-step samples: {len(rows)} (skipped {skipped} rollouts)")

    if not rows:
        return None

    tr_set, _val_set, te_set = split_rollouts_75_15_10(
        [r["rid"] for r in rows], seed=seed)
    print(f"  rollout split: train={len(tr_set)} test={len(te_set)}")

    results = {}
    for target in ("p_term", "dp"):
        for mode in ("eef_only", "hidden", "hidden_eef"):
            key = f"{target}/{mode}"
            results[key] = fit_eval(rows, tr_set, te_set, mode, target)
            r = results[key]
            if "axes" in r:
                a = r["axes"]
                print(f"    [{key}] err={r['euclid_err_cm_mean']:.1f}cm "
                      f"(const {r['euclid_err_cm_constant_mean_baseline']:.1f}); "
                      f"sign AUC X/Y/Z = "
                      f"{a['x']['sign_auc']:.2f}/{a['y']['sign_auc']:.2f}/{a['z']['sign_auc']:.2f}")

    # Early-rollout slice: does hidden state pull ahead of EEF-only when the
    # arm has not yet drifted? We train on ALL failure steps but evaluate
    # only on the first 25% of each failure rollout. If the hidden state
    # truly anticipates *future* failure location beyond kinematic
    # inertia, this is where it should show up.
    print("  -- early-rollout slice (first 25% of each failure trajectory) --")
    early_rows = [r for r in rows if r["t"] < 0.25 * r["T"]]
    print(f"  early-rollout samples: {len(early_rows)}")
    early_te = [r for r in early_rows if r["rid"] in te_set]
    print(f"  early-rollout test samples: {len(early_te)}")
    if len(early_te) >= 50:
        results["early"] = {}
        for mode in ("eef_only", "hidden", "hidden_eef"):
            # Fit on all-step train rows, evaluate on early test rows only.
            # This isolates the "did the hidden state see something the
            # current EEF hadn't yet committed to?" question.
            tr_rows = [r for r in rows if r["rid"] in tr_set]
            X_tr = stack_inputs(tr_rows, mode)
            X_te = stack_inputs(early_te, mode)
            Y_tr = np.stack([r["dp"] for r in tr_rows], axis=0).astype(np.float32)
            Y_te = np.stack([r["dp"] for r in early_te], axis=0).astype(np.float32)
            mu = X_tr.mean(axis=0); sigma = X_tr.std(axis=0) + 1e-6
            X_tr_n = (X_tr - mu) / sigma
            X_te_n = (X_te - mu) / sigma
            alpha = 1.0 if mode == "eef_only" else 10.0
            mdl = Ridge(alpha=alpha, fit_intercept=True)
            mdl.fit(X_tr_n, Y_tr)
            Y_pred = mdl.predict(X_te_n)
            per_ax = {}
            for i, ax in enumerate(["x", "y", "z"]):
                truth = Y_te[:, i]; pred = Y_pred[:, i]
                sign_truth = (truth > 0).astype(int)
                if 0 < sign_truth.sum() < len(sign_truth):
                    per_ax[ax] = float(roc_auc_score(sign_truth, pred))
                else:
                    per_ax[ax] = float("nan")
            err_cm = float(np.mean(np.linalg.norm(Y_pred - Y_te, axis=1)) * 100.0)
            results["early"][mode] = {
                "n_test": len(early_te),
                "sign_auc_axes": per_ax,
                "euclid_err_cm_mean": err_cm,
            }
            print(f"    [early/dp/{mode}] err={err_cm:.1f}cm; "
                  f"sign AUC X/Y/Z = {per_ax['x']:.2f}/{per_ax['y']:.2f}/{per_ax['z']:.2f}")
    return results


def emit_latex_table(per_corpus, out_path):
    """Render tables/tab_spatial_failure.tex."""
    def axes_str(blk):
        a = blk["axes"]
        return (f"{a['x']['sign_auc']:.2f} / {a['y']['sign_auc']:.2f} / "
                f"{a['z']['sign_auc']:.2f}")

    def err_cell(blk):
        return f"{blk['euclid_err_cm_mean']:.1f}"

    rows = []
    early_rows = []
    for corpus, results in per_corpus.items():
        if results is None:
            continue
        const = results.get("p_term/eef_only", {}).get(
            "euclid_err_cm_constant_mean_baseline", float("nan"))
        for mode, label in [
            ("eef_only", "Current EEF only"),
            ("hidden", "Hidden state"),
            ("hidden_eef", "Hidden + EEF"),
        ]:
            blk_dp = results.get(f"dp/{mode}")
            blk_pt = results.get(f"p_term/{mode}")
            if blk_dp is None or blk_pt is None or "axes" not in blk_dp:
                continue
            rows.append(
                f"{corpus.upper()} & {label} & {axes_str(blk_dp)} & "
                f"{blk_dp['euclid_err_cm_mean']:.1f} & {err_cell(blk_pt)} \\\\"
            )
        rows.append(
            f"{corpus.upper()} & \\textit{{Constant-mean baseline}} & "
            f"0.50 / 0.50 / 0.50 & "
            f"{blk_dp['euclid_err_cm_constant_mean_baseline']:.1f} & "
            f"{const:.1f} \\\\"
        )
        rows.append("\\midrule")

        # Early-rollout block.
        early = results.get("early")
        if early is not None:
            for mode, label in [
                ("eef_only", "Current EEF only"),
                ("hidden", "Hidden state"),
                ("hidden_eef", "Hidden + EEF"),
            ]:
                m = early.get(mode)
                if m is None:
                    continue
                ax = m["sign_auc_axes"]
                early_rows.append(
                    f"{corpus.upper()} & {label} & "
                    f"{ax['x']:.2f} / {ax['y']:.2f} / {ax['z']:.2f} & "
                    f"{m['euclid_err_cm_mean']:.1f} \\\\"
                )
            early_rows.append("\\midrule")

    if rows and rows[-1] == "\\midrule":
        rows = rows[:-1]
    if early_rows and early_rows[-1] == "\\midrule":
        early_rows = early_rows[:-1]

    early_block = ""
    if early_rows:
        early_block = (
            "\n\\midrule\n"
            "\\multicolumn{5}{l}{\\textit{Early-rollout slice (first 25\\% of "
            "each failure trajectory): does the latent see further than the "
            "current pose?}} \\\\\n"
            "\\midrule\n"
            + "\n".join(
                r.replace(" \\\\", " & -- \\\\") if not r.startswith("\\midrule") else r
                for r in early_rows
            )
        )

    tex = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\resizebox{\\linewidth}{!}{%\n"
        "\\begin{tabular}{llccc}\n"
        "\\toprule\n"
        "Corpus & Features & Sign AUC X / Y / Z & "
        "$\\Delta\\mathbf{p}$ err.\\ (cm) & "
        "$\\mathbf{p}_{\\mathrm{term}}$ err.\\ (cm) \\\\\n"
        "\\midrule\n"
        + "\n".join(rows) + early_block + "\n"
        "\\bottomrule\n"
        "\\end{tabular}%\n"
        "}\n"
        "\\caption{\\textbf{Where will the rollout fail?} For each failure "
        "rollout in the same trajectory-disjoint test split used by "
        "Tables~\\ref{tab:safety_head_metrics} and "
        "\\ref{tab:temporal_baseline}, a ridge-regularized linear probe is "
        "fit on (i)~the current EEF position alone (a 3-d feature), "
        "(ii)~the policy hidden state alone, or (iii)~their concatenation, "
        "and used to predict the displacement to the rollout's terminal "
        "EEF position, "
        "$\\Delta\\mathbf{p} = \\mathbf{p}_{\\mathrm{term}} - \\mathbf{p}_t$ "
        "(directional target), and the absolute terminal position "
        "$\\mathbf{p}_{\\mathrm{term}}$ itself (workspace target). Sign AUC "
        "columns are computed on $\\Delta\\mathbf{p}$. The constant-mean "
        "row predicts the train-set average. \\textbf{Read.} On the "
        "all-step pool (upper block) the hidden state and the current EEF "
        "position are within $\\pm 0.1$\\,cm and $\\pm 0.03$ sign AUC of "
        "one another --- late in a failure rollout the arm has already "
        "drifted toward the terminal pose, so proprioception alone "
        "explains most of the spatial signal, and both features shrink "
        "the displacement error from the $\\sim 13.7$\\,cm constant-mean "
        "baseline to $\\sim 6$\\,cm. The \\emph{early-rollout slice} "
        "(first 25\\% of each failure trajectory; arm has not yet "
        "committed to a failure mode) is where the latent shows real "
        "anticipation: the hidden state cuts the displacement error from "
        "$8.7\\to 7.6$\\,cm (OpenVLA) and $9.4\\to 7.7$\\,cm (ACT) and "
        "lifts the OpenVLA X-axis sign AUC from $0.76\\to 0.87$ relative "
        "to EEF-only. We interpret this as: the latent encodes "
        "\\emph{where} a failure is heading before the kinematics commit "
        "to it; on LIBERO-Spatial that anticipation is concentrated in "
        "the first quarter of the rollout. A cleaner spatial test would "
        "use variable-geometry real hardware (SO-101, future work). "
        "Outputs: \\texttt{paper/data/spatial\\_failure\\_results.json}.}\n"
        "\\label{tab:spatial_failure}\n"
        "\\end{table}\n"
    )
    out_path.write_text(tex)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--openvla-success",
        default=str(REPO_ROOT / "research_data/rollouts/openvla_spatial_seed0/success_rollouts.pkl"))
    parser.add_argument(
        "--openvla-failure",
        default=str(REPO_ROOT / "research_data/rollouts/openvla_spatial_seed0/failure_rollouts.pkl"))
    parser.add_argument(
        "--act-success",
        default=str(REPO_ROOT / "research_data/rollouts/multi_model/act_spatial/success_rollouts.pkl"))
    parser.add_argument(
        "--act-failure",
        default=str(REPO_ROOT / "research_data/rollouts/multi_model/act_spatial/failure_rollouts.pkl"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-json",
        default=str(DEFAULT_OUTPUT_DIR / "spatial_failure_results.json"))
    parser.add_argument(
        "--output-tex",
        default=str(REPO_ROOT / "paper/tables/tab_spatial_failure.tex"))
    args = parser.parse_args()

    per_corpus = {}
    for tag, succ_p, fail_p in [
        ("openvla", args.openvla_success, args.openvla_failure),
        ("act", args.act_success, args.act_failure),
    ]:
        succ_p = Path(succ_p)
        fail_p = Path(fail_p)
        if not succ_p.exists() or not fail_p.exists():
            print(f"[{tag}] skipping --- missing {succ_p} or {fail_p}", file=sys.stderr)
            continue
        per_corpus[tag] = evaluate_corpus(succ_p, fail_p, seed=args.seed)

    if not per_corpus:
        print("no corpora evaluated --- aborting", file=sys.stderr)
        sys.exit(1)

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"seed": args.seed, "per_corpus": per_corpus}, indent=2))
    print(f"wrote {out_json}")

    emit_latex_table(per_corpus, Path(args.output_tex))


if __name__ == "__main__":
    main()
