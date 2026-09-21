#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score, average_precision_score,
                             roc_curve, precision_recall_curve)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize

DEFAULT_DATA  = "/projectsp/f_miarc_1/Hypertension/Sritama/batch_outputs/model_training/training_data.csv"
DEFAULT_OUT   = "/projectsp/f_miarc_1/Hypertension/Sritama/batch_outputs/model_training"
LABEL_COL     = "cluster"
#LABEL_COL     = "pam_gower_cluster"
ID_COL        = "PAT_MRN_ID_ENCRYPT"



def prepare_features(df: pd.DataFrame, label_col: str, id_col: str):
   
    if label_col not in df.columns:
        sys.exit(f"ERROR: label column '{label_col}' not in data. "
                 f"Columns: {list(df.columns)}")

    y_raw = df[label_col]
    drop_cols = [label_col]
    if id_col in df.columns:
        drop_cols.append(id_col)
    else:
        print(f"NOTE: id column '{id_col}' not found; nothing dropped as id.")

    X = df.drop(columns=drop_cols)

    suspects = [c for c in X.columns
                if X[c].nunique(dropna=False) == len(X)
                and not pd.api.types.is_float_dtype(X[c])]
    if suspects:
        print(f"WARNING: these columns are unique per row and may be "
              f"identifiers/leakage -> review or drop: {suspects}")

    keep = y_raw.notna()
    if (~keep).any():
        print(f"NOTE: dropping {int((~keep).sum())} rows with a missing label.")
    return X.loc[keep].reset_index(drop=True), y_raw.loc[keep].reset_index(drop=True)


def to_categorical(X: pd.DataFrame):
    """Convert object/string columns to pandas 'category' dtype so XGBoost's
    native categorical support can consume them. Returns (X, cat_col_names)."""
    X = X.copy()
    cat_cols = list(X.select_dtypes(include=["object", "category", "string"]).columns)
    for c in cat_cols:
        X[c] = X[c].astype("category")
    return X, cat_cols


def encode_labels(y_raw: pd.Series):
    """Map arbitrary cluster labels (ints or strings) to 0..k-1 for XGBoost."""
    le = LabelEncoder()
    y = le.fit_transform(y_raw.astype(str))
    return y, le


def safe_stratify(y):
    """Return y for stratification only if every class has >=2 samples."""
    counts = pd.Series(y).value_counts()
    return y if counts.min() >= 2 else None

def compute_metrics(y_true, y_pred, class_names):
    """Bundle the standard classification metrics into a dict."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred,
                                                 average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred,
                                           average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred,
                                   average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred,
                                      average="weighted", zero_division=0)),
        "per_class": classification_report(
            y_true, y_pred, target_names=[str(c) for c in class_names],
            output_dict=True, zero_division=0),
    }



def compute_auc_metrics(y_true, y_prob, class_names):
    
    n_classes = len(class_names)

    if n_classes == 2:
        y_bin = np.column_stack([(y_true == 0).astype(int),
                                 (y_true == 1).astype(int)])
    else:
        y_bin = label_binarize(y_true, classes=np.arange(n_classes))

    per_class = {}
    for i, name in enumerate(class_names):
     
        if np.unique(y_bin[:, i]).size < 2:
            auroc = float("nan")
            auprc = float("nan")
        else:
            auroc = float(roc_auc_score(y_bin[:, i], y_prob[:, i]))
            auprc = float(average_precision_score(y_bin[:, i], y_prob[:, i]))

        per_class[str(name)] = {
            "auroc": auroc,
            "auprc": auprc,
            "prevalence": float(y_bin[:, i].mean()),
        }

    valid_roc = [d["auroc"] for d in per_class.values()
                 if np.isfinite(d["auroc"])]
    valid_pr = [d["auprc"] for d in per_class.values()
                if np.isfinite(d["auprc"])]

    return {
        "auroc_macro_ovr": float(np.mean(valid_roc)) if valid_roc else float("nan"),
        "auprc_macro_ovr": float(np.mean(valid_pr)) if valid_pr else float("nan"),
        "auroc_micro_ovr": float(roc_auc_score(y_bin.ravel(), y_prob.ravel())),
        "auprc_micro_ovr": float(
            average_precision_score(y_bin.ravel(), y_prob.ravel())
        ),
        "per_class_auc": per_class,
        "_y_bin": y_bin, 
    }


def format_metrics(m, cv=None):
    lines = ["=== Held-out test metrics ===",
             f"accuracy           : {m['accuracy']:.4f}",
             f"balanced accuracy  : {m['balanced_accuracy']:.4f}",
             f"precision (macro)  : {m['precision_macro']:.4f}",
             f"recall (macro)     : {m['recall_macro']:.4f}",
             f"F1 (macro)         : {m['f1_macro']:.4f}",
             f"F1 (weighted)      : {m['f1_weighted']:.4f}"]
    if "auroc_macro_ovr" in m:
        lines += [
            f"AUROC (macro OVR)  : {m['auroc_macro_ovr']:.4f}",
            f"AUROC (micro OVR)  : {m['auroc_micro_ovr']:.4f}",
            f"AUPRC (macro OVR)  : {m['auprc_macro_ovr']:.4f}",
            f"AUPRC (micro OVR)  : {m['auprc_micro_ovr']:.4f}",
        ]
    lines += ["", "per-class (precision / recall / f1 / support):"]
    for name, d in m["per_class"].items():
        if isinstance(d, dict) and "precision" in d:
            lines.append(f"  {name:<20} "
                         f"{d['precision']:.3f}  {d['recall']:.3f}  "
                         f"{d['f1-score']:.3f}  {int(d['support'])}")

    if "per_class_auc" in m:
        lines += ["", "per-class one-vs-rest (AUROC / AUPRC / prevalence):"]
        for name, d in m["per_class_auc"].items():
            lines.append(f"  {name:<20} "
                         f"{d['auroc']:.3f}  {d['auprc']:.3f}  "
                         f"{d['prevalence']:.3f}")

    if cv is not None:
        lines += ["", "=== Stratified cross-validation (full data) ===",
                  f"accuracy : {cv['acc_mean']:.4f} +/- {cv['acc_std']:.4f}",
                  f"F1 macro : {cv['f1_mean']:.4f} +/- {cv['f1_std']:.4f}"]
    return "\n".join(lines)


def main():
    import xgboost as xgb  

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--label-col", default=LABEL_COL)
    p.add_argument("--id-col", default=ID_COL)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--n-estimators", type=int, default=600)
    args = p.parse_args()

    df = pd.read_csv(args.data)
    X, y_raw = prepare_features(df, args.label_col, args.id_col)
    X, cat_cols = to_categorical(X)
    y, le = encode_labels(y_raw)
    n_classes = len(le.classes_)
    print(f"Loaded {len(X)} rows, {X.shape[1]} features "
          f"({len(cat_cols)} categorical), {n_classes} clusters.")

    if n_classes < 2:
        sys.exit("ERROR: need at least 2 clusters to train a classifier.")

    strat = safe_stratify(y)
    if strat is None:
        print("WARNING: a cluster has <2 members; using a non-stratified split.")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state,
        stratify=strat)


    strat_tr = safe_stratify(y_tr)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_tr, y_tr, test_size=0.15, random_state=args.random_state,
        stratify=strat_tr)

    objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
    eval_metric = "logloss" if n_classes == 2 else "mlogloss"

    model = xgb.XGBClassifier(
        objective=objective,
        eval_metric=eval_metric,
        n_estimators=args.n_estimators,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        reg_lambda=1.0,
        tree_method="hist",
        enable_categorical=True,     
        early_stopping_rounds=40,
        importance_type="gain",
        random_state=args.random_state,
        n_jobs=-1,
    )
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    print(f"Best iteration: {model.best_iteration} "
          f"(of {args.n_estimators} max).")

  
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)

    metrics = compute_metrics(y_te, y_pred, le.classes_)
    auc_metrics = compute_auc_metrics(y_te, y_prob, le.classes_)
    y_te_bin = auc_metrics.pop("_y_bin")  
    metrics.update(auc_metrics)


    cv = None
    min_class = int(pd.Series(y).value_counts().min())
    folds = min(args.cv_folds, min_class)
    if folds >= 2:
        skf = StratifiedKFold(n_splits=folds, shuffle=True,
                              random_state=args.random_state)
        accs, f1s = [], []
        for tr_i, te_i in skf.split(X, y):
            m = xgb.XGBClassifier(**model.get_params())
            m.set_params(early_stopping_rounds=None)  
            m.fit(X.iloc[tr_i], y[tr_i], verbose=False)
            pr = m.predict(X.iloc[te_i])
            accs.append(accuracy_score(y[te_i], pr))
            f1s.append(f1_score(y[te_i], pr, average="macro", zero_division=0))
        cv = {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
              "f1_mean": float(np.mean(f1s)),  "f1_std": float(np.std(f1s))}
    else:
        print(f"NOTE: smallest cluster has {min_class} members; "
              f"skipping cross-validation.")

    print("\n" + format_metrics(metrics, cv))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model.save_model(out / "xgb_cluster_model.json")
    with open(out / "label_mapping.json", "w") as f:
        json.dump({int(i): str(c) for i, c in enumerate(le.classes_)}, f, indent=2)

    metrics_out = dict(metrics)
    metrics_out["cross_validation"] = cv
    metrics_out["n_rows"] = int(len(X))
    metrics_out["n_features"] = int(X.shape[1])
    metrics_out["n_classes"] = n_classes
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    with open(out / "metrics.txt", "w") as f:
        f.write(format_metrics(metrics, cv) + "\n")

    labels_sorted = list(range(n_classes))
    cm = confusion_matrix(y_te, y_pred, labels=labels_sorted)
    cm_df = pd.DataFrame(cm, index=[f"true_{le.classes_[i]}" for i in labels_sorted],
                         columns=[f"pred_{le.classes_[i]}" for i in labels_sorted])
    cm_df.to_csv(out / "confusion_matrix.csv")

    auc_df = pd.DataFrame([
        {"cluster": name,
         "auroc": vals["auroc"],
         "auprc": vals["auprc"],
         "prevalence": vals["prevalence"]}
        for name, vals in metrics["per_class_auc"].items()
    ])
    auc_df.to_csv(out / "per_class_auc.csv", index=False)

    fi = (pd.DataFrame({"feature": X.columns,
                        "importance": model.feature_importances_})
          .sort_values("importance", ascending=False))
    fi.to_csv(out / "feature_importance.csv", index=False)
    print("\nTop features:")
    print(fi.head(15).to_string(index=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        
        fig, ax = plt.subplots(figsize=(1.2 * n_classes + 3,
                                        1.0 * n_classes + 2))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(n_classes)); ax.set_yticks(range(n_classes))
        ax.set_xticklabels(le.classes_, rotation=45, ha="right")
        ax.set_yticklabels(le.classes_)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title("Confusion matrix (test set)")
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(im); fig.tight_layout()
        fig.savefig(out / "confusion_matrix.png", dpi=130)
        plt.close(fig)


        roc_curves = {}
        for i, class_name in enumerate(le.classes_):
            if np.unique(y_te_bin[:, i]).size < 2:
                continue
            fpr, tpr, _ = roc_curve(y_te_bin[:, i], y_prob[:, i])
            roc_curves[str(class_name)] = (fpr, tpr)

   
        if roc_curves:
            mean_fpr = np.linspace(0.0, 1.0, 1001)
            interpolated_tprs = []

            for fpr, tpr in roc_curves.values():
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                interpolated_tprs.append(interp_tpr)

            mean_tpr = np.mean(interpolated_tprs, axis=0)
            mean_tpr[-1] = 1.0

            fig, ax = plt.subplots(figsize=(8, 7))
            ax.plot(
                mean_fpr,
                mean_tpr,
                linewidth=2.5,
                label=f"Macro-average OvR (AUROC={metrics['auroc_macro_ovr']:.4f})"
            )
            ax.plot([0, 1], [0, 1], linestyle=":", linewidth=1.2,
                    label="Chance")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.set_title("Macro-average ROC curve (test set)")
            ax.legend(loc="lower right", fontsize=9)
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(out / "macro_roc_curve.png", dpi=150)
            plt.close(fig)

        
        pr_curves = {}
        for i, class_name in enumerate(le.classes_):
            if y_te_bin[:, i].sum() == 0:
                continue
            precision, recall, _ = precision_recall_curve(
                y_te_bin[:, i], y_prob[:, i]
            )
            pr_curves[str(class_name)] = (precision, recall)

        if pr_curves:
            mean_recall = np.linspace(0.0, 1.0, 1001)
            interpolated_precisions = []

            for precision, recall in pr_curves.values():
               
                recall_inc = recall[::-1]
                precision_inc = precision[::-1]
                interp_precision = np.interp(
                    mean_recall, recall_inc, precision_inc
                )
                interpolated_precisions.append(interp_precision)

            mean_precision = np.mean(interpolated_precisions, axis=0)

            fig, ax = plt.subplots(figsize=(8, 7))
            ax.plot(
                mean_recall,
                mean_precision,
                linewidth=2.5,
                label=f"Macro-average OvR (AUPRC={metrics['auprc_macro_ovr']:.4f})"
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("Macro-average precision-recall curve (test set)")
            ax.legend(loc="lower left", fontsize=9)
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(out / "macro_precision_recall_curve.png", dpi=150)
            plt.close(fig)

       
        fig, ax = plt.subplots(figsize=(8, 7))
        for class_name, (fpr, tpr) in roc_curves.items():
            class_auc = metrics["per_class_auc"][class_name]["auroc"]
            ax.plot(
                fpr,
                tpr,
                linewidth=1.8,
                label=f"Cluster {class_name} (AUROC={class_auc:.4f})"
            )

        ax.plot([0, 1], [0, 1], linestyle=":", linewidth=1.2,
                label="Chance")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("Per-cluster one-vs-rest ROC curves (test set)")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / "per_cluster_roc_curves.png", dpi=150)
        plt.close(fig)


        fig, ax = plt.subplots(figsize=(8, 7))
        for class_name, (precision, recall) in pr_curves.items():
            class_ap = metrics["per_class_auc"][class_name]["auprc"]
            ax.plot(
                recall,
                precision,
                linewidth=1.8,
                label=f"Cluster {class_name} (AUPRC={class_ap:.4f})"
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Per-cluster one-vs-rest precision-recall curves (test set)")
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / "per_cluster_precision_recall_curves.png", dpi=150)
        plt.close(fig)

        print(f"\nSaved model + metrics + plots -> {out}/")
    except Exception as e:
        print(f"\n(Plot generation skipped: {e})")
        print(f"Saved model + metrics -> {out}/")



if __name__ == "__main__":
    main()
