# ============================================================
# PHASE 3 — Model Training & Comparison
# Logistic Regression vs Random Forest on AAPL features
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, classification_report,
                             roc_curve, auc)
import warnings
warnings.filterwarnings("ignore")

# --- 1. Load feature data ---
df = pd.read_csv("data/aapl_features.csv", header=[0, 1], index_col=0)
df.columns = ["Close", "High", "Low", "Open", "Volume",
              "daily_return", "ma_10", "ma_50", "ma_spread",
              "volatility_10", "rsi_14", "volume_change",
              "price_position", "target"]
df.index = pd.to_datetime(df.index)
df = df.sort_index()

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna()

print(f"Total rows loaded: {len(df)}")

# --- 2. Define features and target ---
features = ["daily_return", "ma_spread", "volatility_10",
            "rsi_14", "volume_change", "price_position"]

X = df[features]
y = df["target"]

# --- 3. Time-based train/test split (80/20) ---
# IMPORTANT: We split by time, not randomly.
# This prevents the model from "seeing the future" during training.
split_idx = int(len(df) * 0.80)
split_date = df.index[split_idx]

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"\nTrain period: {df.index[0].date()} → {df.index[split_idx-1].date()}  ({len(X_train)} rows)")
print(f"Test period:  {split_date.date()} → {df.index[-1].date()}  ({len(X_test)} rows)")

# --- 4. Scale features (important for Logistic Regression) ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# --- 5. Train Logistic Regression ---
print("\nTraining Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_preds  = lr_model.predict(X_test_scaled)
lr_proba  = lr_model.predict_proba(X_test_scaled)[:, 1]

# --- 6. Train Random Forest ---
print("Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=200, max_depth=6,
                                   min_samples_leaf=20, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_preds  = rf_model.predict(X_test_scaled)
rf_proba  = rf_model.predict_proba(X_test_scaled)[:, 1]

# --- 7. Evaluation function ---
def evaluate(name, y_true, y_pred, y_prob):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"{'='*40}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {roc_auc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred,
                                target_names=["Down (0)", "Up (1)"]))
    return acc, prec, rec, f1, roc_auc, fpr, tpr

lr_acc, lr_prec, lr_rec, lr_f1, lr_auc, lr_fpr, lr_tpr = evaluate(
    "Logistic Regression", y_test, lr_preds, lr_proba)

rf_acc, rf_prec, rf_rec, rf_f1, rf_auc, rf_fpr, rf_tpr = evaluate(
    "Random Forest", y_test, rf_preds, rf_proba)

# --- 8. Feature importance (Random Forest) ---
importances = rf_model.feature_importances_
feat_imp = pd.Series(importances, index=features).sort_values(ascending=True)

# --- 9. Visualisations ---
fig = plt.figure(figsize=(16, 14))
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.4)

# --- Plot 1: Accuracy comparison bar chart ---
ax1 = fig.add_subplot(gs[0, 0])
models = ["Logistic\nRegression", "Random\nForest"]
accs   = [lr_acc, rf_acc]
colors = ["#1a7abf", "#27ae60"]
bars = ax1.bar(models, [a * 100 for a in accs], color=colors,
               width=0.4, alpha=0.85)
ax1.set_ylim(40, 70)
ax1.set_ylabel("Accuracy (%)")
ax1.set_title("Model Accuracy Comparison")
ax1.grid(True, alpha=0.3, axis="y")
for bar, acc in zip(bars, accs):
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 40.5,
             f"{acc*100:.1f}%", ha="center", va="bottom", fontsize=11)

# --- Plot 2: Metrics comparison grouped bar ---
ax2 = fig.add_subplot(gs[0, 1])
metrics      = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
lr_scores    = [lr_acc, lr_prec, lr_rec, lr_f1, lr_auc]
rf_scores    = [rf_acc, rf_prec, rf_rec, rf_f1, rf_auc]
x = np.arange(len(metrics))
w = 0.35
ax2.bar(x - w/2, lr_scores, width=w, label="Logistic Regression",
        color="#1a7abf", alpha=0.85)
ax2.bar(x + w/2, rf_scores, width=w, label="Random Forest",
        color="#27ae60", alpha=0.85)
ax2.set_xticks(x)
ax2.set_xticklabels(metrics, fontsize=9)
ax2.set_ylim(0, 1)
ax2.set_title("All Metrics Side by Side")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3, axis="y")

# --- Plot 3: ROC Curves ---
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(lr_fpr, lr_tpr, color="#1a7abf", linewidth=2,
         label=f"Logistic Regression (AUC={lr_auc:.3f})")
ax3.plot(rf_fpr, rf_tpr, color="#27ae60", linewidth=2,
         label=f"Random Forest (AUC={rf_auc:.3f})")
ax3.plot([0,1], [0,1], "k--", linewidth=1, alpha=0.5, label="Random baseline")
ax3.set_xlabel("False Positive Rate")
ax3.set_ylabel("True Positive Rate")
ax3.set_title("ROC Curves")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# --- Plot 4: Confusion Matrix — Logistic Regression ---
ax4 = fig.add_subplot(gs[1, 1])
cm_lr = confusion_matrix(y_test, lr_preds)
im = ax4.imshow(cm_lr, cmap="Blues")
ax4.set_xticks([0,1]); ax4.set_yticks([0,1])
ax4.set_xticklabels(["Pred Down", "Pred Up"])
ax4.set_yticklabels(["Actual Down", "Actual Up"])
ax4.set_title("Confusion Matrix — Logistic Regression")
for i in range(2):
    for j in range(2):
        ax4.text(j, i, str(cm_lr[i, j]), ha="center",
                 va="center", fontsize=14, color="white" if cm_lr[i,j] > cm_lr.max()/2 else "black")

# --- Plot 5: Confusion Matrix — Random Forest ---
ax5 = fig.add_subplot(gs[2, 0])
cm_rf = confusion_matrix(y_test, rf_preds)
ax5.imshow(cm_rf, cmap="Greens")
ax5.set_xticks([0,1]); ax5.set_yticks([0,1])
ax5.set_xticklabels(["Pred Down", "Pred Up"])
ax5.set_yticklabels(["Actual Down", "Actual Up"])
ax5.set_title("Confusion Matrix — Random Forest")
for i in range(2):
    for j in range(2):
        ax5.text(j, i, str(cm_rf[i, j]), ha="center",
                 va="center", fontsize=14, color="white" if cm_rf[i,j] > cm_rf.max()/2 else "black")

# --- Plot 6: Feature Importance ---
ax6 = fig.add_subplot(gs[2, 1])
colors_fi = ["#e94f37" if v == feat_imp.max() else "#1a7abf" for v in feat_imp]
feat_imp.plot(kind="barh", ax=ax6, color=colors_fi, alpha=0.85)
ax6.set_title("Feature Importance (Random Forest)")
ax6.set_xlabel("Importance Score")
ax6.grid(True, alpha=0.3, axis="x")

plt.suptitle("AAPL — Model Training Results", fontsize=15, y=1.01)
plt.savefig("data/aapl_model_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nChart saved to data/aapl_model_results.png")

# --- 10. Save predictions for Phase 4 ---
test_df = df.iloc[split_idx:].copy()
test_df["lr_pred"]  = lr_preds
test_df["rf_pred"]  = rf_preds
test_df["lr_proba"] = lr_proba
test_df["rf_proba"] = rf_proba
test_df.to_csv("data/aapl_predictions.csv")
print("Predictions saved to data/aapl_predictions.csv")
print("\nPhase 3 complete. Ready for Phase 4 — Strategy Construction!")