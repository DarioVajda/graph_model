import torch
import numpy as np

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_exact_match(eval_preds):
    preds, labels = eval_preds
    y_true, y_pred = [], []
    
    for i in range(len(labels)):
        valid = labels[i] != -100
        if not np.any(valid): continue
        
        y_true.append("-".join(map(str, labels[i][valid].tolist())))
        y_pred.append("-".join(map(str, preds[i][valid].tolist())))

    # Compute accuracy
    accuracy = np.mean([t == p for t, p in zip(y_true, y_pred)])

    # Compute F1 score
    unique_classes = list(set(y_true))
    majority_class = max(set(y_true), key=y_true.count)
    tp = { cl: 0 for cl in unique_classes }
    fp = { cl: 0 for cl in unique_classes }
    fn = { cl: 0 for cl in unique_classes }
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            tp[true] += 1
        else:
            if pred in unique_classes:
                fp[pred] += 1
            else:
                # Handle model hallucinations by predicting majority class
                fp[majority_class] += 1
            fn[true] += 1
    f1_scores = {
        cl: (2 * tp[cl]) / (2 * tp[cl] + fp[cl] + fn[cl]) if (2 * tp[cl] + fp[cl] + fn[cl]) > 0 else 0.0
        for cl in unique_classes
    }
    macro_f1 = np.mean(list(f1_scores.values()))
    
    # Debugging output
    # print(f"\n--- Metric Debugging ---")
    # print(f"Total Samples: {len(y_true)}")
    # print(f"Unique Classes: {len(unique_classes)}")
    # for cl in unique_classes:
    #     print(f"  Class '{cl}': {tp[cl]} TP, {fp[cl]} FP, {fn[cl]} FN, F1: {f1_scores[cl]:.4f}")
    # print(f"------------------------\n")

    return {
        "em_accuracy": float(accuracy),
        "em_f1": float(macro_f1),
    }

