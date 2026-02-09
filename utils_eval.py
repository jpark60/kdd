import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from utils_data import CustomDataset


def get_task_class_ranges(nb_cl_first, nb_cl, nb_groups):
    """Return list of np.array of class IDs per task."""
    ranges = []
    # Task 1
    ranges.append(np.arange(0, nb_cl_first))
    # Later tasks
    for g in range(nb_groups):
        start = nb_cl_first + g * nb_cl
        ranges.append(np.arange(start, start + nb_cl))
    return ranges

def eval_per_task(model, X_test, y_test, nb_cl_first, nb_cl, cur_iter, 
                  batch_size=256, end_class=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    results = []

    for task_id in range(cur_iter + 1):
        if task_id == 0:
            cls_range = np.arange(0, nb_cl_first)
        else:
            start = nb_cl_first + (task_id - 1) * nb_cl
            cls_range = np.arange(start, start + nb_cl)

        mask = np.isin(y_test, cls_range)
        if mask.sum() == 0:
            results.append(np.nan)
            continue

        X_task = X_test[mask]
        y_task = y_test[mask]
        
        test_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_task), 
            torch.LongTensor(y_task)
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )

        correct = 0
        count = 0
        
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                
                logits, _ = model(xb)
                logits = logits[:, :end_class]
                probs = torch.softmax(logits, dim=1)
                pred = torch.argmax(probs, dim=1)

                correct += (pred == yb).sum().item()
                count += xb.size(0)

        results.append(correct / count)

    return results

def evaluate_model_performance(model, X_test, y_test, nb_cl_first, nb_cl, itera, 
                             end_class):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    mask_seen = y_test < end_class
    X_test_seen = X_test[mask_seen]
    y_test_seen = y_test[mask_seen]
    
    acc_task_list = eval_per_task(model, X_test, y_test,
                                  nb_cl_first, nb_cl, itera, 
                                  batch_size=256, end_class=end_class)
    
    # for t_id, acc in enumerate(acc_task_list, 1):
    #     print(f"> Task {t_id} accuracy after Iter {itera+1}: {acc:.4f}")

    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_test_seen), 
        torch.LongTensor(y_test_seen)
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, shuffle=False
    )
    
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits, _ = model(x_batch)
            logits = logits[:, :end_class]
            probs = torch.softmax(logits, dim=1)
            y_pred = torch.argmax(probs, dim=1)
            all_preds.extend(y_pred.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Task {itera+1}, Overall Accuracy: {accuracy:.4f}")
    unique_preds, pred_counts = np.unique(all_preds, return_counts=True)
    unique_labels, label_counts = np.unique(all_labels, return_counts=True)
    
    if len(np.unique(all_labels)) > 1:
        f1_post_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    else:
        f1_post_macro = 0.0
    return {
        'task_accuracies': acc_task_list.copy(),
        'overall_accuracy': accuracy,
        'f1_macro': f1_post_macro,
        'unique_predictions': dict(zip(unique_preds, pred_counts)),
        'unique_labels': dict(zip(unique_labels, label_counts)),
    }


def print_results_summary(iteration_results, nb_groups):
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    def find_final_result(itera_num):
        for result in iteration_results:
            if result['iteration'] == itera_num and result['phase'] == 'final':
                return result
        return None
    
    # Show task-level accuracies for each iteration
    # print("\n[Task Accuracies by Iteration]")
    # for itera_num in range(nb_groups + 1):
    #     result = find_final_result(itera_num)
    #     if result:
    #         task_accs = result['task_accuracies']
    #         print(f"\n  Iteration {itera_num+1}:")
    #         for task_idx, acc in enumerate(task_accs):
    #             if not np.isnan(acc):
    #                 print(f"    Task {task_idx + 1}: {acc:.4f}")
    
    # overall accuracy
    # print("\n" + "-"*60)
    print("\n[Overall Accuracy by Task]")
    for itera_num in range(nb_groups + 1):
        result = find_final_result(itera_num)
        if result:
            print(f"  Task {itera_num+1}: {result['overall_accuracy']:.4f}")
    
    # F1-macro 
    print("\n[F1-Macro by Task]")
    for itera_num in range(nb_groups + 1):
        result = find_final_result(itera_num)
        if result and 'f1_macro' in result:
            print(f"  Task {itera_num+1}: {result['f1_macro']:.4f}")
    
    print("\n" + "="*80)
