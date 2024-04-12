import torch


def top_k_accuracy(output, target, top_k=(1,)):
    with torch.no_grad():
        max_k = max(top_k)
        batch_size = target.size(0)
        _, y_pred = output.topk(k=max_k, dim=1)
        y_pred = y_pred.t()

        target_reshaped = target.view(1, -1).expand_as(y_pred)
        correct = (y_pred == target_reshaped)

        top_k_acc_all = []
        for k in top_k:
            ind_which_top_k_matched_truth = correct[:k]
            flattened_indicator_which_top_k_matched_truth = ind_which_top_k_matched_truth.reshape(-1).float()
            tot_correct_top_k = flattened_indicator_which_top_k_matched_truth.float().sum(dim=0, keepdim=True)
            top_k_acc = tot_correct_top_k / batch_size
            top_k_acc_all.append(top_k_acc)

        return top_k_acc_all
