import torch
import time
import os

import torch.nn as nn
import pandas as pd
import numpy as np

from sklearn.metrics import f1_score, cohen_kappa_score

from .util import top_k_accuracy


def train(model, train_dataloader, val_dataloader, test_dataloader, optimizer, scheduler,
          epochs, device, output_dir, evaluation=True):
    loss_fn = nn.CrossEntropyLoss()
    train_loss_list, train_acc_list, train_f1_list, train_acc5_list = [], [], [], []
    val_loss_list, val_acc_list, val_f1_list, val_acc5_list = [], [], [], []
    test_loss_list, test_acc_list, test_f1_list, test_acc5_list = [], [], [], []

    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        print(f"Start training (epoch {epoch_i+1}) ...")
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Train Acc':^9} | {'Train Acc5':^9} | "
              f"{'Train F1':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Val Acc5':^9} | {'Val F1':^12} | {'Elapsed':^9}")
        print("-"*150)

        t0_epoch, t0_batch = time.time(), time.time()
        total_loss, total_acc, total_f1, total_acc5, batch_loss, batch_acc, batch_f1, batch_acc5, batch_counts = \
            0, 0, 0, 0, 0, 0, 0, 0, 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch_counts += 1
            b_input_ids, b_attn_mask, b_labels, b_sub_lens = tuple(t.to(device) for t in batch)
            b_labels = b_labels.type(torch.LongTensor)
            b_labels = b_labels.to(device)

            model.zero_grad()
            logits = model(b_input_ids, b_attn_mask, b_sub_lens)
            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            predict_score = torch.argmax(logits, dim=1).flatten()

            f1 = f1_score(b_labels.cpu().data, predict_score.cpu().data, average="weighted")
            batch_f1 += f1
            total_f1 += f1

            top_k_acc = top_k_accuracy(logits, b_labels, top_k=(1, 3))
            accuracy = top_k_acc[0].item()
            acc5 = top_k_acc[1].item()
            batch_acc += accuracy
            total_acc += accuracy
            batch_acc5 += acc5
            total_acc5 += acc5

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                time_elapsed = time.time() - t0_batch
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | "
                      f"{batch_acc / batch_counts:^9.6f} | {batch_acc5 / batch_counts:^9.6f} | "
                      f"{batch_f1 / batch_counts:^12.6f} | {'-':^10} | {'-':^10} | {'-':^10} | {'-':^10} | "
                      f"{time_elapsed:^9.2f}")
                batch_loss, batch_acc, batch_acc5, batch_f1, batch_counts = 0, 0, 0, 0, 0
                t0_batch = time.time()

        avg_train_loss = total_loss / len(train_dataloader)
        avg_train_acc = total_acc / len(train_dataloader)
        avg_train_acc5 = total_acc5 / len(train_dataloader)
        avg_train_f1 = total_f1 / len(train_dataloader)
        train_loss_list.append(avg_train_loss)
        train_acc_list.append(avg_train_acc)
        train_acc5_list.append(avg_train_acc5)
        train_f1_list.append(avg_train_f1)

        model_save_path = os.path.join(output_dir, f"epoch-{epoch_i+1}.pt")
        print("-"*150)
        print(f"Saving model after epoch {epoch_i+1} (path: {model_save_path}) ...")
        print("-" * 150)
        torch.save(model, model_save_path)

        # =======================================
        #               Evaluation
        # =======================================
        print("Start evaluating on valid data ...")
        val_loss, val_accuracy, val_acc5, val_f1 = evaluate(model, val_dataloader, device)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_accuracy)
        val_acc5_list.append(val_acc5)
        val_f1_list.append(val_f1)
        time_elapsed = time.time() - t0_epoch
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Train Acc':^9} | {'Train Acc5':^9} | "
              f"{'Train F1':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Val Acc5':^9} | {'Val F1':^12} | {'Elapsed':^9}")
        print("-"*150)
        print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {avg_train_acc:^9.6f} | {avg_train_acc5:^9.6f} | "
              f"{avg_train_f1:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.6f} | {val_acc5:^9.6f} | {val_f1:^10.6f} | "
              f"{time_elapsed:^9.2f}")
        print("-"*150)

        # =======================================
        #               Test
        # =======================================
        if evaluation:
            print("Start evaluating on test data ...")
            test_loss, test_accuracy, test_acc5, test_f1 = evaluate(model, test_dataloader, device)
            test_loss_list.append(test_loss)
            test_acc_list.append(test_accuracy)
            test_acc5_list.append(test_acc5)
            test_f1_list.append(test_f1)
            time_elapsed = time.time() - t0_epoch

            print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Train Acc':^9} | {'Train Acc5':^9} | "
                  f"{'Train F1':^12} | {'Test Loss':^10} | {'Test Acc':^9} | {'Test Acc5':^9} | {'Test F1':^12} | "
                  f"{'Elapsed':^9}")
            print("-"*150)
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {avg_train_acc:^9.6f} | {avg_train_acc5:^9.6f} | "
                  f"{avg_train_f1:^12.6f} | {test_loss:^10.6f} | {test_accuracy:^9.6f} | {test_acc5:^9.6f} | {test_f1:^10.6f} | "
                  f"{time_elapsed:^9.2f}")
            print("-"*150)

        print("\n\n\n")

    epoch_ids = list(range(1, epochs+1))
    if evaluation:
        metric_df = pd.DataFrame(
            np.array([epoch_ids, train_loss_list, train_acc_list, train_acc5_list, train_f1_list, val_loss_list, val_acc_list,
                      val_acc5_list, val_f1_list, test_loss_list, test_acc_list, test_acc5_list, test_f1_list]).T,
            columns=["epoch_id", "train_loss", "train_acc", "train_acc5", "train_f1", "val_loss", "val_acc", "val_acc5", "val_f1",
                     "test_loss", "test_acc", "test_acc5", "test_f1"])
    else:
        metric_df = pd.DataFrame(
            np.array([epoch_ids, train_loss_list, train_acc_list, train_acc5_list, train_f1_list, val_loss_list,
                      val_acc_list, val_acc5_list, val_f1_list]).T,
            columns=["epoch_id", "train_loss", "train_acc", "train_acc5", "train_f1", "val_loss", "val_acc", "val_acc5",
                     "val_f1"])
    metric_df["epoch_id"] = metric_df["epoch_id"].astype(int)
    result_path = os.path.join(output_dir, "metric_df.csv")
    metric_df.to_csv(result_path, encoding="utf-8", index=False)

    return metric_df


def evaluate(model, dataloader, device):
    loss_fn = nn.CrossEntropyLoss()
    model.eval()

    total_acc = 0
    total_acc5 = 0
    total_loss = 0
    total_f1 = 0
    full_b_labels, full_predict_score = [], []

    for batch in dataloader:
        b_input_ids, b_attn_mask, b_labels, b_sub_lens = tuple(t.to(device) for t in batch)
        b_labels = b_labels.type(torch.LongTensor)
        b_labels = b_labels.to(device)

        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask, b_sub_lens)

        loss = loss_fn(logits, b_labels)
        total_loss += loss.item()

        predict_score = torch.argmax(logits, dim=1).flatten()
        f1 = f1_score(b_labels.cpu().data, predict_score.cpu().data, average="weighted")
        total_f1 += f1

        full_b_labels.extend(b_labels.cpu().data)
        full_predict_score.extend(predict_score.cpu().data)

        top_k_acc = top_k_accuracy(logits, b_labels, top_k=(1, 3))
        accuracy = top_k_acc[0].item()
        acc5 = top_k_acc[1].item()
        total_acc += accuracy
        total_acc5 += acc5

    val_loss = total_loss / len(dataloader)
    val_acc = total_acc / len(dataloader)
    val_acc5 = total_acc5 / len(dataloader)
    val_f1 = total_f1 / len(dataloader)
    print("QWK metric is: ", cohen_kappa_score(full_b_labels, full_predict_score, weights="quadratic"))

    return val_loss, val_acc, val_acc5, val_f1
