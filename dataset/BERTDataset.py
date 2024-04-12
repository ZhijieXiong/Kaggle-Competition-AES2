import torch
from torch.utils.data import Dataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BERTDataset(Dataset):
    def __init__(self, txt_list, labels, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.labels = torch.tensor(labels)
        self.input_ids = None
        self.attn_masks = None
        self.preprocess(txt_list, tokenizer, max_length)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx], self.labels[idx]

    def preprocess(self, data, tokenizer, max_len):
        input_ids = []
        attention_masks = []

        for sent in data:
            encoded_sent = tokenizer.encode_plus(
                text=sent,
                # Add `[CLS]` and `[SEP]`
                add_special_tokens=True,
                max_length=max_len,
                padding='max_length',
                # Return PyTorch tensor
                # return_tensors='pt',
                # Return attention mask
                return_attention_mask=True
            )

            input_ids.append(encoded_sent.get('input_ids'))
            attention_masks.append(encoded_sent.get('attention_mask'))

        self.input_ids = torch.tensor(input_ids).to(DEVICE)
        self.attn_masks = torch.tensor(attention_masks).to(DEVICE)
