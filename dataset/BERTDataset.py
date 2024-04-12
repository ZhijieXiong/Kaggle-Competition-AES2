import torch
from torch.utils.data import Dataset


class BERTDataset(Dataset):
    def __init__(self, txt_list, labels, tokenizer, max_length=128):
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
                text=sent,  # text_preprocessing(sent),  # Preprocess sentence
                add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
                max_length=max_len,  # Max length to truncate/pad
                pad_to_max_length=True,  # Pad sentence to max length
                # return_tensors='pt',           # Return PyTorch tensor
                return_attention_mask=True  # Return attention mask
            )

            input_ids.append(encoded_sent.get('input_ids'))
            attention_masks.append(encoded_sent.get('attention_mask'))

        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)

        self.input_ids = input_ids
        self.attn_masks = attention_masks
