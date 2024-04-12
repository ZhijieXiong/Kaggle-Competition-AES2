import torch
from torch.utils.data import Dataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BERTDataset(Dataset):
    def __init__(self, txt_list, labels, tokenizer):
        self.tokenizer = tokenizer
        self.labels = torch.tensor(labels)
        self.input_ids = None
        self.attn_masks = None
        self.sub_token_list_lens = None
        self.preprocess(txt_list, tokenizer)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx], self.labels[idx], self.sub_token_list_lens[idx]

    def preprocess(self, data, tokenizer):
        input_ids = []
        attention_masks = []
        sub_token_list_lens = []
        max_sub_text_num = 1
        for text in data:
            encoded_text = tokenizer.encode_plus(text=text, add_special_tokens=True)
            input_ids_one = encoded_text.get("input_ids")
            attention_mask_one = encoded_text.get("attention_mask")
            start_idx = 0
            sub_token_ids_list = []
            sub_attention_mask_list = []
            token_ids_len = len(input_ids_one)
            current_sub_text_num = 0
            while start_idx < token_ids_len:
                sub_token_ids = input_ids_one[start_idx:start_idx+512]
                sub_token_ids += [0] * (512 - len(sub_token_ids))
                sub_attention_mask = attention_mask_one[start_idx:start_idx + 512]
                sub_attention_mask += [0] * (512 - len(sub_attention_mask))

                sub_token_ids_list.append(sub_token_ids)
                sub_attention_mask_list.append(sub_attention_mask)

                start_idx += 512
                current_sub_text_num += 1
            max_sub_text_num = max(max_sub_text_num, current_sub_text_num)

            input_ids.append(sub_token_ids_list)
            attention_masks.append(sub_attention_mask_list)
            sub_token_list_lens.append(current_sub_text_num)

        for input_id, attention_mask in zip(input_ids, attention_masks):
            for _ in range(max_sub_text_num - len(input_id)):
                input_id.append([1] * 512)
                attention_mask.append([1] * 512)

        self.input_ids = torch.tensor(input_ids).to(DEVICE)
        self.attn_masks = torch.tensor(attention_masks).to(DEVICE)
        self.sub_token_list_lens = torch.tensor(sub_token_list_lens).to(DEVICE)
