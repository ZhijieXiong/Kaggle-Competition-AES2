import torch
import torch.nn as nn
from transformers import BertModel


class BERTClassifier(nn.Module):
    def __init__(self, freeze_bert=False, dim_in=768, dim_out=20):
        """
            @param    bert: a BertModel object
            @param    classifier: a torch.nn.Module classifier
            @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(dim_in, dim_out)

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.classifier.apply(init_weights)

    def forward(self, input_ids, attention_mask):
        """
            Feed input to BERT and the classifier to compute logits.
            @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                            max_length)
            @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                            information with shape (batch_size, max_length)
            @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                            num_labels)
        """

        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        last_hidden_state_cls = outputs[0][:, 0, :]

        logits = self.classifier(last_hidden_state_cls)

        return logits
