"""BERTimbau heads for the regression and classification probes."""
from __future__ import annotations

from torch import nn
from transformers import BertModel

HIDDEN_SIZE = 768  # BERT-base pooled-output dimension


class BertForRegression(nn.Module):
    """BERTimbau encoder with a single linear head predicting vowel density."""

    def __init__(self, model_name: str, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        pooled = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        return self.regressor(pooled).squeeze(-1)


class BertForClassification(nn.Module):
    """BERTimbau encoder with a linear head over the density bands."""

    def __init__(self, model_name: str, num_classes: int = 3, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        pooled = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        return self.classifier(pooled)
