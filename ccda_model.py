import torch
import torch.nn as nn
from transformers import BertModel

class CCDA(nn.Module):
    def __init__(self, bert_model="bert-base-uncased", feature_dim=10):
        super(CCDA, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.feature_fc = nn.Linear(feature_dim, 768)
        self.attention = nn.MultiheadAttention(embed_dim=768, num_heads=8)
        self.fc = nn.Linear(768, 3)

    def forward(self, text_inputs, features):
        text_embeddings = self.bert(text_inputs).last_hidden_state[:, 0, :]
        feature_embeddings = self.feature_fc(features)
        combined_embeddings = torch.cat((text_embeddings.unsqueeze(0), feature_embeddings.unsqueeze(0)), dim=0)
        attn_output, _ = self.attention(combined_embeddings, combined_embeddings, combined_embeddings)
        output = self.fc(attn_output[0])
        return output

model = CCDA()
