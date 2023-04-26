

import torch
import torch.nn as nn
from models.lebert import LEBertModel
from models.cnn import MaskCNN
from transformers import BertPreTrainedModel, BertModel




class NestLEBertForNer(BertPreTrainedModel):
    def __init__(self, config):
        super(NestLEBertForNer, self).__init__(config)
        self.word_embeddings = nn.Embedding(config.word_vocab_size, config.word_embed_dim)
        self.bert = LEBertModel(config)
        self.biaffine_size = 200
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.inner_dim = int(self.config.hidden_size / self.config.num_attention_heads)
        self.fc = torch.nn.Linear(self.config.hidden_size, self.biaffine_size * self.inner_dim * 2)
        self.down_fc = torch.nn.Linear(self.biaffine_size, self.config.num_labels)

        if config.use_cnn:
            self.cnn_dim=200
            self.cnn_fc = nn.Linear(self.config.num_labels, self.cnn_dim)
            self.cnn_fc2 = nn.Linear(self.cnn_dim, self.config.num_labels)
            self.cnn = MaskCNN(self.cnn_dim, self.cnn_dim, kernel_size=3, depth=3)

        self.init_weights()

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)
        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)],
                                 dim=-1)
        embeddings = embeddings.repeat(
            (batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to('cuda')
        return embeddings

    def get_table(self, input_ids, attention_mask, token_type_ids, word_ids, word_mask):
        word_embeddings = self.word_embeddings(word_ids)
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
            word_embeddings=word_embeddings, word_mask=word_mask
        )
        encoded_layers = outputs[0]

        batch_size, seq_len = encoded_layers.size(0), encoded_layers.size(1)
        encoded_layers = self.dropout(encoded_layers)

        outputs = self.fc(encoded_layers)
        outputs = torch.split(outputs, self.inner_dim * 2,
                              dim=-1)
        outputs = torch.stack(outputs, dim=-2)
        qw, kw = outputs[..., :self.inner_dim], outputs[...,
                                                self.inner_dim:]

        pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len,
                                                     self.inner_dim)

        cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2,
                                                             dim=-1)
        sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
        qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
        qw2 = qw2.reshape(qw.shape)
        qw = qw * cos_pos + qw2 * sin_pos
        kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
        kw2 = kw2.reshape(kw.shape)
        kw = kw * cos_pos + kw2 * sin_pos

        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.biaffine_size, seq_len, seq_len)


        if hasattr(self, 'cnn'):
            cnn_logits = logits
            biaffine_logits = logits
            cnn_logits = self.cnn(cnn_logits, pad_mask)
            logits = biaffine_logits + cnn_logits
            logits = logits * pad_mask - (1 - pad_mask) * 1e12

        else:
            logits = logits * pad_mask - (1 - pad_mask) * 1e12

            mask = torch.tril(torch.ones_like(logits), -1)
            logits -= mask * 1e12

        logits = self.down_fc(logits.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return logits / self.inner_dim ** 0.5

    def forward(self, input_ids, attention_mask, token_type_ids, word_ids, word_mask, labels=None):

        logits = self.get_table(input_ids, attention_mask, token_type_ids, word_ids, word_mask)
        outputs = (logits,)
        if labels is not None:
            loss = self.compute_loss(logits, labels)
            outputs = (loss,) + outputs
        return outputs

    def compute_loss(self, pred, labels):
        batch_size, num_labels = labels.shape[:2]
        y_true = labels.reshape(batch_size * num_labels, -1)
        y_pred = pred.reshape(batch_size * num_labels, -1)
        loss = self.multilabel_categorical_crossentropy(y_pred, y_true)
        return loss

    def multilabel_categorical_crossentropy(self, y_pred, y_true):
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        return (neg_loss + pos_loss).mean()
























































































































