import torch
from typing import Tuple
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch import nn
import torch.nn.functional as F
import numpy as np


class RNNClassifier(nn.Module):
    def __init__(self, arch: str, static_input_size: int, dynamic_input_size: int, static_embedding_size: int,
                 hidden_size: int, dropout: int, rnn_layers: int, bidirectional: bool, use_attention: bool,
                 attention_type: str, attention_fields: str, device, fc_layers: int,
                 use_prior_prob_label: bool = False):
        super().__init__()
        assert arch in ['lstm', 'gru']
        assert attention_fields in ['both', 'static_dynamic', 'dynamic_dynamic']
        self.arch = arch
        self.use_attention = use_attention
        self.attention_fields = attention_fields
        self.static_input_size = static_input_size
        self.dynamic_input_size = dynamic_input_size
        self.static_embedding_size = static_embedding_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.rnn_layers = rnn_layers
        self.bidirectional = bidirectional
        self.directions = 2 if self.bidirectional else 1
        self.device = device
        self.fc_layers = fc_layers

        # Layers for operating without attention
        self.static_embedding_layer = nn.Linear(self.static_input_size, self.static_embedding_size)
        self.dropout_static = nn.Dropout(self.dropout)

        # RNN
        self.rnn = nn.LSTM(self.dynamic_input_size, self.hidden_size // self.directions, num_layers=self.rnn_layers,
                           bidirectional=self.bidirectional, dropout=self.dropout, batch_first=True) if arch == 'lstm' \
            else nn.GRU(self.dynamic_input_size, self.hidden_size // self.directions, num_layers=self.rnn_layers,
                        bidirectional=self.bidirectional, dropout=self.dropout, batch_first=True)

        # Attention Mechanisms
        # Static + Dynamic vector attention
        # Static: BSZ, Features => BSZ, Features, 1: features to be analyzed one by one.
        if attention_fields == 'both' or attention_fields == 'static_dynamic':
            self.attn_st_dy = Attention(1, attention_type=attention_type)
        if attention_fields == 'both' or attention_fields == 'dynamic_dynamic':
            # Dynamic vector with dynamic vectors attention
            self.attn_dy_dy = Attention(self.hidden_size, attention_type=attention_type)

        # FCNN
        if self.use_attention:
            if self.attention_fields == 'static_dynamic':
                in_layer = self.static_input_size
                out_layer = self.static_input_size // 2
            elif self.attention_fields == 'both':
                in_layer = self.static_input_size + self.hidden_size
                out_layer = (self.static_input_size + self.hidden_size) // 2
            else:
                in_layer = self.static_embedding_size + self.hidden_size
                out_layer = (self.static_embedding_size + self.hidden_size) // 2
        else:
            in_layer = self.static_embedding_size + self.hidden_size
            out_layer = (self.static_embedding_size + self.hidden_size) // 2

        self.fc = [nn.Linear(in_layer, out_layer)]
        extra_layers = []
        last_out_layer_size = out_layer
        for _ in range(1, self.fc_layers):
            extra_layers.append(nn.Linear(last_out_layer_size, last_out_layer_size // 2))
            last_out_layer_size = last_out_layer_size // 2

        self.fc = nn.ModuleList(self.fc + extra_layers)
        if use_prior_prob_label:
            self.classifier = LinearWithPriorBias(last_out_layer_size, 1)
        else:
            self.classifier = nn.Linear(last_out_layer_size, 1)
        self.classifier_dropout = nn.Dropout(self.dropout)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], seq_idx: int):
        static_data, dynamic_data, lengths, initial_hidden, dynamic_previous = x

        if not self.use_attention or (self.use_attention and self.attention_fields == 'dynamic_dynamic'):
            static_data = F.relu(self.static_embedding_layer(static_data))
            static_output = self.dropout_static(static_data)

        dynamic_data = pack_padded_sequence(dynamic_data, lengths, batch_first=True, enforce_sorted=False)

        _, hidden_orig = self.rnn(dynamic_data, initial_hidden)
        if isinstance(hidden_orig, tuple):
            hidden = hidden_orig[0]
        else:
            hidden = hidden_orig.clone()
        if self.bidirectional:
            dynamic_output = hidden.view(self.rnn_layers, hidden.shape[1], hidden.shape[2] * 2)
        else:
            dynamic_output = hidden

        dynamic_output = dynamic_output[-1, :, :]

        if self.use_attention:
            # static_data + dynamic_output attention
            # static_data: [batch_size, n_features] => [batch_size, n_features, 1]
            static_data = torch.unsqueeze(static_data, 2)
            # dynamic_output: [batch_size, embedding_dim] => [batch_size, embedding_dim, 1]
            dynamic_output = torch.unsqueeze(dynamic_output, 2)
            # add new RNN dynamic_output to attention sequence dynamic_previous
            # dynamic_output: [batch_size, embedding_dim] => [batch_size, embedding_dim]
            dynamic_prev = dynamic_previous.clone()
            dyn_out = torch.zeros(dynamic_prev.shape[0], dynamic_prev.shape[2], 1).to(self.device)
            dyn_out[:dynamic_output.shape[0], :, :] = dynamic_output.clone()

            dynamic_prev[:, seq_idx, :] = torch.squeeze(dyn_out)
            if self.attention_fields == 'both':
                # attention_w_static_dynamic => [batch_size, n_features, embedding_dim]
                attention_static_dynamic, attention_w_static_dynamic = self.attn_st_dy(static_data, dynamic_output)
                attention_static_dynamic = torch.squeeze(
                    attention_static_dynamic.view(attention_static_dynamic.shape[0],
                                                  attention_static_dynamic.shape[2],
                                                  attention_static_dynamic.shape[1]))

                # dynamic_output: [batch_size, embedding_dim] => [batch_size, 1, embedding_dim]
                # attention_dynamic_dynamic [batch_size, 1, embedding_dim]
                # attention_w_dynamic_dynamic [batch size, 1, max_seq_len]
                attention_dynamic_dynamic, attention_w_dynamic_dynamic = self.attn_dy_dy(
                    torch.unsqueeze(torch.squeeze(dyn_out, dim=2), 1), dynamic_prev)
                attention_dynamic_dynamic = torch.squeeze(attention_dynamic_dynamic, dim=1)
                attention_dynamic_dynamic = torch.squeeze(attention_dynamic_dynamic[:dynamic_output.shape[0], :])

                features = torch.cat((attention_static_dynamic, attention_dynamic_dynamic), dim=-1)
            elif self.attention_fields == 'static_dynamic':
                # attention_w_static_dynamic => [batch_size, n_features, embedding_dim]
                attention_static_dynamic, attention_w_static_dynamic = self.attn_st_dy(static_data, dynamic_output)
                attention_static_dynamic = torch.squeeze(
                    attention_static_dynamic.view(attention_static_dynamic.shape[0],
                                                  attention_static_dynamic.shape[2],
                                                  attention_static_dynamic.shape[1]))
                # dynamic + dynamics attention
                dynamic_prev = dynamic_previous.clone()

                # add new RNN dynamic_output to attention sequence dynamic_previous
                # dynamic_output: [batch_size, embedding_dim] => [batch_size, embedding_dim]
                dyn_out = torch.zeros(dynamic_prev.shape[0], dynamic_prev.shape[2], 1)
                dyn_out[:dynamic_output.shape[0], :, :] = dynamic_output.clone()
                features = attention_static_dynamic
                attention_w_dynamic_dynamic = None
            else:
                attention_dynamic_dynamic, attention_w_dynamic_dynamic = self.attn_dy_dy(
                    torch.unsqueeze(torch.squeeze(dyn_out, dim=2), 1), dynamic_prev)
                attention_dynamic_dynamic = torch.squeeze(attention_dynamic_dynamic, dim=1)
                attention_dynamic_dynamic = torch.squeeze(attention_dynamic_dynamic[:dynamic_output.shape[0], :])
                if len(attention_dynamic_dynamic.shape) != 2:
                    attention_dynamic_dynamic = torch.unsqueeze(attention_dynamic_dynamic, 0)
                features = torch.cat((static_output, attention_dynamic_dynamic), dim=-1)
                attention_w_static_dynamic = None
        else:
            features = torch.cat((static_output, dynamic_output), dim=1)
            dynamic_output = torch.unsqueeze(dynamic_output, 2)
            dynamic_prev = dynamic_previous.clone()
            dyn_out = torch.zeros(dynamic_prev.shape[0], dynamic_prev.shape[2], 1).to(self.device)
            dyn_out[:dynamic_output.shape[0], :, :] = dynamic_output.clone()
            dynamic_prev[:, seq_idx, :] = torch.squeeze(dyn_out)
            attention_w_static_dynamic = None
            attention_w_dynamic_dynamic = None

        for fc_layer in self.fc:
            features = F.relu(fc_layer(features))
            features = self.classifier_dropout(features)

        output = self.classifier(features).squeeze()

        return output, hidden_orig, torch.squeeze(dyn_out), attention_w_static_dynamic, \
               attention_w_dynamic_dynamic

    def init_hidden(self, batch_size: int):
        if self.arch == 'lstm':
            return (torch.zeros(
                self.rnn_layers * self.directions, batch_size, self.hidden_size // self.directions).to(self.device),
                    torch.zeros(
                        self.rnn_layers * self.directions, batch_size, self.hidden_size // self.directions).to(
                        self.device))
        return torch.zeros(
            self.rnn_layers * self.directions, batch_size, self.hidden_size // self.directions).to(self.device)


# Module taken from https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/attention.html
class Attention(nn.Module):
    """ Applies attention mechanism on the `context` using the `query`.

    **Thank you** to IBM for their initial implementation of :class:`Attention`. Here is
    their `License
    <https://github.com/IBM/pytorch-seq2seq/blob/master/LICENSE>`__.

    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`

    Example:

         >>> attention = Attention(256)
         >>> query = torch.randn(5, 1, 256)
         >>> context = torch.randn(5, 5, 256)
         >>> output, weights = attention(query, context)
         >>> output.size()
         torch.Size([5, 1, 256])
         >>> weights.size()
         torch.Size([5, 1, 5])
    """

    def __init__(self, dimensions, attention_type='dot'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.reshape(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, dimensions)

        # Include mask on PADDING_INDEX?

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, context)

        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output, attention_weights


# Focal loss implementation, adapted from: https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938
# based on the paper: https://arxiv.org/abs/1708.02002
class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.loss = nn.BCELoss()

    def forward(self, inputs, targets):
        bce_loss = self.loss(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = (1 - pt) ** self.gamma * bce_loss
        return focal_loss


# Implement a linear classification with prior bias for class-imbalanced datasets
# TODO: set a prior probability value that fit the HM dataset
class LinearWithPriorBias(nn.Module):
    def __init__(self, input_features, output_features, prior_prob=0.01):
        super(LinearWithPriorBias, self).__init__()
        self.prior_prob = prior_prob
        self.prior_bias = np.log((1 - self.prior_prob) / self.prior_prob, dtype=np.float32)
        self.linear = nn.Linear(input_features, output_features)
        with torch.no_grad():
            self.linear.bias = nn.Parameter(torch.tensor(self.prior_bias))

    def forward(self, x):
        return self.linear(x)
