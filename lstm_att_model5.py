from torch.utils import data
from random import choice, randrange
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from torch.utils.data.dataloader import DataLoader
from config_trAISformer import Config
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn.utils.rnn as rnn_utils

SOS_token = 0
EOS_token = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

teacher_forcing_ratio = 0.5

class EncoderRNN(nn.Module):
    def __init__(self):
        super(EncoderRNN, self).__init__()
        self.input_size = 4
        self.batch_size = 32
        self.hidden_size = 4
        self.layers = 1
        self.dnn_layers = 0
        self.dropout = 0
        self.bi = False
        if self.dnn_layers > 0:
            for i in range(self.dnn_layers):
                self.add_module('dnn_' + str(i), nn.Linear(
                    in_features=self.input_size if i == 0 else self.hidden_size,
                    out_features=self.hidden_size
                ))
        lstm_input_dim = self.input_size if self.dnn_layers == 0 else self.hidden_size
        self.rnn = nn.LSTM(
            self.batch_size,
            lstm_input_dim,
            self.hidden_size,
            self.layers,
            dropout=self.dropout,
            bidirectional=self.bi,
            batch_first=True)
        self.gpu = False

    def run_dnn(self, x):
        for i in range(self.dnn_layers):
            x = F.relu(getattr(self, 'dnn_'+str(i))(x))
        return x

    def forward(self, inputs, hidden, input_lengths):
        if self.dnn_layers > 0:
            inputs = self.run_dnn(inputs)
        x = rnn_utils.pack_padded_sequence(inputs, input_lengths, batch_first=True, enforce_sorted=False)
        output, state = self.rnn(x, hidden)
        output, _ = rnn_utils.pad_packed_sequence(output, batch_first=True, padding_value=0.)

        if self.bi:
            output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        return output, state

    def init_hidden(self, batch_size):
        h0 = Variable(torch.zeros(2 if self.bi else 1, batch_size, self.hidden_size))
        if self.gpu:
            h0 = h0.cuda()
        return h0


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.batch_size = 32
        self.hidden_size = 4
        embedding_dim = None
        self.embedding_dim = embedding_dim if embedding_dim is not None else self.hidden_size
        self.embedding = nn.Embedding(32, self.embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(
            input_size=self.embedding_dim+self.hidden_size,
            #  if config['decoder'].lower() == 'bahdanau' else self.embedding_dim
            hidden_size=self.hidden_size,
            num_layers=1,
            dropout=0,
            bidirectional=False,
            batch_first=True)
        self.attention = Attention(
            self.batch_size,
            self.hidden_size,
            method="dot",
            mlp=False)

        self.gpu = False
        self.decoder_output_fn = F.log_softmax 
        # if config.get('loss', 'NLL') == 'NLL' else None

    def forward(self, **kwargs):
        """ Must be overrided """
        raise NotImplementedError

class AttentionDecoder(Decoder):
    """
        Corresponds to AttnDecoderRNN
    """

    def __init__(self, output_size):
        super(AttentionDecoder, self).__init__()
        self.output_size = output_size
        self.character_distribution = nn.Linear(2*self.hidden_size, self.output_size)

    def forward(self, input, prev_hidden, encoder_outputs, seq_len):
        """
        :param input: [B]
        :param prev_context: [B, H]
        :param prev_hidden: [B, H]
        :param encoder_outputs: [B, T, H]
        :return: output (B, V), context (B, H), prev_hidden (B, H), weights (B, T)
        Official Tensorflow documentation says : Perform a step of attention-wrapped RNN.
        - Step 1: Mix the `inputs` and previous step's `attention` output via
          `cell_input_fn`.
        - Step 2: Call the wrapped `cell` with this input and its previous state.
        - Step 3: Score the cell's output with `attention_mechanism`.
        - Step 4: Calculate the alignments by passing the score through the
          `normalizer`.
        - Step 5: Calculate the context vector as the inner product between the
          alignments and the attention_mechanism's values (memory).
        - Step 6: Calculate the attention output by concatenating the cell output
          and context through the attention layer (a linear layer with
          `attention_layer_size` outputs).
        Args:
          inputs: (Possibly nested tuple of) Tensor, the input at this time step.
          state: An instance of `AttentionWrapperState` containing
            tensors from the previous time step.
        Returns:
          A tuple `(attention_or_cell_output, next_state)`, where:
          - `attention_or_cell_output` depending on `output_attention`.
          - `next_state` is an instance of `AttentionWrapperState`
             containing the state calculated at this time step.
        Raises:
          TypeError: If `state` is not an instance of `AttentionWrapperState`.
        """

        # RNN (Eq 7 paper)
        embedded = self.embedding(input).unsqueeze(1) # [B, H]
        prev_hidden = prev_hidden.unsqueeze(0)
        rnn_output, hidden = self.rnn(embedded, prev_hidden)
        rnn_output = rnn_output.squeeze(1)

        # Attention weights (Eq 6 paper)
        weights = self.attention.forward(rnn_output, encoder_outputs, seq_len) # B x T
        context = weights.unsqueeze(1).bmm(encoder_outputs).squeeze(1)  # [B x N]

        # Projection (Eq 8 paper)
        # /!\ Don't apply tanh on outputs, it screws everything up
        output = self.character_distribution(torch.cat((rnn_output, context), 1))

        # Apply log softmax if loss is NLL
        if self.decoder_output_fn:
            output = self.decoder_output_fn(output, -1)

        if len(output.size()) == 3:
            output = output.squeeze(1)

        return output, hidden.squeeze(0), weights

class Attention(nn.Module):
    """
    Inputs:
        last_hidden: (batch_size, hidden_size)
        encoder_outputs: (batch_size, max_time, hidden_size)
    Returns:
        attention_weights: (batch_size, max_time)
    """
    def __init__(self, batch_size, hidden_size, method="dot", mlp=False):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        if method == 'dot':
            pass
        elif method == 'general':
            self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
        elif method == "concat":
            self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
            self.va = nn.Parameter(torch.FloatTensor(batch_size, hidden_size))
        elif method == 'bahdanau':
            self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
            self.Ua = nn.Linear(hidden_size, hidden_size, bias=False)
            self.va = nn.Parameter(torch.FloatTensor(batch_size, hidden_size))
        else:
            raise NotImplementedError

        self.mlp = mlp
        if mlp:
            self.phi = nn.Linear(hidden_size, hidden_size, bias=False)
            self.psi = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, last_hidden, encoder_outputs, seq_len=None):
        batch_size, seq_lens, _ = encoder_outputs.size()
        if self.mlp:
            last_hidden = self.phi(last_hidden)
            encoder_outputs = self.psi(encoder_outputs)

        attention_energies = self._score(last_hidden, encoder_outputs, self.method)

        # if seq_len is not None:
        #     attention_energies = mask_3d(attention_energies, seq_len, -float('inf'))

        return F.softmax(attention_energies, -1)

    def _score(self, last_hidden, encoder_outputs, method):
        """
        Computes an attention score
        :param last_hidden: (batch_size, hidden_dim)
        :param encoder_outputs: (batch_size, max_time, hidden_dim)
        :param method: str (`dot`, `general`, `concat`, `bahdanau`)
        :return: a score (batch_size, max_time)
        """

        assert encoder_outputs.size()[-1] == self.hidden_size

        if method == 'dot':
            last_hidden = last_hidden.unsqueeze(-1)
            return encoder_outputs.bmm(last_hidden).squeeze(-1)

        elif method == 'general':
            x = self.Wa(last_hidden)
            x = x.unsqueeze(-1)
            return encoder_outputs.bmm(x).squeeze(-1)

        elif method == "concat":
            x = last_hidden.unsqueeze(1)
            x = F.tanh(self.Wa(torch.cat((x, encoder_outputs), 1)))
            return x.bmm(self.va.unsqueeze(2)).squeeze(-1)

        elif method == "bahdanau":
            x = last_hidden.unsqueeze(1)
            out = F.tanh(self.Wa(x) + self.Ua(encoder_outputs))
            return out.bmm(self.va.unsqueeze(2)).squeeze(-1)

        else:
            raise NotImplementedError

    def extra_repr(self):
        return 'score={}, mlp_preprocessing={}'.format(
            self.method, self.mlp)

# def mask_3d():
def train(train_dataset, test_dataset, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, split, max_length=Config.max_seqlen):
    is_train = split == 'Training'
    data = train_dataset if is_train else test_dataset
    loader = DataLoader(data, shuffle=True, pin_memory=True,
                        batch_size=Config.batch_size,
                        num_workers=Config.num_workers)
    losses = []
    pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    for it, (seqs, masks, seqlens, mmsis, time_starts) in pbar:
        idxs = seqs
        inputs = idxs[:,:-1,:].contiguous()
        targets = idxs[:,1:,:].contiguous()
        batchsize, seqlen, _ = inputs.size()
        encoder_hidden = encoder.init_hidden(batchsize)
        test = inputs[0,:,:]
        input_length = inputs.size(0)
        target_length = targets.size(0)
        assert seqlen <= Config.max_seqlen, "Cannot forward, model block size is exhausted."
        encoder_output, encoder_hidden = encoder(inputs, encoder_hidden, seqlens)
        # for i in range(batchsize):
        #     input_tensor = idxs[i,:-1,:].contiguous()
        #     target_tensor = idxs[i,1:,:].contiguous()
        #     input_length = input_tensor.size(0)
        #     target_length = target_tensor.size(0)

        #     encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        #     loss = 0

        #     for ei in range(2):
        #         encoder_output, encoder_hidden = encoder(input_tensor[ei,ei].type(torch.IntTensor), (torch.zeros(1, 1, 2), torch.zeros(1, 1, 2)))
        #         encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[[SOS_token]]], device=device)

        decoder_hidden = encoder_hidden
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, 120)
            loss += criterion(decoder_output, targets)
            # decoder_input = targets[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, 120)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, targets)
            if decoder_input.item() == EOS_token:
                break

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()
    return loss