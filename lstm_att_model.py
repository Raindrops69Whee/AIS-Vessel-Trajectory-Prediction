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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class LSTM(nn.Module):

#     def __init__(self, embedding_dim, hidden_dim, embed_size, tagset_size):
#         super(LSTM, self).__init__()
#         self.hidden_dim = hidden_dim

#         self.embedding = nn.Embedding(embed_size, embedding_dim)

#         # The LSTM takes word embeddings as inputs, and outputs hidden states
#         # with dimensionality hidden_dim.
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim)

#         # The linear layer that maps from hidden state space to tag space
#         self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

#     def forward(self, sentence):
#         embeds = self.embedding(sentence)
#         lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
#         tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
#         tag_scores = F.log_softmax(tag_space, dim=1)
#         return tag_scores
# class lstm_encoder(nn.Module):
#     ''' Encodes time-series sequence '''

#     def __init__(self, input_size, hidden_size, num_layers = 1):
        
#         '''
#         : param input_size:     the number of features in the input X
#         : param hidden_size:    the number of features in the hidden state h
#         : param num_layers:     number of recurrent layers (i.e., 2 means there are
#         :                       2 stacked LSTMs)
#         '''
        
#         super(lstm_encoder, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers

#         # define LSTM layer
#         self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
#                             num_layers = num_layers)

#     def forward(self, x_input):
        
#         '''
#         : param x_input:               input of shape (seq_len, # in batch, input_size)
#         : return lstm_out, hidden:     lstm_out gives all the hidden states in the sequence;
#         :                              hidden gives the hidden state and cell state for the last
#         :                              element in the sequence 
#         '''
        
#         lstm_out, self.hidden = self.lstm(x_input.view(x_input.shape[0], x_input.shape[1], self.input_size))
        
#         return lstm_out, self.hidden     
    
#     def init_hidden(self, batch_size):
        
#         '''
#         initialize hidden state
#         : param batch_size:    x_input.shape[1]
#         : return:              zeroed hidden state and cell state 
#         '''
        
#         return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
#                 torch.zeros(self.num_layers, batch_size, self.hidden_size))


# class lstm_decoder(nn.Module):
#     ''' Decodes hidden state output by encoder '''
    
#     def __init__(self, input_size, hidden_size, num_layers = 1):

#         '''
#         : param input_size:     the number of features in the input X
#         : param hidden_size:    the number of features in the hidden state h
#         : param num_layers:     number of recurrent layers (i.e., 2 means there are
#         :                       2 stacked LSTMs)
#         '''
        
#         super(lstm_decoder, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers

#         self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
#                             num_layers = num_layers)
#         self.linear = nn.Linear(hidden_size, input_size)           

#     def forward(self, x_input, encoder_hidden_states):
        
#         '''        
#         : param x_input:                    should be 2D (batch_size, input_size)
#         : param encoder_hidden_states:      hidden states
#         : return output, hidden:            output gives all the hidden states in the sequence;
#         :                                   hidden gives the hidden state and cell state for the last
#         :                                   element in the sequence 
 
#         '''
        
#         lstm_out, self.hidden = self.lstm(x_input.unsqueeze(0), encoder_hidden_states)
#         output = self.linear(lstm_out.squeeze(0))     
        
#         return output, self.hidden
# class Encoder(nn.Module):
#     def __init__(self, input_size, hidden_size, bidirectional = True):
#         super(Encoder, self).__init__()
#         self.hidden_size = hidden_size
#         self.input_size = input_size
#         self.bidirectional = bidirectional
    
#         self.lstm = nn.LSTM(input_size, hidden_size, bidirectional = bidirectional)
  
#     def forward(self, inputs, hidden):
    
#         output, hidden = self.lstm(inputs.view(1, 1, self.input_size), hidden)
#         return output, hidden
    
#     def init_hidden(self):
#         return (torch.zeros(1 + int(self.bidirectional), 1, self.hidden_size),
#         torch.zeros(1 + int(self.bidirectional), 1, self.hidden_size))

# class AttentionDecoder(nn.Module):
  
#     def __init__(self, hidden_size, output_size, vocab_size):
#         super(AttentionDecoder, self).__init__()
#         self.hidden_size = hidden_size
#         self.output_size = output_size
    
#         self.attn = nn.Linear(hidden_size + output_size, 1)
#         self.lstm = nn.LSTM(hidden_size + vocab_size, output_size) #if we are using embedding hidden_size should be added with embedding of vocab size
#         self.final = nn.Linear(output_size, vocab_size)
  
#     def forward(self, decoder_hidden, encoder_outputs, input):
    
#         weights = []
#         for i in range(len(encoder_outputs)):
#             print(decoder_hidden[0][0].shape)
#             print(encoder_outputs[0].shape)
#             weights.append(self.attn(torch.cat((decoder_hidden[0][0], 
#                                           encoder_outputs[i]), dim = 1)))
#         normalized_weights = F.softmax(torch.cat(weights, 1), 1)
    
#         attn_applied = torch.bmm(normalized_weights.unsqueeze(1),
#                              encoder_outputs.view(1, -1, self.hidden_size))
    
#         input_lstm = torch.cat((attn_applied[0], input[0]), dim = 1) #if we are using embedding, use embedding of input here instead
    
#         output, hidden = self.lstm(input_lstm.unsqueeze(0), decoder_hidden)
    
#         output = self.final(output[0])
    
#         return output, hidden, normalized_weights

#     def init_hidden(self):
#         return (torch.zeros(1, 1, self.output_size),
#         torch.zeros(1, 1, self.output_size))

SOS_token = 0
EOS_token = 1

class EncoderLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=120):
        super(EncoderLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, input, hidden):
        print(input.size())
        output, hidden = self.lstm(input, hidden)
        print(output.size())
        return output, hidden

    def initHidden(self):
        return (torch.zeros(1, 32, self.hidden_size, device=device), torch.zeros(1, 32, self.hidden_size, device=device))

class AttnDecoderLSTM(nn.Module):
    def __init__(self, hidden_size=100, output_size=4, dropout_p=0.1, max_length=Config.max_seqlen):
        super(AttnDecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        print(input.size())
        print(encoder_outputs.size())
        embedded = self.embedding(input.type(torch.LongTensor)).view(1, 1, -1)
        # embedded = self.embedding(input.type(torch.LongTensor)).unsqueeze(1)
        embedded = self.dropout(embedded)
        print(embedded.size())
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded, hidden[0]), 1)), dim=1)
        print(attn_weights.size())
        attn_applied = torch.bmm(attn_weights, encoder_outputs)
        print(embedded.size())
        print(attn_applied.size())
        output = torch.cat((embedded[0], torch.reshape(attn_applied, (1, 1, 2))[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return (torch.zeros(1, 1, self.hidden_size, device=device), torch.zeros(1, 1, self.hidden_size, device=device))

teacher_forcing_ratio = 0.5


def train(train_dataset, test_dataset, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, split, max_length=Config.max_seqlen):
    is_train = split == 'Training'
    data = train_dataset if is_train else test_dataset
    loader = DataLoader(data, shuffle=True, pin_memory=True,
                        batch_size=Config.batch_size,
                        num_workers=Config.num_workers)
    encoder_hidden = encoder.initHidden()
    losses = []
    # n_batches = len(loader) 
    pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    for it, (seqs, masks, seqlens, mmsis, time_starts) in pbar:
        idxs = seqs
        inputs = idxs[:,:-1,:].contiguous()
        targets = idxs[:,1:,:].contiguous()
        batchsize, seqlen, _ = inputs.size()
        test = inputs[0,:,:]
        input_length = inputs.size(0)
        target_length = targets.size(0)
        assert seqlen <= Config.max_seqlen, "Cannot forward, model block size is exhausted."
        encoder_output, encoder_hidden = encoder(inputs, encoder_hidden)
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
                decoder_input, decoder_hidden, encoder_output)
            loss += criterion(decoder_output, targets)
            # decoder_input = targets[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, targets)
            if decoder_input.item() == EOS_token:
                break

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

    return loss

def trainIters(train_dataset, test_dataset, encoder, decoder, max_epochs, print_every=1000, plot_every=100, learning_rate=0.01):
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    assert isinstance(encoder, EncoderLSTM), "wtf"
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for iter in range(1, max_epochs + 1):
        loss = train(train_dataset, test_dataset, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, "Training")
        print_loss_total += loss
        plot_loss_total += loss

    showPlot(plot_losses)

def showPlot(points):
    plt.switch_backend('agg')
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

# def to_indexes(x, mode="uniform"):
#         """Convert tokens to indexes.
        
#         Args:
#             x: a Tensor of size (batchsize, seqlen, 4). x has been truncated 
#                 to [0,1).
#             model: currenly only supports "uniform".
        
#         Returns:
#             idxs: a Tensor (dtype: Long) of indexes.
#         """
#         bs, seqlen, data_dim = x.shape
#         if mode == "uniform":
#             idxs = (x*self.att_sizes).long()
#             return idxs, idxs
#         elif mode in ("freq", "freq_uniform"):
            
#             idxs = (x*self.att_sizes).long()
#             idxs_uniform = idxs.clone()
#             discrete_lats, discrete_lons, lat_ids, lon_ids = self.partition_model(x[:,:,:2])
# #             pdb.set_trace()
#             idxs[:,:,0] = torch.round(lat_ids.reshape((bs,seqlen))).long()
#             idxs[:,:,1] = torch.round(lon_ids.reshape((bs,seqlen))).long()                               
#             return idxs, idxs_uniform
class Trainer_lstm():
    def __init__(self, encoder, decoder, train_dataset, test_dataset, config, learning_rate, max_epochs, savedir=None, device=torch.device("cpu"), max_length=Config.max_seqlen):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.device = device
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
    
    def train(self):
        test_losses = []
        encoder, decoder, config = self.encoder, self.decoder, self.config
        encoder_optimizer = optim.SGD(encoder.parameters(), lr=self.learning_rate)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=self.learning_rate)
        criterion = nn.NLLLoss()
        def run_epoch(split):
            is_train = split == 'Training'
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)
            encoder_hidden = encoder.initHidden()
            losses = []
            # n_batches = len(loader) 
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            for it, (seqs, masks, seqlens, mmsis, time_starts) in pbar:
                idxs = seqs
                inputs = idxs[:,:-1,:].contiguous()
                batchsize, seqlen, _ = inputs.size()
                assert seqlen <= Config.max_seqlen, "Cannot forward, model block size is exhausted."
                for i in range(batchsize):
                    input_tensor = idxs[i,:-1,:].contiguous()
                    target_tensor = idxs[i,1:,:].contiguous()
                    input_length = input_tensor.size(0)
                    target_length = target_tensor.size(0)

                    encoder_outputs = torch.zeros(self.max_length, encoder.hidden_size, device=device)

                    loss = 0

                    for ei in range(input_length):
                        encoder_output, encoder_hidden = encoder(
                            input_tensor[ei], encoder_hidden)
                        encoder_outputs[ei] = encoder_output[0, 0]

                    decoder_input = torch.tensor([[SOS_token]], device=device)

                    decoder_hidden = encoder_hidden

                    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

                    if use_teacher_forcing:
                        # Teacher forcing: Feed the target as the next input
                        for di in range(target_length):
                            decoder_output, decoder_hidden, decoder_attention = decoder(
                                decoder_input, decoder_hidden, encoder_outputs)
                            loss += criterion(decoder_output, target_tensor[di])
                            decoder_input = target_tensor[di]  # Teacher forcing

                    else:
                        # Without teacher forcing: use its own predictions as the next input
                        for di in range(target_length):
                            decoder_output, decoder_hidden, decoder_attention = decoder(
                                decoder_input, decoder_hidden, encoder_outputs)
                            topv, topi = decoder_output.topk(1)
                            decoder_input = topi.squeeze().detach()  # detach from history as input

                            loss += criterion(decoder_output, target_tensor[di])
                            if decoder_input.item() == EOS_token:
                                break

                    loss.backward()

                    encoder_optimizer.step()
                    decoder_optimizer.step()

                    losses.append(loss.item() / target_length)
            return float(np.mean(losses))
        for epoch in range(config.max_epochs):
            _ = run_epoch('Training',epoch=epoch)
            if self.test_dataset is not None:
                test_loss = run_epoch('Valid',epoch=epoch)
                test_losses.append(test_loss)
        return test_losses