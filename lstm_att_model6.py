import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from config_trAISformer import Config

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional = True):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.bidirectional = bidirectional
    
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional = bidirectional)
  
    def forward(self, inputs, hidden):
    
        output, hidden = self.lstm(inputs.view(1, 1, self.input_size), hidden)
        return output, hidden
    
    def init_hidden(self):
        return (torch.zeros(1 + int(self.bidirectional), 1, self.hidden_size),
            torch.zeros(1 + int(self.bidirectional), 1, self.hidden_size))

class AttentionDecoder(nn.Module):
  
    def __init__(self, hidden_size, output_size, vocab_size):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
    
        self.attn = nn.Linear(hidden_size + output_size, 1)
        self.lstm = nn.LSTM(hidden_size + vocab_size, output_size) #if we are using embedding hidden_size should be added with embedding of vocab size
        self.final = nn.Linear(output_size, vocab_size)
  
    def init_hidden(self):
        return (torch.zeros(1, 1, self.output_size),
            torch.zeros(1, 1, self.output_size))
  
    def forward(self, decoder_hidden, encoder_outputs, input):
    
        weights = []
        for i in range(len(encoder_outputs)):
            print(decoder_hidden[0][0].shape)
            print(encoder_outputs[0].shape)
            weights.append(self.attn(torch.cat((decoder_hidden[0][0], 
                                          encoder_outputs[i]), dim = 1)))
        normalized_weights = F.softmax(torch.cat(weights, 1), 1)
    
        attn_applied = torch.bmm(normalized_weights.unsqueeze(1),
                             encoder_outputs.view(1, -1, self.hidden_size))
    
        input_lstm = torch.cat((attn_applied[0], input[0]), dim = 1) #if we are using embedding, use embedding of input here instead
    
        output, hidden = self.lstm(input_lstm.unsqueeze(0), decoder_hidden)
    
        output = self.final(output[0])
    
        return output, hidden, normalized_weights

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
        encoder_hidden = encoder.init_hidden()
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
