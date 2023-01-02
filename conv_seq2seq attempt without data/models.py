# coding=utf-8
# Copyright 2021, Duong Nguyen
#
# Licensed under the CECILL-C License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.cecill.info
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Models for TrAISformer.
    https://arxiv.org/abs/2109.03958

The code is built upon:
    https://github.com/karpathy/minGPT
"""

import math
import logging
import pdb


import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)
from config_trAISformer import Config
cfg=Config()
device=cfg.device


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.max_seqlen, config.max_seqlen))
                                     .view(1, 1, config.max_seqlen, config.max_seqlen))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y
class CausalCrossAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.max_seqlen+1, config.max_seqlen+1))
                                     .view(1, 1, config.max_seqlen+1, config.max_seqlen+1))
        self.n_head = config.n_head

    def forward(self, x, state, layer_past=None):
        B, T, C = x.size()
        B2, T2, C2 = state.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(state).view(B2, T2, self.n_head, C2 // self.n_head).transpose(1, 2) # (B2, nh, T2, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B2, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T2) -> (B, nh, T, T2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T2) x (B, nh, T2, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Conv1d(config.n_embd, 2*config.n_embd, kernel_size=3, padding=1),
            nn.GLU(dim=1),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        # x = x + self.attn(self.ln1(x))
        x = x + self.mlp(x.transpose(-2,-1)).transpose(-2,-1)
        x = x + self.attn(x)
        return x

class Encoder(nn.Module):
    """ Encoder """

    def __init__(self, config):
        super().__init__()
        self.conv=nn.Conv1d(config.n_embd, 2*config.n_embd, kernel_size=3)
        self.glu=nn.GLU(dim=1)
        self.dropout=nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        # x = x + self.attn(self.ln1(x))
        conved=self.conv(x.transpose(-2,-1))
        conved1, conved2=torch.unsqueeze(conved[:,::2,:], 1), torch.unsqueeze(conved[:,1::2,:], 1)
        conv_array=[self.glu(torch.cat((conved1[:,i,:], conved2[:,i,:]), dim=1)) for i in range(conved1.shape[1])]
        out=torch.cat(tuple(conv_array), dim=1)
        x = out.transpose(-2,-1)
        return x

class Decoder(nn.Module):
    """ Decoder """

    def __init__(self, config):
        super().__init__()
        self.conv=nn.Conv1d(config.n_embd, 2*config.n_embd, kernel_size=3)
        self.glu=nn.GLU(dim=1)
        self.dropout=nn.Dropout(config.resid_pdrop)
        self.n_embd=config.n_embd

    def forward(self, x, with_targets):
        # x = x + self.attn(self.ln1(x))
        if with_targets:
            conved=self.conv(x.transpose(-2,-1))
            conved1, conved2=torch.unsqueeze(conved[:,::2,:], 1), torch.unsqueeze(conved[:,1::2,:], 1)
            conv_array=[self.glu(torch.cat((conved1[:,i,:], conved2[:,i,:]), dim=1)) for i in range(conved1.shape[1])]
            out=torch.cat(tuple(conv_array), dim=1)
        else:
            x_t=x.transpose(-2,-1)
            y=[]
            for i in range(x_t.shape[2]-2):
                y_temp=self.conv(x_t[:,:,i:i+3])
                y.append(self.glu(y_temp))
                if i<x_t.shape[2]-4:
                    x_t[:,:,i+3]=self.glu(y_temp).squeeze()
            out=torch.cat(tuple([i for i in y]), dim=-1)
        x = out.transpose(-2,-1)
        return x

class TrAISformer(nn.Module):
    """Transformer for AIS trajectories."""

    def __init__(self, config, partition_model = None):
        super().__init__()

        self.lat_size = config.lat_size
        self.lon_size = config.lon_size
        self.sog_size = config.sog_size
        self.cog_size = config.cog_size
        self.full_size = config.full_size
        self.n_lat_embd = config.n_lat_embd
        self.n_lon_embd = config.n_lon_embd
        self.n_sog_embd = config.n_sog_embd
        self.n_cog_embd = config.n_cog_embd
        self.max_seqlen=config.max_seqlen
        self.register_buffer(
            "att_sizes", 
            torch.tensor([config.lat_size, config.lon_size, config.sog_size, config.cog_size]))
        self.register_buffer(
            "emb_sizes", 
            torch.tensor([config.n_lat_embd, config.n_lon_embd, config.n_sog_embd, config.n_cog_embd]))
        
        if hasattr(config,"partition_mode"):
            self.partition_mode = config.partition_mode
        else:
            self.partition_mode = "uniform"
        self.partition_model = partition_model
        
        if hasattr(config,"blur"):
            self.blur = config.blur
            self.blur_learnable = config.blur_learnable
            self.blur_loss_w = config.blur_loss_w
            self.blur_n = config.blur_n
            if self.blur:
                self.blur_module = nn.Conv1d(1, 1, 3, padding = 1, padding_mode = 'replicate', groups=1, bias=False)
                if not self.blur_learnable:
                    for params in self.blur_module.parameters():
                        params.requires_grad = False
                        params.fill_(1/3)
            else:
                self.blur_module = None
                
        
        if hasattr(config,"lat_min"): # the ROI is provided.
            self.lat_min = config.lat_min
            self.lat_max = config.lat_max
            self.lon_min = config.lon_min
            self.lon_max = config.lon_max
            self.lat_range = config.lat_max-config.lat_min
            self.lon_range = config.lon_max-config.lon_min
            self.sog_range = 30.
            
        if hasattr(config,"mode"): # mode: "pos" or "velo".
            # "pos": predict directly the next positions.
            # "velo": predict the velocities, use them to 
            # calculate the next positions.
            self.mode = config.mode
        else:
            self.mode = "pos"
    

        # Passing from the 4-D space to a high-dimentional space
        self.lat_emb = nn.Embedding(self.lat_size, config.n_lat_embd)
        self.lon_emb = nn.Embedding(self.lon_size, config.n_lon_embd)
        self.sog_emb = nn.Embedding(self.sog_size, config.n_sog_embd)
        self.cog_emb = nn.Embedding(self.cog_size, config.n_cog_embd)
        self.t_lat_emb = nn.Embedding(self.lat_size, config.n_lat_embd, device=device)
        self.t_lon_emb = nn.Embedding(self.lon_size, config.n_lon_embd, device=device)
        self.t_sog_emb = nn.Embedding(self.sog_size, config.n_sog_embd, device=device)
        self.t_cog_emb = nn.Embedding(self.cog_size, config.n_cog_embd, device=device)
            
            
        self.pos_emb = nn.Parameter(torch.zeros(1, config.max_seqlen+3, config.n_embd))
        self.t_pos_emb = nn.Parameter(torch.zeros(1, config.max_seqlen+3, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        
        # transformer
        self.block = Block(config)
        self.encoder=Encoder(config)
        self.decoder=Decoder(config)
        self.attn=CausalCrossAttention(config)
        
        
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        if self.mode in ("mlp_pos","mlp"):
            self.head = nn.Linear(config.n_embd, config.n_embd, bias=False)
        else:
            self.head = nn.Linear(config.n_embd, self.full_size, bias=False) # Classification head
            
        self.max_seqlen = config.max_seqlen
        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_max_seqlen(self):
        return self.max_seqlen

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        no_decay.add('t_pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer
   
    
    def to_indexes(self, x, mode="uniform"):
        """Convert tokens to indexes.
        
        Args:
            x: a Tensor of size (batchsize, seqlen, 4). x has been truncated 
                to [0,1).
            model: currenly only supports "uniform".
        
        Returns:
            idxs: a Tensor (dtype: Long) of indexes.
        """
        bs, seqlen, data_dim = x.shape
        if mode == "uniform":
            idxs = (x*self.att_sizes).long()
            return idxs, idxs
        elif mode in ("freq", "freq_uniform"):
            
            idxs = (x*self.att_sizes).long()
            idxs_uniform = idxs.clone()
            discrete_lats, discrete_lons, lat_ids, lon_ids = self.partition_model(x[:,:,:2])
#             pdb.set_trace()
            idxs[:,:,0] = torch.round(lat_ids.reshape((bs,seqlen))).long()
            idxs[:,:,1] = torch.round(lon_ids.reshape((bs,seqlen))).long()                               
            return idxs, idxs_uniform
    
    
    def forward(self, x, masks = None, with_targets=False, return_loss_tuple=False):
        """
        Args:
            x: a Tensor of size (batchsize, seqlen, 4). x has been truncated 
                to [0,1).
            masks: a Tensor of the same size of x. masks[idx] = 0. if 
                x[idx] is a padding.
            with_targets: if True, inputs = x[:,:-1,:], targets = x[:,1:,:], 
                otherwise inputs = x.
        Returns: 
            logits, loss
        """
        
        if self.mode in ("mlp_pos","mlp",):
            idxs, idxs_uniform = x, x # use the real-values of x.
        else:            
            # Convert to indexes
            idxs, idxs_uniform = self.to_indexes(x, mode=self.partition_mode)
        
        if with_targets:
            inputs = idxs[:,:-1,:].contiguous()
            targets = idxs[:,1:,:].contiguous()
            targets2=idxs[:,1:,:].contiguous()
            targets_uniform = idxs_uniform[:,1:,:].contiguous()
            inputs_real = x[:,:-1,:].contiguous()
            targets_real = x[:,1:,:].contiguous()
        else:
            inputs_real = x
            inputs = idxs
            targets = None
            
        batchsize, seqlen, _ = inputs.size()
        assert seqlen <= self.max_seqlen, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        if not with_targets:
            targets2=torch.full((inputs.shape[0], inputs.shape[1], inputs.shape[2]), 3) 
        padding=torch.full((inputs.shape[0], 1, inputs.shape[2]), 2, device=device)
        start=torch.full((inputs.shape[0], 1, inputs.shape[2]), 3, device=device)
        end=torch.full((inputs.shape[0], 1, inputs.shape[2]), 4, device=device)
        if with_targets:
            targets2=torch.cat((padding, padding, start, targets2), dim=1)
        else:
            targets2=torch.cat((padding, padding, start, targets2), dim=1)
        inputs=torch.cat((padding, inputs, end, padding), dim=1)
        lat_embeddings = self.lat_emb(inputs[:,:,0]) # (bs, seqlen, lat_size)
        lon_embeddings = self.lon_emb(inputs[:,:,1]) 
        sog_embeddings = self.sog_emb(inputs[:,:,2]) 
        cog_embeddings = self.cog_emb(inputs[:,:,3])      
        token_embeddings = torch.cat((lat_embeddings, lon_embeddings, sog_embeddings, cog_embeddings),dim=-1)
        targets2=targets2.to(device)
        t_lat_embeddings = self.t_lat_emb(targets2[:,:,0]) # (bs, seqlen, lat_size)
        t_lon_embeddings = self.t_lon_emb(targets2[:,:,1]) 
        t_sog_embeddings = self.t_sog_emb(targets2[:,:,2]) 
        t_cog_embeddings = self.t_cog_emb(targets2[:,:,3])
        t_token_embeddings = torch.cat((t_lat_embeddings, t_lon_embeddings, t_sog_embeddings, t_cog_embeddings),dim=-1)
        # padding=torch.full((token_embeddings.shape[0], 1, token_embeddings.shape[2]), -1, device=device)
        # start=torch.full((token_embeddings.shape[0], 1, token_embeddings.shape[2]), -2, device=device)
        # end=torch.full((token_embeddings.shape[0], 1, token_embeddings.shape[2]), 2, device=device)
        # if with_targets:
        #     end_targets=torch.full((targets.shape[0], 1, targets.shape[2]), 2, device=device)
        # if with_targets:
        #     t_token_embeddings=torch.cat((padding, padding, start,t_token_embeddings), dim=1)
        # else:
        #     t_token_embeddings=torch.cat((padding, padding, start,t_token_embeddings), dim=1)
        # token_embeddings=torch.cat((padding, token_embeddings, end, padding), dim=1)
            
        position_embeddings = self.pos_emb[:, :seqlen+3, :] # each position maps to a (learnable) vector (1, seqlen, n_embd)    
        t_position_embeddings = self.t_pos_emb[:, :seqlen+3, :] # each position maps to a (learnable) vector (1, seqlen, n_embd)
        fea = self.drop(token_embeddings + position_embeddings)
        targ = self.drop(t_token_embeddings + t_position_embeddings)
        enc = self.encoder(fea)
        dec=self.decoder(targ, with_targets)
        a=self.attn(enc, dec)
        # fea = self.ln_f(fea) # (bs, seqlen, n_embd)
        output=dec+a
        output=output[:,:-1,:]
        logits = self.head(output) # (bs, seqlen, full_size) or (bs, seqlen, n_embd)
        
        lat_logits, lon_logits, sog_logits, cog_logits =\
            torch.split(logits, (self.lat_size, self.lon_size, self.sog_size, self.cog_size), dim=-1)
        
        # Calculate the loss
        loss = None
        loss_tuple = None
        if targets is not None:

            sog_loss = F.cross_entropy(sog_logits.view(-1, self.sog_size), 
                                       targets[:,:,2].view(-1), 
                                       reduction="none").view(batchsize,seqlen)
            cog_loss = F.cross_entropy(cog_logits.view(-1, self.cog_size), 
                                       targets[:,:,3].view(-1), 
                                       reduction="none").view(batchsize,seqlen)
            lat_loss = F.cross_entropy(lat_logits.view(-1, self.lat_size), 
                                       targets[:,:,0].view(-1), 
                                       reduction="none").view(batchsize,seqlen)
            lon_loss = F.cross_entropy(lon_logits.view(-1, self.lon_size), 
                                       targets[:,:,1].view(-1), 
                                       reduction="none").view(batchsize,seqlen)                     

            if self.blur:
                lat_probs = F.softmax(lat_logits, dim=-1) 
                lon_probs = F.softmax(lon_logits, dim=-1)
                sog_probs = F.softmax(sog_logits, dim=-1)
                cog_probs = F.softmax(cog_logits, dim=-1)

                for _ in range(self.blur_n):
                    blurred_lat_probs = self.blur_module(lat_probs.reshape(-1,1,self.lat_size)).reshape(lat_probs.shape)
                    blurred_lon_probs = self.blur_module(lon_probs.reshape(-1,1,self.lon_size)).reshape(lon_probs.shape)
                    blurred_sog_probs = self.blur_module(sog_probs.reshape(-1,1,self.sog_size)).reshape(sog_probs.shape)
                    blurred_cog_probs = self.blur_module(cog_probs.reshape(-1,1,self.cog_size)).reshape(cog_probs.shape)

                    blurred_lat_loss = F.nll_loss(blurred_lat_probs.view(-1, self.lat_size),
                                                  targets[:,:,0].view(-1),
                                                  reduction="none").view(batchsize,seqlen)
                    blurred_lon_loss = F.nll_loss(blurred_lon_probs.view(-1, self.lon_size),
                                                  targets[:,:,1].view(-1),
                                                  reduction="none").view(batchsize,seqlen)
                    blurred_sog_loss = F.nll_loss(blurred_sog_probs.view(-1, self.sog_size),
                                                  targets[:,:,2].view(-1),
                                                  reduction="none").view(batchsize,seqlen)
                    blurred_cog_loss = F.nll_loss(blurred_cog_probs.view(-1, self.cog_size),
                                                  targets[:,:,3].view(-1),
                                                  reduction="none").view(batchsize,seqlen)

                    lat_loss += self.blur_loss_w*blurred_lat_loss
                    lon_loss += self.blur_loss_w*blurred_lon_loss
                    sog_loss += self.blur_loss_w*blurred_sog_loss
                    cog_loss += self.blur_loss_w*blurred_cog_loss

                    lat_probs = blurred_lat_probs
                    lon_probs = blurred_lon_probs
                    sog_probs = blurred_sog_probs
                    cog_probs = blurred_cog_probs
                    

            loss_tuple = (lat_loss, lon_loss, sog_loss, cog_loss)
            loss = sum(loss_tuple)
        
            if masks is not None:
                loss = (loss*masks).sum(dim=1)/masks.sum(dim=1)
        
            loss = loss.mean()
        
        if return_loss_tuple:
            return logits, loss, loss_tuple
        else:
            return logits, loss