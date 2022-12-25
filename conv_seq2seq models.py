"""
Models for Convoluted Sequence to Sequence learning.
"""

import math
import logging
import pdb
import sys


import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, encoder_output, decoder_output, positional_embeddings, token_embeddings, target_embeddings):
        lin=nn.Linear(decoder_output.shape[-1], target_embeddings.shape[-1])
        d=lin(decoder_output)+target_embeddings
        z=encoder_output
        p=positional_embeddings
        e=token_embeddings
        m=z.shape
        a=F.softmax(torch.mul(d, z), dim=-1)
        c=torch.mul(a,z+e)
        return c

class Encoder(nn.Module):
    """Just the encoder"""

    def __init__(self, config, x):
        super().__init__()
        self.config=config
        self.x=x
        self.mlp = nn.Sequential(
            nn.Conv1d(x.shape[1], 2*x.shape[1], 1, padding=0, device=device),
            nn.GLU(dim=1),
            nn.Dropout(config.resid_pdrop),
        )
    
    def forward(self, x):
        x = x + self.mlp(x)
        return x

class Decoder(nn.Module):
    """Just the decoder"""

    def __init__(self, config, x):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(x.shape[1], 2*x.shape[1], 1, padding=0, device=device),
            nn.GLU(dim=1),
            nn.Dropout(config.resid_pdrop),
        )
    
    def forward(self, x):
        x = x + self.mlp(x)
        return x
    


class conv_seq2seq(nn.Module):
    """Convoluted sequence to sequence for AIS trajectories."""

    def __init__(self, config, partition_model = None):
        super().__init__()

        self.config = config
        self.lat_size = config.lat_size
        self.lon_size = config.lon_size
        self.sog_size = config.sog_size
        self.cog_size = config.cog_size
        self.full_size = config.full_size
        self.n_lat_embd = config.n_lat_embd
        self.n_lon_embd = config.n_lon_embd
        self.n_sog_embd = config.n_sog_embd
        self.n_cog_embd = config.n_cog_embd
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
            
            
        self.pos_emb = nn.Parameter(torch.zeros(1, config.max_seqlen, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        
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
            dec_targets = idxs[:,1:,:].contiguous()
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
        lat_embeddings = self.lat_emb(inputs[:,:,0]) # (bs, seqlen, lat_size)
        lon_embeddings = self.lon_emb(inputs[:,:,1]) 
        sog_embeddings = self.sog_emb(inputs[:,:,2]) 
        cog_embeddings = self.cog_emb(inputs[:,:,3])
        token_embeddings = torch.cat((lat_embeddings, lon_embeddings, sog_embeddings, cog_embeddings),dim=-1)
        if with_targets:
            dec_lat_embeddings = self.lat_emb(targets[:,:,0]) # (bs, seqlen, lat_size)
            dec_lon_embeddings = self.lon_emb(targets[:,:,1]) 
            dec_sog_embeddings = self.sog_emb(targets[:,:,2]) 
            dec_cog_embeddings = self.cog_emb(targets[:,:,3])       
            dec_token_embeddings = torch.cat((dec_lat_embeddings, dec_lon_embeddings, dec_sog_embeddings, dec_cog_embeddings),dim=-1)
        position_embeddings = self.pos_emb[:, :seqlen, :] # each position maps to a (learnable) vector (1, seqlen, n_embd)
        fea = self.drop(token_embeddings + position_embeddings)
        attn=Attention()
        self.encoder=Encoder(self.config, fea)
        self.z=self.encoder(fea)
        if not with_targets:
            dec_targets=self.z
        else:
            dec_targets = self.drop(dec_token_embeddings + position_embeddings)
        self.decoder=Decoder(self.config, dec_targets)
        self.h=self.decoder(dec_targets)
        self.p=position_embeddings
        self.e=token_embeddings

        fea=attn(self.z, self.h, self.p, self.e, dec_targets)
        fea=fea+self.h
        #fea2=fea.reshape(fea.shape[0], fea.shape[2], fea.shape[1])
        #fea2 = self.ln_f(fea2) # (bs, seqlen, n_embd)
        logits = self.head(fea) # (bs, seqlen, full_size) or (bs, seqlen, n_embd)
        
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
