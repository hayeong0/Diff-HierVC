import torch
from torch import nn
from torch.nn import functional as F 

from model.base import BaseModule
from model.diffusion_mel import Diffusion as Mel_Diffusion
from model.diffusion_f0 import Diffusion as F0_Diffusion
from model.styleencoder import StyleEncoder  
 
import copy 
import transformers
import typing as tp

from module.modules import *
from module.utils import * 
 

class Wav2vec2(torch.nn.Module):
    def __init__(self, layer=12): 
        super().__init__() 
        self.wav2vec2 = transformers.Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-xls-r-300m")
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
            param.grad = None
        self.wav2vec2.eval()
        self.feature_layer = layer
        
    @torch.no_grad()
    def forward(self, x): 
        outputs = self.wav2vec2(x.squeeze(1), output_hidden_states=True)
        y = outputs.hidden_states[self.feature_layer]    
        
        return y.permute((0, 2, 1))    


class Encoder(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 mel_size=80,
                 gin_channels=0,
                 p_dropout=0):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, p_dropout=p_dropout)
        self.proj = nn.Conv1d(hidden_channels, mel_size, 1)

    def forward(self, x, x_mask, g=None):
        x = self.pre(x * x_mask) * x_mask
        x = self.enc(x, x_mask, g=g)
        x = self.proj(x) * x_mask

        return x


class SynthesizerTrn(nn.Module):
    def __init__(self, encoder_hidden_size=128):
        super().__init__()
        self.emb_c = nn.Conv1d(1024, encoder_hidden_size, 1)
        self.emb_c_f0 = nn.Conv1d(1024, encoder_hidden_size, 1)
        self.emb_f0 = nn.Conv1d(1, encoder_hidden_size, kernel_size=9, stride=4, padding=4) 
        self.emb_norm_f0 = nn.Conv1d(1, encoder_hidden_size, 1)
        self.emb_g = StyleEncoder(in_dim=80, hidden_dim=256, out_dim=256)
        
        self.mel_enc_c = Encoder(encoder_hidden_size, encoder_hidden_size, 5, 1, 8, 80, gin_channels=256, p_dropout=0.1)
        self.mel_enc_f = Encoder(encoder_hidden_size, encoder_hidden_size, 5, 1, 8, 80, gin_channels=256, p_dropout=0.1)
        self.f0_enc = Encoder(encoder_hidden_size, encoder_hidden_size, 5, 1, 8, 128, gin_channels=256, p_dropout=0.1)
        self.proj = nn.Conv1d(encoder_hidden_size, 1, 1)

    def forward(self, w2v, norm_f0, f0, x_mel, x_mask, f0_mask):
        content = self.emb_c(w2v) 
        content_f = self.emb_c_f0(w2v)
        f0 = self.emb_f0(f0)  
        norm_f0 = self.emb_norm_f0(norm_f0)

        g = self.emb_g(x_mel, x_mask).unsqueeze(-1)
        y_cont = self.mel_enc_c(F.relu(content), x_mask, g=g)
        y_f0 = self.mel_enc_f(F.relu(f0), x_mask, g=g)
        y_mel = y_cont + y_f0

        content_f = F.interpolate(content_f, norm_f0.shape[-1])
        enc_f0 = self.f0_enc(F.relu(content_f+norm_f0), f0_mask, g=g)
        y_f0_hat = self.proj(enc_f0)
        
        return g, y_mel, enc_f0, y_f0_hat
    
    def spk_embedding(self, mel, length):
        x_mask = torch.unsqueeze(commons.sequence_mask(length, mel.size(-1)), 1).to(mel.dtype) 
        
        return self.emb_g(mel, x_mask).unsqueeze(-1)

    def mel_predictor(self, w2v, x_mask, spk, pred_f0):
        content = self.emb_c(w2v) 
        pred_f0 = self.emb_f0(pred_f0) 

        y_cont = self.mel_enc_c(F.relu(content), x_mask, g=spk)
        y_f0 = self.mel_enc_f(F.relu(pred_f0), x_mask, g=spk)
        y_mel = y_cont + y_f0
        
        return y_mel
    
    def f0_predictor(self, w2v, x_f0_norm, y_mel, y_mask, f0_mask):
        content_f = self.emb_c_f0(w2v)
        norm_f0 = self.emb_norm_f0(x_f0_norm)
        g = self.emb_g(y_mel, y_mask).unsqueeze(-1)
        content_f = F.interpolate(content_f, norm_f0.shape[-1])
        
        enc_f0 = self.f0_enc(F.relu(content_f+norm_f0), f0_mask, g=g)
        y_f0_hat = self.proj(enc_f0) 
       
        return g, y_f0_hat, enc_f0


class DiffHierVC(BaseModule):
    def __init__(self, n_feats, spk_dim, dec_dim, beta_min, beta_max, hps):
        super(DiffHierVC, self).__init__()
        self.n_feats = n_feats
        self.spk_dim = spk_dim
        self.dec_dim = dec_dim
        self.beta_min = beta_min
        self.beta_max = beta_max

        self.encoder = SynthesizerTrn(hps.model.encoder_hidden_size)
        self.f0_dec = F0_Diffusion(n_feats, 64, spk_dim, beta_min, beta_max)  
        self.mel_dec = Mel_Diffusion(n_feats, dec_dim, spk_dim, beta_min, beta_max) 
    
    def infer_vc(self, x, x_w2v, x_f0_norm, x_f0, x_length, y, y_length, diffpitch_ts, diffvoice_ts):
        x_mask = sequence_mask(x_length, x.size(2)).unsqueeze(1).to(x.dtype)
        y_mask = sequence_mask(y_length, y.size(2)).unsqueeze(1).to(y.dtype)
        f0_mask = sequence_mask(x_length*4, x.size(2)*4).unsqueeze(1).to(x.dtype)

        spk, y_f0_hat, enc_f0 = self.encoder.f0_predictor(x_w2v, x_f0_norm, y, y_mask, f0_mask)

        # Diff-Pitch
        z_f0 = self.f0_dec.compute_diffused_z_pr(x_f0, f0_mask, y_f0_hat, 1.0) 
        z_f0 += torch.randn_like(z_f0, device=z_f0.device)
        pred_f0 = self.f0_dec.reverse(z_f0, f0_mask, y_f0_hat*f0_mask, enc_f0*f0_mask, spk, ts=diffpitch_ts)
        
        # Diff-Voice
        y_mel = self.encoder.mel_predictor(x_w2v, x_mask, spk, pred_f0)
        z_mel = self.mel_dec.compute_diffused_z_pr(x, x_mask, y_mel, 1.0) 
        z_mel += torch.randn_like(z_mel, device=z_mel.device) 
        o_mel = self.mel_dec.reverse(z_mel, x_mask, y_mel, spk, ts=diffvoice_ts)
      
        return o_mel[:, :, :x_length]
 