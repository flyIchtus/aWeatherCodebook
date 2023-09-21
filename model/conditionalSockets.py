import torch
import torch.nn as nn
import torch.nn.functional as F
from  op import FusedLeakyReLU, fused_leaky_relu
from attention import LinearAttention, CrossAttention, SpatialSelfAttention
from cStylegan2 import EqualLinear
from autoencoder import VQmodule

class LinearSocket(nn.Module):
    def __init__(self,in_ch, in_h, in_w, style_dim):
        super().__init__()
        self.linear_dim = in_ch * in_h * in_w
        self.Socket = EqualLinear(linear_dim, style_dim, activation='fused_lrelu')
    
    def forward(self,x):

        y = torch.flatten(x, dim=(1,2,3))
        y = self.Socket(y)

        return y

class AttentionSocket(nn.Module):
    #TODO : finish the attention socket
    def __init__(self,in_ch, in_h, in_w, heads, use_cross_attention=False, use_linear_attention=False):
        super().__init__()
        self.linear_dim = in_ch * in_h * in_w
        self.Socket = EqualLinear(linear_dim, style_dim, activation='fused_lrelu')

class vqSocket(nn.Module):
    def __init__(self, VQparams, discrete_level=0):
        super().__init_()

        # choose if output should use continuous embeddings alone (0), continuous + discrete (1) or discrete only (2)
        self.discrete_level = discrete_level

        self.VQ = VQmodule(**VQparams)

        # continuous  + discrete doubles the number of channels

    def forward(self,x):
        if self.discrete_level==0 :
                y = self.VQ.encoder(x)
                
        elif self.discrete_level==1:
                h1 = self.VQ.encode(x)
                h2 = self.VQ.quant_conv(h1)
                h2,_,_ = self.VQ.quantize(h2)
                y = torch.cat((h1,h2), dim=1)
            
        
        elif self.discrete_level==2:
                y,_,_ = self.VQ.encode(x)
        
        else :
            raise ValueError(f'discrete_level should be 0 (continuous), 1 (discrete + continuous), 2 (discrete), not {self.discrete_level}')

        return y