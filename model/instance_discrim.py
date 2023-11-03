import torch
import torch.nn as nn
import torch.nn.functional as F

def tripletNTXent(a,p,n, temperature):

    ap_sim = F.cosine_similarity(a,p).unsqueeze(1) / temperature
    an_sim = F.cosine_similarity(a,n).unsqueeze(1) / temperature

    pn_sim = F.cosine_similarity(p,n).unsqueeze(1) / temperature

    a_v_p = - 0.5 * torch.log( torch.exp(ap_sim) / (torch.exp(ap_sim) + torch.exp(pn_sim)) + 1e-20)
    a_v_n = - 0.5 * torch.log( torch.exp(an_sim) / (torch.exp(an_sim) + torch.exp(pn_sim)) + 1e-20)

    return (a_v_p + a_v_n)
