import torch
import torch.nn as nn
import torch.nn.functional as F

def closest_neighbors(a, b, in_gamut):
    a_f = a.view(a.shape[0],-1)
    b_f = b.view(a.shape[0],-1)
    a_f_expanded = a_f.unsqueeze(2).expand(-1, -1, in_gamut.shape[0])
    b_f_expanded = b_f.unsqueeze(2).expand(-1, -1, in_gamut.shape[0])
    in_gamut_unsqueeze = in_gamut.unsqueeze(0).unsqueeze(0)
    in_gamut_expanded = in_gamut_unsqueeze.expand(a_f.shape[0], a_f.shape[1], -1, -1)
    dist = torch.sqrt((in_gamut_expanded[:,:, :, 0] - a_f_expanded) ** 2 + (in_gamut_expanded[:,:, :, 1] - b_f_expanded) ** 2)
    _, topbins = torch.topk(dist, 5, dim=2, largest = False)
    return topbins.view(a.shape[0], a.shape[1], a.shape[2], 5)

def soft_encoding(a, b, in_gamut):
    soft_encoding = torch.zeros(a.shape[0], a.shape[1], a.shape[2], in_gamut.shape[0])
    p = torch.zeros(a.shape[0], a.shape[1], a.shape[2], 5)
    bins = closest_neighbors(a, b, in_gamut)
    for bin in range(bins.shape[3]):
        a_b = in_gamut[bins[:,:,:,bin],0]
        b_b = in_gamut[bins[:,:,:,bin],1]
        dist = torch.sqrt((a - a_b) ** 2 + (b - b_b) ** 2)
        p[:,:,:,bin] = (torch.exp(-(dist ** 2) / (2 * (5 ** 2))))
    s = p.sum(dim=-1)
    s = s.unsqueeze(-1)
    p = p/s
    for bin in range(bins.shape[3]):
        sub_p = p[:,:,:,bin]
        soft_encoding[:,:,:,bin] = sub_p
    return soft_encoding


class ColorizationLoss(nn.Module):
    def __init__(self, gamut, num_class = 313):
        super(ColorizationLoss, self).__init__()
        self.num_class = num_class
        self.gamut = gamut

    def forward(self, Zbar, Y, rebalance):
        B, _, H, W = Y.shape
        Z = torch.zeros(B, self.num_class, H, W)
        a_b = Y[:, 1:3, :, :]
        a = a_b[:, 0, :, :]
        b = a_b[:, 1, :, :]
        soft_enc = soft_encoding(a, b, self.gamut)
        Z = soft_enc
        Zbar_f = Zbar.view(-1, self.num_class)
        Z_f = Z.view(-1, self.num_class)
        rebalance_f = rebalance.view(-1)
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        rebalance_f = rebalance_f.to(device=device)
        Zbar_f = Zbar_f.to(device=device)
        Z_f = Z_f.to(device=device)
        loss = F.cross_entropy(Zbar_f, Z_f, weight = rebalance_f)
        return loss
    
def rebalance(p_e, l = 0.5, Q = 313):
    w =  ((1-l) * p_e  + l/Q)**-1
    w = 1/w
    w = w/torch.sum(w)
    return w

def prob(dataset, gamut):
    dist = torch.zeros(gamut.shape[0], device = gamut.device)
    a = dataset[:, 1, :, :]
    b = dataset[:, 2, :, :]
    in_gamut_unsqueeze = gamut.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    in_gamut_expanded = gamut.expand(dataset.shape[0], dataset.shape[2], dataset.shape[3], -1, -1)
    a_expanded = a.unsqueeze(-1)
    b_expanded = b.unsqueeze(-1)
    min_dist = torch.sqrt((in_gamut_expanded[:, :, :, :, 0] - a_expanded) ** 2 + (in_gamut_expanded[:, :, :, :, 1] - b_expanded) ** 2)
    min_ind = torch.argmin(min_dist,-1).view(-1)
    dist = torch.bincount(min_ind, minlength=gamut.shape[0])
    p_e = dist/torch.sum(dist)
    return p_e


def prediction(z, T = 0.38):
    z1=z.view(z.shape[0], z.shape[2], z.shape[3], z.shape[1])
    f_t = torch.exp(torch.log(z1)/T)/(torch.sum(torch.exp(torch.log(z1)/T), -1).unsqueeze(-1))
    expected = torch.mean(f_t, dim = -1).unsqueeze(-1)
    expected = expected.view(expected.shape[0], expected.shape[3], expected.shape[1], expected.shape[2])
    return expected