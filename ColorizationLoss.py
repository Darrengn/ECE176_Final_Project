import torch
import torch.nn as nn
import torch.nn.functional as F

def closest_neighbors(a, b, in_gamut):
    #flatten a and b: B,H*W
    a_f = a.view(a.shape[0],-1)
    b_f = b.view(a.shape[0],-1)
    #expand a and b for broadcasting: B,H*W, 313
    a_f_expanded = a_f.unsqueeze(2).expand(-1, -1, in_gamut.shape[0])
    b_f_expanded = b_f.unsqueeze(2).expand(-1, -1, in_gamut.shape[0])
    #expand gamut size to B,H*W,313,2
    in_gamut_unsqueeze = in_gamut.unsqueeze(0).unsqueeze(0)
    in_gamut_expanded = in_gamut_unsqueeze.expand(a_f.shape[0], a_f.shape[1], -1, -1)
    #Euclidean distance between A,B and the in-gamut values
    dist = torch.sqrt((in_gamut_expanded[:,:, :, 0] - a_f_expanded) ** 2 + (in_gamut_expanded[:,:, :, 1] - b_f_expanded) ** 2)
    #Get the index of the 5 smallest distances
    _, topbins = torch.topk(dist, 5, dim=2, largest = False)
    #return the smallest bins shaped as B,H,W,5
    return topbins.view(a.shape[0], a.shape[1], a.shape[2], 5)

def soft_encoding(a, b, in_gamut):
    #soft_encoding is B,H,W,313
    soft_encoding = torch.zeros(a.shape[0], a.shape[1], a.shape[2], 313)
    #p is B,H,W,5 to show 5 closest bins
    p = torch.zeros(a.shape[0], a.shape[1], a.shape[2], 5)
    bins = closest_neighbors(a, b, in_gamut)
    #find distance for each bin
    for bin in range(bins.shape[3]):
        #B,H,W
        a_b = in_gamut[bins[:,:,:,bin],0]
        b_b = in_gamut[bins[:,:,:,bin],1]
        #B,H,W
        dist = torch.sqrt((a - a_b) ** 2 + (b - b_b) ** 2)
        #p is B,H,W,5, current bin is assigned value
        p[:,:,:,bin] = (torch.exp(-(dist ** 2) / (2 * (5 ** 2))))
    #s is B,H,W
    s = p.sum(dim=-1)
    #s is B,H,W,1
    s = s.unsqueeze(-1)
    p = p/s
    #sets closest bins to calculated value
    for bin in range(bins.shape[3]):
        #sub_p is B,H,W,1 which is soft encoding for that bin
        sub_p = p[:,:,:,bin]
        #set soft_encoding bin to normalized value
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
        #a and b are B,H,W
        a = Y[:, 1, :, :]
        b = Y[:, 2, :, :]
        #soft encode Y->Z: B,313,H,W
        Z = soft_encoding(a, b, self.gamut)
        #flatten Zbar for cross entropy: B*H*W*313
        Zbar_f = Zbar.view(-1, self.num_class)
        #flatten Z for cross entropy: B*H*W*313
        Z_f = Z.view(-1, self.num_class)
        #flatten rebalance for cross entropy: 313
        rebalance_f = rebalance.view(-1)
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        #change devices to match available device
        rebalance_f = rebalance_f.to(device=device)
        Zbar_f = Zbar_f.to(device=device)
        Z_f = Z_f.to(device=device)
        #cross entropy computation
        loss = F.cross_entropy(Zbar_f, Z_f, weight = rebalance_f)
        return loss
    
def rebalance(p_e, l = 0.5, Q = 313):
    w =  ((1-l) * p_e  + l/Q)**-1
    w = 1/w
    w = w/torch.sum(w)
    return w

def prob(dataset, gamut):
    dist = torch.zeros(gamut.shape[0], device = gamut.device)
    #get a values B,H,W
    a = dataset[:, 1, :, :]
    #get b values B,H,W
    b = dataset[:, 2, :, :]
    #add dimension to a for broadcasting: B,H,W,1
    a_expanded = a.unsqueeze(-1)
    #add dimension to b for broadcasting: B,H,W,1
    b_expanded = b.unsqueeze(-1)
    #Euclidean distance: B,H,W,313
    min_dist = torch.sqrt((gamut[:, 0] - a_expanded) ** 2 + (gamut[:, 1] - b_expanded) ** 2)
    #find index of min along last dimension: B,H,W,1
    min_ind = torch.argmin(min_dist,-1)
    #flattens min_ind for bincount(): B*H*W,1
    min_ind = min_ind.view(-1)
    #counts how many times each index appears: 313
    dist = torch.bincount(min_ind, minlength=gamut.shape[0])
    #normalize how often each index appears
    p_e = dist/torch.sum(dist)
    return p_e


def prediction(z, T = 0.38):
    z1=z.view(z.shape[0], z.shape[2], z.shape[3], z.shape[1])
    f_t = torch.exp(torch.log(z1)/T)/(torch.sum(torch.exp(torch.log(z1)/T), -1).unsqueeze(-1))
    expected = torch.mean(f_t, dim = -1).unsqueeze(-1)
    expected = expected.view(expected.shape[0], expected.shape[3], expected.shape[1], expected.shape[2])
    return expected