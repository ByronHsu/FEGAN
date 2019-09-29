import torch

def cal_smooth(flow):
    '''
        Smooth constraint for flow map
    '''
    gx = torch.abs(flow[:, :, :, :-1] - flow[:, :, :, 1:])  # NCHW
    gy = torch.abs(flow[:, :, :-1, :] - flow[:, :, 1:, :])  # NCHW
    smooth = torch.mean(gx) + torch.mean(gy)
    return smooth

def radial_constraint(flow):
    '''
        Flow map has dimension (n, h, w, 2).
    '''
    def normalize_flow(flow):
        norm = torch.sqrt(flow[:, :, :, 0] ** 2 + flow[:, :, :, 1] ** 2)
        flow /= norm.unsqueeze(3).repeat(1, 1, 1, 2)
        return flow
    
    n, h, w, _ = flow.shape
    x = torch.arange(0, h, dtype = torch.float).unsqueeze(1).repeat(1, w) - (h - 1) / 2
    y = torch.arange(0, w, dtype = torch.float).unsqueeze(0).repeat(h, 1) - (w - 1) / 2
    v = torch.cat((x.unsqueeze(2), y.unsqueeze(2)), dim = 2).unsqueeze(0).repeat(n, 1, 1, 1)
    v = normalize_flow(v)
    flow = normalize_flow(flow)
    inner_product = torch.mul(v[:, :, :, 0], flow[:, :, :, 0]) + torch.mul(v[:, :, :, 1], flow[:, :, :, 1])
    radial_loss = torch.sum(inner_product.view(n, -1), dim = 1)
    return torch.mean(radial_loss)
    
    
h = 20
w = 30
n = 5

flow = torch.randn((n, h, w, 2))
loss = radial_constraint(flow)
print(torch.mean(torch.ones(5)))