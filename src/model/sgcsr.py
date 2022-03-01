from model import common

import torch.nn as nn
import torch

def make_model(args, parent=False):
    return SGCSR(args)

class WindowPartition(nn.Module):
    def __init__(self, window_size=8):
        self.window_size = window_size
        super(WindowPartition, self).__init__()

    def forward(self, x):
        # x: (B, C, H, W)
        window_size = self.window_size
        B, C, H, W = x.shape
        x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
        windows = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(-1, C, window_size, window_size)
        return windows

class WindowReverse(nn.Module):
    def __init__(self, H = 96, W = 96, window_size=8):
        self.window_size = window_size
        self.H = H
        self.W = W
        super(WindowReverse, self).__init__()

    def forward(self, x):
        # x: (num_windows*B, C, window_size, window_size)
        window_size = self.window_size
        H = self.H
        W = self.W
        B = int(x.shape[0] / (H * W / window_size / window_size))
        x = x.view(B, -1, H // window_size, W // window_size, window_size, window_size)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, -1, H, W)
        return x

class SGC(nn.Module):
    def __init__(self, portion = 0.2, k = 3, windowSize = 8):
        self.portion = portion 
        self.k = k
        self.windowSize = windowSize
        super(SGC, self).__init__()

    def _sgcCompute(self, adj, features):
        for _ in range(self.k):
            features = torch.matmul(adj, features)
        return features

    def forward(self, x):
        windowSize = self.windowSize
        B, C, H, W = x.shape
        #features: num_node * embedding size
        features = x.view(B, C, H // windowSize, windowSize, W // windowSize, windowSize)
        features = features.permute(0, 2, 4, 3, 5, 1).contiguous().view(B, -1, windowSize*windowSize*C)
        features = nn.functional.normalize(features,dim=2)
        matrix = torch.matmul(features, features.permute(0, 2, 1))
        kIndex = int(matrix.shape[1]*matrix.shape[1]*(1 - self.portion))
        kValue = torch.kthvalue(matrix.view(B,-1),kIndex,-1,True)[0].unsqueeze(-1)
        adj = (matrix>=kValue).float()
        # D = adj.sum(1).float()
        # D = torch.diag_embed(torch.pow(D , -0.5))
        # adj = torch.matmul(torch.matmul(D,adj),D)
        for _ in range(self.k):
            features = torch.matmul(adj, features)
        #y: reshaped features
        y = features.view(B, H // windowSize, W // windowSize, windowSize, windowSize, C)
        y = y.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)
        return y


# class GetRelationMatrix(nn.Module):
#     def __init__(self,portion):
#         super(GetRelationMatrix, self).__init__()
#         self.portion = portion
#     def forward(self, x):
        # mat = tf.matmul(x,tf.transpose(x,[0,2,1]))
        # batch_size, m = tf.shape(mat)[0], tf.shape(mat)[1]
        # mat1 = tf.reshape(mat,[batch_size,-1])
        # k = tf.cast(tf.cast(m*m,tf.float32)*self.portion,tf.int32)
        # top_k_value = tf.nn.top_k(mat1, k).values[:,-1]
        # top_k_value = tf.expand_dims(top_k_value,axis=-1)
        # mask = tf.cast(mat1 >= top_k_value,tf.float32)
        # mask = tf.reshape(mask,[batch_size,m,m])
        # return x

class SGCSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(SGCSR, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        portion = args.sgc_portion
        kernel_size = 3 
        scale = args.scale[0]
        act = nn.ReLU(True)
        self.url = None
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        pre_blocks = int(n_resblocks//2)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body1 = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(pre_blocks)
        ]

        m_body2 = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks-pre_blocks)
        ]
        m_body2.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]
        # self.windowPartiton = WindowPartition()
        self.head = nn.Sequential(*m_head)
        self.body1 = nn.Sequential(*m_body1)
        self.sgc = SGC(portion = portion)
        self.bn1 = nn.BatchNorm2d(n_feats)
        self.bn2 = nn.BatchNorm2d(n_feats)
        self.body2 = nn.Sequential(*m_body2)
        # self.windowReverse = WindowReverse()
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        # x = self.sub_mean(x)

        x = self.head(x)
        x = self.body1(x)
        x = self.bn1(x)
        x = self.sgc(x)
        x = self.bn2(x)
        x = self.body2(x)

        x = self.tail(x)

        # x = self.add_mean(x)

        # x.mul_(255.0)

        return x 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

