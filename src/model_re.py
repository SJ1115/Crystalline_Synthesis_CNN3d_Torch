import torch
import torch.nn as nn
import numpy as np

class ConvBlk(nn.Module):
    def __init__(self, is_enc:bool, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, pool=2, dropout=0):
        super(ConvBlk, self).__init__()
        Layer = [
            nn.BatchNorm3d(in_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        ]
        
        if is_enc:
            Pool = [nn.MaxPool3d(pool)]
        else: # is_dec
            Pool = [nn.Upsample(scale_factor=pool)]
        Pool.append(nn.Dropout(dropout))
        
        self.Layer = nn.Sequential(*Layer)
        self.Pool  = nn.Sequential(*Pool)
        self.Res = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
    
    def forward(self, x):
        x = self.Res(x) + self.Layer(x)
        x = self.Pool(x)
        return x

class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        
        channels = [config.input_channel] + config.channels
        
        self.Encoder = nn.Sequential(*[
            ConvBlk(is_enc=True, in_channels= channels[i], out_channels=channels[i+1], kernel_size=config.kernel_size, stride=config.stride, pool=config.pool[i], dropout=config.dropout_conv)
                for i in range(len(channels)-1)
        ])
        
        ## C * W * H * D == C * W**3 (W=H=D)
        fc_in  = int(channels[-1] * ( config.input_size / np.product(config.pool) ) ** 3)
        self.FC = nn.Sequential(
            nn.Dropout(config.dropout_FC),
            nn.Linear(fc_in, 1),
            nn.Sigmoid())  ## We will use Binary CE Loss.
        
    def forward(self, x):
        x = self.Encoder(x)
        x = self.FC(x.flatten(1)).squeeze(1)
        return x

class CAE(nn.Module):
    def __init__(self, config):
        super(CAE, self).__init__()
        
        channels = [config.input_channel] + config.channels
        
        Encoder = [
            ConvBlk(is_enc=True,  in_channels= channels[i], out_channels=channels[i+1], kernel_size=config.kernel_size, stride=config.stride, pool=config.pool[i], dropout=config.dropout)
                for i in range(len(channels)-1)
        ]

        Decoder = [
            ConvBlk(is_enc=False, in_channels= channels[i+1], out_channels=channels[i], kernel_size=config.kernel_size, stride=config.stride, pool=config.pool[i], dropout=config.dropout)
                for i in reversed(range(len(channels)-1))
        ]
        last_dec = ConvBlk(is_enc=False, in_channels=config.input_channel, out_channels=config.input_channel, kernel_size=config.kernel_size, stride=config.stride, pool=1, dropout=config.dropout)
        last_dec.Pool = nn.Dropout(config.dropout) ## pop Upsample()
        Decoder.append(last_dec)
        
        self.Encoder = nn.Sequential(*Encoder)
        self.Decoder = nn.Sequential(*Decoder)
    
    def forward(self, x):
        x = self.Encoder(x)
        x = self.Decoder(x)
        return x


class CAE_MLP(nn.Module):
    def __init__(self, CAE_model:CAE, config):
        super(CAE_MLP, self).__init__()
        
        self.Encoder = CAE_model.Encoder
        
        fc_in = int(config.channels[-1] * ( config.input_size / np.product(config.pool) ) ** 3)
        self.FC = nn.Sequential(
            nn.BatchNorm1d(fc_in),
            nn.Dropout(config.dropout_FC),
            nn.Linear(fc_in, 1),
            nn.Sigmoid())  ## We will use Binary CE Loss.

        #nn.init.xavier_uniform_(self.FC[1].weight)
        #nn.init.xavier_uniform_(self.FC[3].weight)
        
    def forward(self, x):
        x = self.Encoder(x)
        x = self.FC(x.flatten(1)).squeeze(1)
        return x
