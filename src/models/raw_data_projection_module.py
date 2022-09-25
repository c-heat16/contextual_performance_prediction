__author__ = 'Connor Heaton'

import torch.nn as nn


class RawDataProjectionModule(nn.Module):
    def __init__(self, in_dim, out_dim, n_layers, dropout_p=0.15, activation_fn='relu', do_layernorm=False,
                 suppress_final_activation=False, hidden_dim=None, elementwise_affine=False,
                 do_batchnorm=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.activation_fn = activation_fn
        self.do_layernorm = do_layernorm
        self.do_batchnorm = do_batchnorm
        self.suppress_final_activation = suppress_final_activation
        self.hidden_dim = hidden_dim if hidden_dim is not None else in_dim

        self.dropout = nn.Dropout(self.dropout_p)
        if self.activation_fn == 'relu':
            self.activation = nn.ReLU()

        for i in range(self.n_layers):
            if self.do_layernorm and i > 0:
                setattr(self, 'proj_layernorm_{}'.format(i),
                        nn.LayerNorm([self.hidden_dim], elementwise_affine=elementwise_affine))
            elif self.do_batchnorm and i > 0:
                setattr(self, 'proj_bn_{}'.format(i), nn.BatchNorm1d(self.hidden_dim))

            if i == 0:
                setattr(self, 'proj_layer_{}'.format(i), nn.Linear(self.in_dim, self.hidden_dim))
            elif i < self.n_layers - 1:
                setattr(self, 'proj_layer_{}'.format(i), nn.Linear(self.hidden_dim, self.hidden_dim))
            else:
                setattr(self, 'proj_layer_{}'.format(i), nn.Linear(self.hidden_dim, out_dim))

            layer = getattr(self, 'proj_layer_{}'.format(i))
            layer.bias.data.zero_()
            layer.weight.data.normal_(mean=0.0, std=0.01)

    def forward(self, x):
        for i in range(self.n_layers):
            # if i != 0:
            x = self.dropout(x)

            # if self.do_layernorm and 0 < i < (self.n_layers - 1):
            if self.do_layernorm and i > 0:
                norm_layer = getattr(self, 'proj_layernorm_{}'.format(i))
                x = norm_layer(x)
            elif self.do_batchnorm and i > 0:
                norm_layer = getattr(self, 'proj_bn_{}'.format(i))
                x = norm_layer(x)

            # print('i: {} x: {}'.format(i, x.shape))
            proj_layer = getattr(self, 'proj_layer_{}'.format(i))
            # print('proj_layer: {}'.format(proj_layer.weight.shape))
            x = proj_layer(x)

            if self.suppress_final_activation and i == self.n_layers - 1:
                x = x
            else:
                x = self.activation(x)

        return x
