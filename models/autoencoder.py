import torch
import torch.nn as nn


class Autoencoder(nn.Module):

    def __init__(self, input_dim, encoder_hidden_dims, decoder_hidden_dims=None, added_features_dim=0, append_mode=1, batch_normalization=False, dropout_rate=None, sigmoid = False) -> None:
        super(Autoencoder, self).__init__()
        self.append_mode = append_mode
        if (append_mode == 1) or (append_mode == 3):
            encoder_dims = [input_dim + added_features_dim, *encoder_hidden_dims]
        else:
            encoder_dims = [input_dim, *encoder_hidden_dims]

        layers = []
        for i in range(len(encoder_dims) - 1):
            layers.append(nn.Linear(encoder_dims[i], encoder_dims[i + 1]))
            layers.append(nn.ReLU())
            if dropout_rate is not None:
                layers.append(nn.Dropout(dropout_rate))
            if batch_normalization:
                layers.append(nn.BatchNorm1d(encoder_dims[i + 1]))
        self.encoder = nn.Sequential(*layers)

        if decoder_hidden_dims is None:
            if len(encoder_hidden_dims) == 1:
                decoder_hidden_dims = []
            else:
                decoder_hidden_dims = encoder_hidden_dims[::-1][1:]

        if (append_mode == 2) or (append_mode == 3):
            decoder_dims = [encoder_dims[-1] + added_features_dim, *decoder_hidden_dims, input_dim]
        else:
            decoder_dims = [encoder_dims[-1], *decoder_hidden_dims, input_dim]
        layers = []
        for i in range(len(decoder_dims) - 1):
            layers.append(nn.Linear(decoder_dims[i], decoder_dims[i + 1]))
            if i <= (len(decoder_dims) - 3):
                layers.append(nn.ReLU())
                if dropout_rate is not None:
                    layers.append(nn.Dropout(dropout_rate))
                if batch_normalization:
                    layers.append(nn.BatchNorm1d(decoder_dims[i + 1]))
            
        if sigmoid:
              layers.append(nn.Sigmoid())  
              
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        if (type(x) == list) or (type(x) == tuple):
            input_shape = x[0].size()
            x_in = x[0].flatten(start_dim=1)
            if self.append_mode == 1: # append at encoder
                out = self.encoder(torch.cat([x_in, x[1]], dim=1))
                out = self.decoder(out)
            elif self.append_mode == 2: # append at decoder
                out = self.encoder(x_in)
                out = self.decoder(torch.cat([out, x[1]], dim=1))
            elif self.append_mode == 3: # append at encoder and decoder
                out = self.encoder(torch.cat([x_in, x[1]], dim=1))
                out = self.decoder(torch.cat([out, x[1]], dim=1))
        else:
            input_shape = x.size()
            x_in = x.flatten(start_dim=1)
            out = self.encoder(x_in)
            out = self.decoder(out)
        return out.view(input_shape)