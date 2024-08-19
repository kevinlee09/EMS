import torch
import torch.nn as nn


# RNN
##############################################################################
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size=256, n_layer=1, bidirectional=False):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(input_size, hidden_size, n_layer, bidirectional=bidirectional, dropout=0.2 if n_layer==2 else 0)

        self.init_hidden = self.initHidden()

    def forward(self, input, init_hidden):
        """
        :param input: (seq_len, batch_size, feature_dim)
        :return:
            output: (seq_len, batch, num_directions * hidden_size)
            h_n: (num_layers * num_directions, batch, hidden_size)
        """
        output, hidden = self.gru(input, init_hidden)
        return output, hidden

    def initHidden(self, batch_size=1):
        return torch.zeros(self.n_layer * self.num_directions, batch_size, self.hidden_size, requires_grad=False)
    
if __name__ == "__main__":
    encoder =EncoderRNN(input_size=3, hidden_size=256, n_layer=1, bidirectional=False)
    init_hidden = encoder.initHidden()
    input_x = torch.randn(10, 1, 3)    # batch_size: 1
    out, hidden = encoder(input_x, init_hidden)
    print(out.shape)