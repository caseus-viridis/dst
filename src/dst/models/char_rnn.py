import torch
import torch.nn as nn
from ..modules import DSLinear, DSRNNCell, DSLSTMCell, DSGRUCell


class RNNCellStack(nn.Module):
    r"""
    Multiplayer RNN cells
    """

    def __init__(self, cell_list):
        super(RNNCellStack, self).__init__()
        self.cell_list = cell_list
        self.num_layers = len(self.cell_list)
        self.check_sizes(self.cell_list)
        for i, layer in enumerate(self.cell_list):
            self.add_module(str(i), layer)

    def __len__(self):
        return self.num_layers

    def __getitem__(self, idx):
        assert idx < self.__len__(), "layer index {} out of bound ({})".format(
            idx, self.__len__())
        return self.cell_list[idx]

    @staticmethod
    def check_sizes(cell_list):
        hidden_size = None
        for cell in cell_list:
            if hidden_size is not None and cell.input_size != hidden_size:
                raise RuntimeError(
                    "cell {} has input_size {}, not matching previous hidden_size {}"
                    .format(cell, cell.input_size, hidden_size))
            hidden_size = cell.hidden_size

    def forward(self, input, states):
        output, states_ = input, states
        for i in range(self.num_layers):
            output, states_[i] = self.cell_list[i](output, states[i])
        return output, states_

    def zero_states(self, batch_size=1, device='cpu'):
        return [
            cell.zero_state(batch_size, device) for cell in self.cell_list
        ]


class OneHot(nn.Module):
    def __init__(self, vocab_size):
        super(OneHot, self).__init__()
        self.vocab_size = vocab_size
        self.register_buffer('idmat', torch.eye(vocab_size))

    def forward(self, input):
        return nn.functional.embedding(input, self.idmat)


def create_rnn(cell_type, embed_size, hidden_size, num_layers, **rnn_config):
    if cell_type == 'rnn':
        rnn_cell = DSRNNCell
    elif cell_type == 'lstm':
        rnn_cell = DSLSTMCell
    elif cell_type == 'gru':
        rnn_cell = DSGRUCell
    else:
        raise RuntimeError("unsupported RNN cell type {}".format(cell_type))
    return RNNCellStack([
        rnn_cell(embed_size if _i == 0 else hidden_size, hidden_size,
                 **rnn_config) for _i in range(num_layers)
    ])


class DSCharRNN(nn.Module):
    r"""
    Dynamic sparse character-level RNN model
    """

    def __init__(self,
                 batch_size,
                 seq_len,
                 input_size,
                 output_size,
                 hidden_size=128,
                 depth=2,
                 cell_type='lstm',
                 device='cpu',
                 **rnn_config):
        super(DSCharRNN, self).__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.depth = depth
        self.cell_type = cell_type
        self.device = device
        self.head = OneHot(input_size)
        self.body = create_rnn(cell_type, input_size,
                               hidden_size, depth, **rnn_config)
        self.tail = DSLinear(hidden_size, output_size, bias=False)
        self.to(device)

    def forward(self, input):
        # states = [
        #     s.to(self.device) for s in self.body.zero_states(self.batch_size)
        # ]
        states = self.body.zero_states(self.batch_size, self.device)
        for t in range(input.shape[1]):
            x, states = self.body(self.head(input[:, t]), states)
            yield self.tail(x)


net = DSCharRNN