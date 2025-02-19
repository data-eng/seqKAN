import torch
import torch.nn as nn

from .kan import KAN


class seqKAN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, kan_params: dict, device: torch.device):
        """
        Proposed seqKAN architecture
        :param input_size: Input feature size
        :param hidden_size: Number of hidden states
        :param output_size: Output size
        :param kan_params: Parameters to initialize the KAN layers
        :param device: CPU or GPU
        """
        super(seqKAN, self).__init__()

        self.device = device
        self.hidden_size = hidden_size

        self.KANlayer = KAN(width=[input_size + hidden_size, hidden_size],
                            grid=kan_params['hidden']['grid'], grid_range=kan_params['hidden']['grid_range'],
                            k=kan_params['hidden']['k'],
                            seed=42,
                            device=self.device)

        self.KANoutput = KAN(width=[input_size + hidden_size, output_size],
                             grid=kan_params['output']['grid'], grid_range=kan_params['output']['grid_range'],
                             k=kan_params['output']['k'],
                             seed=42,
                             device=self.device)

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        hidden_state = torch.zeros(batch_size, self.hidden_size).to(self.device)
        x_prev = torch.zeros(x[:, 0, :].shape).to(self.device)

        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]

            combined = torch.cat((x_t, hidden_state), dim=1)
            hidden_state = self.KANlayer(combined)

            combined = torch.cat((x_prev, hidden_state), dim=1)
            output_t = self.KANoutput(combined)

            outputs.append(output_t)

            x_prev = x_t

        outputs = torch.stack(outputs, dim=1)

        return outputs[:, -1, :]


class seqKAN_wide(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, kan_params: dict, device: torch.device):
        """
        seqKAN/wide architecture
        :param input_size: Input feature size
        :param hidden_size: Number of hidden states
        :param output_size: Output size
        :param kan_params: Parameters to initialize the KAN layers
        :param device: CPU or GPU
        """
        super(seqKAN_wide, self).__init__()

        self.device = device
        self.hidden_size = hidden_size

        self.KANlayer = KAN(width=[input_size + hidden_size, hidden_size],
                            grid=kan_params['hidden']['grid'], grid_range=kan_params['hidden']['grid_range'],
                            k=kan_params['hidden']['k'],
                            seed=42,
                            device=self.device)

        self.KANoutput = KAN(width=[hidden_size, output_size],
                             grid=kan_params['output']['grid'], grid_range=kan_params['output']['grid_range'],
                             k=kan_params['output']['k'],
                             seed=42,
                             device=self.device)

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        hidden_state = torch.zeros(batch_size, self.hidden_size).to(self.device)

        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]

            combined = torch.cat((x_t, hidden_state), dim=1)
            hidden_state = self.KANlayer(combined)
            output_t = self.KANoutput(hidden_state)

            outputs.append(output_t)

        outputs = torch.stack(outputs, dim=1)

        return outputs[:, -1, :]
