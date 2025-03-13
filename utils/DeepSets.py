import torch
import torch.nn as nn

layer_map = {
    "Linear": nn.Linear,
    "Conv1d": lambda in_c, out_c: nn.Conv1d(in_channels=in_c, out_channels=out_c, kernel_size=1),
    "ReLU": nn.ReLU,
    "Tanh": nn.Tanh,
    "Softmax": lambda: nn.Softmax(dim=1),  # Requires dim argument,
    "mean": torch.mean,
    "sum": torch.sum,
    "max": torch.max
}


class DeepSetsClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, n_constituents, layers_config, device='cpu', perm_invar_fun='sum'):
        super(DeepSetsClassifier, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_constituents = n_constituents
        self.layers_config = layers_config
        self.perm_invar_fun = layer_map[perm_invar_fun]
        self.needs_permute = False
        # Feature extractor (phi function
        '''
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        ).to(self.device)
        
        # Output function (rho function)
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        ).to(self.device)
        '''
        self.phi = self.construct_network(network_type='phi').to(self.device)
        self.rho = self.construct_network(network_type='rho').to(self.device)


    def construct_network(self, network_type: "phi"):
        layers = []
        if network_type=="phi": layers_list = self.layers_config['phi']
        else: layers_list = self.layers_config['rho']
        for ii, layer in enumerate(layers_list):
            print(layer)
            layer_type = layer['type']
            if layer_type == 'Conv1d': self.needs_permute = True
            if layer_type in layer_map:
                if 'in_dim' in layer:   # Linear layer
                    if ii==0 and network_type=="phi": layer['in_dim'] = self.input_dim
                    if ii==len(layers_list)-1 and network_type=='rho': layer['out_dim'] = self.output_dim
                    layers.append(layer_map[layer_type](layer["in_dim"], layer["out_dim"]))
                else:  # Activation Layer
                    layers.append(layer_map[layer_type]())
        return nn.Sequential(*layers)
                

    def forward(self, X):
        """
        X: Tensor of shape (batch_size, set_size, input_dim)
        """
        if self.n_constituents is not None:
            X = X[:, :self.n_constituents, :]

        if self.needs_permute:
            X = X.permute(0, 2, 1)
            mask = torch.any(X != 0, dim=-2).unsqueeze(-2)  # Shape: (batch_size, set_size, 1)
            perm_fun_dim = 2
        else:
            mask = torch.any(X != 0, dim=-1).unsqueeze(-1)  # Shape: (batch_size, set_size, 1)
            perm_fun_dim = 1
        # Apply phi to all set elements at once
        phi_x = self.phi(X)  # Shape: (batch_size, set_size, hidden_dim)
        # Mask out zero-padded elements
        phi_x = phi_x * mask  # Zero out padded features
        # Aggregate (sum, mean, or max)
        aggregated = self.perm_invar_fun(phi_x, dim=perm_fun_dim)
        # Apply rho to the aggregated features
        output = self.rho(aggregated)  # Shape: (batch_size, output_dim)
        
        return output
