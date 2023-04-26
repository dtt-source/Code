import torch.nn as nn
import math
from torch.nn import init
import torch

class GraphConvolution(nn.Module):
    """
    Simple GCN layer
    """
    def __init__(self, args, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input_tensor, adj):

        
        input_tensor = input_tensor.unsqueeze(2)

        
        hidden = torch.matmul(input_tensor.float(), self.weight.float())
        denom = torch.sum(adj, dim=2, keepdim=True) + 1

        
        adj = adj.unsqueeze(-1)

        
        output = torch.matmul(adj.float(), hidden)

        
        
        relations_keys = self.relative_positions_embeddings[
            : input_tensor.shape[1], : input_tensor.shape[1], :
        ].detach().clone().to(input_tensor.device)
        
        relations_keys = relations_keys.unsqueeze(0)

        
        output = output + relations_keys
        output = torch.mean(output, dim=2, keepdim=False)    
        output = output / denom
        if self.bias is not None:
            output = output + self.bias

        return F.relu(output.type_as(input_tensor))