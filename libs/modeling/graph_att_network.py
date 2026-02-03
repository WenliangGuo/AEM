import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.6):
        """
        Flexible GCN model for feature extraction with the same input/output interface as the GAT.

        Args:
            in_channels (int): Number of input features per node.
            hidden_channels (int): Number of hidden units.
            out_channels (int): Dimensionality of the output features.
            num_layers (int): Total number of GCN layers in the model.
                              If 1, a single GCN layer is used.
            dropout (float): Dropout probability.
        """
        super(GCN, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        if num_layers == 1:
            # Single-layer GCN: directly projects from input to output features.
            self.convs.append(GCNConv(in_channels, out_channels))
        else:
            # First layer: project input features to hidden space.
            self.convs.append(GCNConv(in_channels, hidden_channels))
            # Intermediate layers.
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            # Final layer: project hidden representation to output features.
            self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, data):
        """
        Forward pass through the Flexible GCN model.

        Args:
            data (torch_geometric.data.Data or Batch): Data object containing:
                - x: Node feature matrix of shape [N, in_channels].
                - edge_index: Graph connectivity in COO format with shape [2, E].
                - batch (optional): Vector that maps each node to its graph.

        Returns:
            tuple:
                - torch.Tensor: Node-level features of shape [N, out_channels].
                - torch.Tensor or None: The batch vector if available, else None.
        """
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else None

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if i != self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x, batch
    

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, heads=8, dropout=0.6):
        """
        Flexible GAT model for feature extraction.
        
        Args:
            in_channels (int): Number of input features per node.
            hidden_channels (int): Number of hidden units per head.
            out_channels (int): Dimensionality of the output features.
            num_layers (int): Total number of GAT layers in the model.
                              If 1, a single GAT layer is used.
            heads (int): Number of attention heads for the first (and hidden) layers.
            dropout (float): Dropout probability.
        """
        super(GAT, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        if num_layers == 1:
            # Single-layer GAT: directly projects from input to output features.
            self.convs.append(
                GATConv(in_channels, out_channels, heads=1, concat=False, dropout=dropout)
            )
        else:
            # First layer: project input features to hidden space using multiple heads.
            self.convs.append(
                GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
            )
            # Intermediate hidden layers.
            for _ in range(num_layers - 2):
                self.convs.append(
                    GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
                )
            # Final layer: project hidden representation to output features.
            self.convs.append(
                GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
            )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else None

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if i != self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x, batch