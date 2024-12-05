import torch
import torch_geometric.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch_geometric.nn import GCNConv, SGConv, TAGConv, GATConv, SAGEConv
import torch.nn.functional as F
from models.conv import CalibAttentionLayer, CustomSAGEConv


def initialize_model_and_optimizer(args, nfeat, nclass, data, device):
    if args.backbone == 'GCN':

        base_model = GCN(nfeat, args.hidden, nclass, data.num_edges, args.dropout).to(device)
    elif args.backbone == 'TAGCN':
        base_model = TAGCN(nfeat, args.hidden, nclass, data.num_edges, args.dropout).to(device)
    elif args.backbone == 'SGC':
        base_model = SGC(nfeat, args.hidden, nclass, data.num_edges, args.dropout).to(device)
    elif args.backbone == 'GraphSAGE':
        base_model = GraphSAGE(nfeat, args.hidden, nclass, data.num_edges, args.dropout).to(device)

    elif args.backbone == 'GAT':
        base_model = GAT(nfeat, args.hidden, nclass, data.num_edges, args.dropout).to(device)

    else:
        raise ValueError(f"Unsupported backbone type: {args.backbone}")
    optimizer_base = torch.optim.Adam([
        dict(params=base_model.conv1.parameters(), weight_decay=args.base_weight_decay),
        dict(params=base_model.conv2.parameters(), weight_decay=0)
    ], lr=args.base_lr)

    return base_model, optimizer_base


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_edge, dropout):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

        self.edge_weight = torch.ones(num_edge).cuda()

        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            edge_weight = torch.ones(len(edge_index[0])).cuda()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x=x, edge_index=edge_index, edge_weight=edge_weight).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x=x, edge_index=edge_index, edge_weight=edge_weight)
        return x


class TAGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_edge, dropout):
        super().__init__()
        self.dropout = dropout
        self.conv1 = TAGConv(in_channels, hidden_channels)
        self.conv2 = TAGConv(hidden_channels, out_channels)

        self.edge_weight = torch.ones(num_edge).cuda()

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            edge_weight = torch.ones(len(edge_index[0])).cuda()
        x = F.relu(self.conv1(x, edge_index=edge_index, edge_weight=edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index=edge_index, edge_weight=edge_weight)
        return F.log_softmax(x, dim=1)


class SGC(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_edge, dropout):
        super().__init__()
        self.dropout = dropout
        self.conv1 = SGConv(in_channels, hidden_channels, K=2)
        self.conv2 = SGConv(hidden_channels, out_channels, K=2)

        self.edge_weight = torch.ones(num_edge).cuda()

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            edge_weight = torch.ones(len(edge_index[0])).cuda()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index=edge_index, edge_weight=edge_weight).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index=edge_index, edge_weight=edge_weight)
        return x


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_edge, dropout):
        super().__init__()
        self.dropout = dropout
        self.conv1 = CustomSAGEConv(in_channels, hidden_channels)
        self.conv2 = CustomSAGEConv(hidden_channels, out_channels)
        self.edge_weight = torch.ones(num_edge).cuda()

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            edge_weight = torch.ones(len(edge_index[0])).cuda()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x=x, edge_index=edge_index, edge_weight=edge_weight).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x=x, edge_index=edge_index, edge_weight=edge_weight)
        return x


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_edge, dropout):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATConv(in_channels, hidden_channels, heads=8, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * 8, out_channels, heads=1, concat=True, dropout=dropout)
        self.edge_weight = torch.ones(num_edge).cuda()

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            edge_weight = torch.ones(len(edge_index[0])).cuda()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_weight).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x=x, edge_index=edge_index, edge_attr=edge_weight)

        return x


class Edge_Weight_MLP(torch.nn.Module):

    def __init__(self, base_model, out_channels, dropout):
        super(Edge_Weight_MLP, self).__init__()
        self.base_model = base_model

        self.extractor = nn.MLP([out_channels * 2, out_channels * 4, 1], dropout=dropout)

        for para in self.extractor.parameters():
            para.requires_grad = True

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            edge_weight = self.get_weight(x, edge_index)
        logist = self.base_model(x, edge_index, edge_weight)
        return logist

    def get_weight(self, x, edge_index):
        emb = self.base_model(x, edge_index)
        col, row = edge_index
        f1, f2 = emb[col], emb[row]
        f12 = torch.cat([f1, f2], dim=-1)
        edge_weight = self.extractor(f12)
        return edge_weight.relu()


class Edge_Weight_Transformer(torch.nn.Module):
    def __init__(self, base_model, out_channels, dropout):
        super(Edge_Weight_Transformer, self).__init__()
        self.base_model = base_model
        self.dropout = dropout
        d_model = out_channels * 2
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=8, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=6)
        for para in self.base_model.parameters():
            para.requires_grad = False

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            edge_weight = self.get_weight(x, edge_index)
        logist = self.base_model(x, edge_index, edge_weight)
        return logist

    def get_weight(self, x, edge_index):
        emb = self.base_model(x, edge_index)
        col, row = edge_index
        f1, f2 = emb[col], emb[row]
        f12 = torch.cat([f1, f2], dim=-1)
        edge_weight = self.transformer_encoder(f12.unsqueeze(1))

        edge_weight = edge_weight.squeeze(1).relu()

        edge_weight.requires_grad = True

        return edge_weight


class VS(torch.nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.temperature = torch.nn.Parameter(torch.ones(num_classes))
        self.bias = torch.nn.Parameter(torch.ones(num_classes))

        for para in self.base_model.parameters():
            para.requires_grad = False

    def forward(self, x, edge_index, edge_weight=None):
        logits = self.base_model(x, edge_index, edge_weight)
        temperature = self.temperature.unsqueeze(0).expand(logits.size(0), logits.size(1))
        return logits * temperature + self.bias


class Temperature_Scalling(torch.nn.Module):
    def __init__(self, base_model):
        super(Temperature_Scalling, self).__init__()
        self.base_model = base_model
        self.temperature = torch.nn.Parameter(torch.ones(1))

        for para in self.base_model.parameters():
            para.requires_grad = False

    def forward(self, x, edge_index, edge_weight=None):
        logist = self.base_model(x, edge_index, edge_weight)
        temperature = self.temperature.expand(logist.size(0), logist.size(1))
        return logist * temperature

    def reset_parameters(self):
        self.temperature.data.fill_(1)


class CaGCN(torch.nn.Module):
    def __init__(self, base_model, out_channels, hidden_channels):
        super(CaGCN, self).__init__()
        self.base_model = base_model
        self.conv1 = GCNConv(out_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 1)

        for para in self.base_model.parameters():
            para.requires_grad = False

    def forward(self, x, edge_index, edge_weight=None):
        logist = self.base_model(x, edge_index, edge_weight)
        x = F.dropout(logist, p=0.5, training=self.training)
        x = self.conv1(x=x, edge_index=edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        temperature = self.conv2(x=x, edge_index=edge_index)
        temperature = torch.log(torch.exp(temperature) + torch.tensor(1.1))
        return logist * temperature


class GATS(torch.nn.Module):
    def __init__(self, base_model, edge_index, num_nodes, train_mask, num_class, dist_to_train=None):
        super().__init__()
        self.base_model = base_model
        self.num_nodes = num_nodes
        self.cagat = CalibAttentionLayer(in_channels=num_class,
                                         out_channels=1,
                                         edge_index=edge_index,
                                         num_nodes=num_nodes,
                                         train_mask=train_mask,
                                         dist_to_train=dist_to_train,
                                         heads=8,
                                         bias=1).cuda()
        for para in self.base_model.parameters():
            para.requires_grad = False

    def forward(self, x, edge_index, edge_weight=None):
        logits = self.base_model(x, edge_index, edge_weight)
        temperature = self.graph_temperature_scale(logits)
        return logits / temperature

    def graph_temperature_scale(self, logits):
        temperature = self.cagat(logits).view(self.num_nodes, -1)
        return temperature.expand(self.num_nodes, logits.size(1))
