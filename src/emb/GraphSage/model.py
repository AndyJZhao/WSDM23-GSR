import dgl
import torch as th
import torch.nn as nn
import dgl.nn.pytorch as dglnn

class SAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, aggregator))
        # for i in range(1, n_layers - 1):
        #     self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, aggregator))
        # self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, aggregator))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x, cf):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.
        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.

        y_list = []
        for l, layer in enumerate(self.layers):

            if l == 0:
                W = layer.fc_neigh
                y0 = W(x.to(cf.device))
                # y0 = self.activation(y0)
                y_list.append(y0.cpu())

            y = th.zeros(g.num_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)
            y_keep = th.zeros(g.num_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                th.arange(g.num_nodes()),
                sampler,
                batch_size=cf.fe_batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=cf.num_workers)

            # for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0].to(cf.device)
                h = x[input_nodes].to(cf.device)
                h = layer(block, h)
                h_keep = h
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()
                y_keep[output_nodes] = h_keep.cpu()
            x = y
            y_list.append(y_keep)
        return y, y_list