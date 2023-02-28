import os.path as osp
import sys

sys.path.append((osp.abspath(osp.dirname(__file__)).split('src')[0] + 'src'))

from utils.early_stopper import EarlyStopping

from models.GSR.GSR import *
from models.GSR.config import GSRConfig
from models.GSR.data_utils import *
from utils.util_funcs import exp_init, time_logger, print_log, get_ip
from models.GSR.cl_utils import *
from utils.proj_settings import P_EPOCHS_SAVE_LIST
from models.GSR.trainer import *
from utils.conf_utils import *
from models.GSR.SEAL.model import GCN, DGCNN
from torch.nn import BCEWithLogitsLoss
from dgl import NID, EID
from models.GSR.SEAL.seal_utils import construct_negative_graph, parse_arguments
from models.GSR.SEAL.sampler import SEALData
from dgl.dataloading import GraphDataLoader
from models.GSR.SEAL.subgraph import extract_subgraph

import argparse
from torch_poly_lr_decay import PolynomialLRDecay

def train(model, dataloader, loss_fn, optimizer, device, num_graphs=32, total_graphs=None):
    model.train()
    total_loss = 0
    for g, labels in tqdm(dataloader, ncols=100):

        g = g.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(g, g.ndata['z'], g.ndata[NID], g.edata[EID])
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * num_graphs

    return total_loss / total_graphs


@time_logger
def train_GSR_seal(args):
    exp_init(args.seed, gpu_id=args.gpu)
    # ! Config
    cf = GSRConfig(args)
    cf.device = th.device("cuda:0" if args.gpu >= 0 else "cpu")

    seal_args = parse_arguments()
    seal_args.random_seed = cf.seed

    # ! Load Graph
    g, features, cf.n_feat, cf.n_class, labels, train_x, val_x, test_x = \
        preprocess_data(cf.dataset, cf.train_percentage)
    num_nodes = g.num_nodes()

    train_data, val_data, test_data = extract_subgraph(seal_args, g, features)
    train_graphs = len(train_data.graph_list)

    train_loader = GraphDataLoader(train_data, batch_size=seal_args.batch_size, num_workers=seal_args.num_workers)
    val_loader = GraphDataLoader(val_data, batch_size=seal_args.batch_size, num_workers=seal_args.num_workers)
    test_loader = GraphDataLoader(test_data, batch_size=seal_args.batch_size, num_workers=seal_args.num_workers)

    # feat = {'F': features, 'S': get_structural_feature(g, cf)}
    # cf.feat_dim = {v: feat.shape[1] for v, feat in feat.items()}
    # supervision = SimpleObject({'train_x': train_x, 'val_x': val_x, 'test_x': test_x, 'labels': labels})
    # ! Train Init
    print(f'{cf}\nStart training..')
    ### pretrain model initialization
    p_model = GCN(num_layers=seal_args.num_layers,
                hidden_units=seal_args.hidden_units,
                gcn_type=seal_args.gcn_type,
                pooling_type=seal_args.pooling,
                node_attributes=features,
                edge_weights=None,
                node_embedding=None,
                use_embedding=True,
                num_nodes=num_nodes,
                dropout=seal_args.dropout)

    p_model = p_model.to(cf.device)
    parameters = p_model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=seal_args.lr)
    loss_fn = BCEWithLogitsLoss()
    print(p_model)
    # ! Train Phase 1: Pretrain

    for epoch in range(seal_args.epochs):
        loss = train(model=p_model,
                     dataloader=train_loader,
                     loss_fn=loss_fn,
                     optimizer=optimizer,
                     device=cf.device,
                     num_graphs=seal_args.batch_size,
                     total_graphs=train_graphs)

    # p_model.eval()
    # logits = p_model(g, g.ndata['z'], g.ndata[NID], g.edata[EID])


    th.save(p_model.state_dict(), cf.pretrain_model_ckpt)


    ### how the extract the edge index and refine the graph
    # ! Train Phase 2: Graph Structure Refine
    print(f'>>>> PHASE 2 - Graph Structure Refine <<<<< ')
    if cf.p_epochs <= 0 or cf.add_ratio + cf.rm_ratio == 0:
        print('Use original graph!')
        g_new = g
    else:
        if os.path.exists(cf.refined_graph_file):
            print(f'Refined graph loaded from {cf.refined_graph_file}')
            g_new = dgl.load_graphs(cf.refined_graph_file)[0][0]
        else:
            g_new = p_model.refine_graph(g, feat)
            dgl.save_graphs(cf.refined_graph_file, [g_new])


    #### do we need to copy the parameter
    # ! Train Phase 3:  Node Classification
    f_model = GSR_finetune(cf).to(cf.device)
    print(f_model)
    # Copy parameters
    if cf.p_epochs > 0:
        para_copy(f_model, p_model.encoder.F, paras_to_copy=['conv1.weight', 'conv1.bias'])
    optimizer = th.optim.Adam(f_model.parameters(), lr=cf.lr, weight_decay=cf.weight_decay)
    stopper = EarlyStopping(patience=cf.early_stop, path=cf.checkpoint_file) if cf.early_stop else None
    del g, feat, p_model
    th.cuda.empty_cache()

    print(f'>>>> PHASE 3 - Node Classification <<<<< ')
    trainer_func = FullBatchTrainer
    trainer = trainer_func(model=f_model, g=g_new, features=features, sup=supervision, cf=cf,
                           stopper=stopper, optimizer=optimizer, loss_func=th.nn.CrossEntropyLoss())
    trainer.run()
    trainer.eval_and_save()

    return cf


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training settings")
    # dataset = 'citeseer'
    # dataset = 'arxiv'
    dataset = 'cora'
    # dataset = 'flickr'
    # dataset = 'airport'
    # dataset = 'blogcatalog'print
    # ! Settings
    parser.add_argument("-g", "--gpu", default=1, type=int, help="GPU id to use.")
    parser.add_argument("-d", "--dataset", type=str, default=dataset)
    parser.add_argument("-t", "--train_percentage", default=0, type=int)
    parser.add_argument("-e", "--early_stop", default=100, type=int)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--seed", default=0)
    args = parser.parse_args()
    #
    # if '192.168.0' in get_ip():
    #     args.gpu = -1
    #     args.dataset = args.dataset if args.dataset != 'arxiv' else 'cora'
    # ! Train
    train_GSR_seal(args)

# python /home/zja/PyProject/MGSL/src/models/GSR/trainGSR.py -darxiv
