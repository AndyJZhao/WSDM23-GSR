import os.path as osp
import sys

sys.path.append((osp.abspath(osp.dirname(__file__)).split('src')[0] + 'src'))

from utils.early_stopper import EarlyStopping
from utils.evaluation import eval_logits, eval_and_save

from models.MGSL_PretrainedFeatEmb.MGSL import MGSL_PretrainedFeatEmb, NP_Encoder
from models.MGSL_PretrainedFeatEmb.config import MGSL_PretrainedFeatConfig
from models.MGSL_PretrainedFeatEmb.data_loader import *
from models.MGSL_PretrainedFeatEmb.GraphMSGenerator import *
from utils.util_funcs import exp_init, time_logger, print_log, get_ip
from time import time
from models.MGSL_PretrainedFeatEmb.cl_utils import *


@time_logger
def train_mgsl_pretrained_feat(args):
    exp_init(args.seed, gpu_id=args.gpu)
    # ! config
    cf = MGSL_PretrainedFeatConfig(args)
    cf.device = th.device("cuda:0" if args.gpu >= 0 else "cpu")

    # ! Load Graph
    g, features, cf.n_feat, cf.n_class, labels, train_x, val_x, test_x = \
        preprocess_data(cf.dataset, cf.train_percentage, cf.device)
    g = graph_normalization(g, args.gpu >= 0)
    g = get_node_property_prior(g, cf, features, labels, train_x)

    train_y, val_y, test_y = labels[train_x], labels[val_x], labels[test_x]

    # ! Train Init
    print(f'{cf}\nStart training..')
    cla_loss = th.nn.CrossEntropyLoss()
    model = MGSL_PretrainedFeatEmb(g, cf)
    model.to(cf.device)

    print(model)
    optimizer = th.optim.Adam(
        model.parameters(), lr=cf.lr, weight_decay=cf.weight_decay)
    stopper = EarlyStopping(patience=cf.early_stop, path=cf.checkpoint_file)

    # ! Train Phase 1: Pretrain
    print(f'>>>> PHASE 1 <<<<< Pretraining node properties...')

    if os.path.exists(cf.pretrain_model_ckpt):
        model.np_encoder.load_state_dict(th.load(cf.pretrain_model_ckpt, map_location=cf.device))
        print(f'Pretrain embedding loaded from {cf.pretrain_model_ckpt}')
    else:
        # Construct virtual relation triples
        if cf.enc_conf.split('_')[0] == 'MLP':
            model_ema = MGSL_PretrainedFeatEmb(g, cf).to(cf.device)
            moment_update(model, model_ema, 0)  # Copy
            moco_memories = {v: MemoryMoCo(cf.n_hidden, cf.nce_k,  # Single-view contrast
                                           cf.nce_t, device=cf.device).to(cf.device)
                             for v in cf.views}
            criterion = NCESoftmaxLoss(cf.device)
            pretrain_loader = mlp_pretrain_loader(g, train_x, cf.batch_size, cf.p_epochs)
            for epoch_id, batch in enumerate(pretrain_loader):
                t0 = time()
                # print(f'Model Para: {model.np_encoder.encoder.F.layers[0].weight[0,:4].cpu().detach().numpy().tolist()}')
                # print(f'ModelEma Para: {model_ema.np_encoder.encoder.F.layers[0].weight[0, :4].cpu().detach().numpy().tolist()}')
                ssl_q_list, ssl_k_list, cla_nodes = lot_to_tol(batch)
                # ssl_q_list, ssl_k_list, cla_nodes = [th.LongTensor(_).to(cf.device) for _ in [ssl_q_list, ssl_k_list, cla_nodes]]
                model.train()
                # ===================Moco forward=====================
                q_emb = model.np_encoder(g, ssl_q_list)
                with th.no_grad():
                    k_emb = model_ema.np_encoder(g, ssl_k_list)
                intra_out, inter_out = [], []

                for tgt_view, memory in moco_memories.items():
                    for src_view in cf.views:
                        if src_view == tgt_view:
                            intra_out.append(memory(
                                q_emb[f'{tgt_view}'], k_emb[f'{tgt_view}']))
                        else:
                            inter_out.append(memory(
                                q_emb[f'{src_view}->{tgt_view}'], k_emb[f'{tgt_view}']))

                # ===================backward=====================
                # ! Self-Supervised Learning
                intra_loss = th.stack([criterion(out_) for out_ in intra_out]).mean()
                inter_loss = th.stack([criterion(out_) for out_ in inter_out]).mean()
                # ! Loss Fusion
                loss_tensor = th.stack([intra_loss, inter_loss])
                loss_weights = th.tensor([float(_) for _ in cf.pretrain_weights.split('_')], device=cf.device)
                loss = th.dot(loss_weights, loss_tensor)
                # ! Semi-Supervised Learning
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                moment_update(model, model_ema, cf.momentum_factor)
                print_log(epoch_id, {'Time': time() - t0, 'loss': loss.item()})
            th.save(model.np_encoder.state_dict(), cf.pretrain_model_ckpt)
        elif cf.encoder.split('_')[0] == 'GCN':
            pass

    # ! Train Phase 2: Graph Structure Coarse Filtering
    model.coarse_filter_cand_graphs()
    # ! Train Phase 3: Graph Structure Learning
    print(f'>>>> PHASE 2 <<<<< Graph Structure Learning and Classification')
    for epoch in range(cf.epochs):
        t0 = time()
        model.train()
        logits, weights = model(features)
        loss = cla_loss(logits[train_x], train_y)
        train_acc, train_f1, train_mif1 = eval_logits(logits, train_x, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Valid
        model.eval()
        with th.no_grad():
            logits, _ = model(features)
            val_acc, val_f1, val_mif1 = eval_logits(logits, val_x, val_y)
            test_acc, test_maf1, test_mif1 = eval_logits(logits, test_x, test_y)
        log_dict = {**{'Time': time() - t0, 'loss': loss.item(), 'TrainAcc': train_acc,
                       'ValAcc': val_acc, 'TestAcc': test_acc}, **weights}
        print_log(epoch, log_dict)

        if cf.early_stop > 0:
            if stopper.step(val_acc, model, epoch):
                print(f'Early stopped, loading model from epoch-{stopper.best_epoch}')
                break
    if cf.early_stop > 0:
        model.load_state_dict(th.load(cf.checkpoint_file))
    logits, weights = model(features)
    # ! Save interested intermediate results to config file
    cf.att_weight = str(weights)
    cf.update__model_conf_list(['att_weight'])
    eval_and_save(cf, logits, test_x, test_y, val_x, val_y, stopper)
    return cf


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training settings")
    dataset = 'pubmed'
    dataset = 'citeseer'
    dataset = 'cora'
    # ! Settings
    parser.add_argument("-g", "--gpu", default=1, type=int, help="GPU id to use.")
    parser.add_argument("-d", "--dataset", type=str, default=dataset)
    parser.add_argument("-b", "--block_log", action="store_true", help="block log or not")
    parser.add_argument("-t", "--train_percentage", default=15, type=int)
    parser.add_argument("-e", "--early_stop", default=100, type=int)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--seed", default=0)
    args = parser.parse_args()

    if '192.168.0' in get_ip():
        args.gpu = -1
    # ! Train
    cf = train_mgsl_pretrained_feat(args)
