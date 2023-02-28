import os.path as osp
import sys

sys.path.append((osp.abspath(osp.dirname(__file__)).split('src')[0] + 'src'))

from utils.early_stopper import EarlyStopping
from utils.evaluation import eval_logits, eval_and_save

from models.MGSL_KGE.MGSL_KGE import MGSL_KGE
from models.MGSL_KGE.config import MGSL_KGEConfig
from models.MGSL_KGE.data_loader import *
from models.MGSL_KGE.GraphMSGenerator import *
from utils.util_funcs import exp_init, time_logger, print_log
from numpy import mean
from time import time


@time_logger
def train_MGSL_KGE(args):
    exp_init(args.seed, gpu_id=args.gpu)
    # ! config
    cf = MGSL_KGEConfig(args)
    cf.device = th.device("cuda:0" if args.gpu >= 0 else "cpu")

    # ! Load Graph
    g, features, cf.n_feat, cf.n_class, labels, train_x, val_x, test_x = \
        preprocess_data(cf.dataset, cf.train_percentage, cf.device)
    g = graph_normalization(g, args.gpu >= 0)
    # TODO: Check Labels == -1
    # TODO: 和温博 check 阶数
    g = gen_meta_sources(g, cf, features, labels, train_x)

    train_y, val_y, test_y = labels[train_x], labels[val_x], labels[test_x]

    # ! Train Init
    print(f'{cf}\nStart training..')
    cla_loss = th.nn.CrossEntropyLoss()
    model = MGSL_KGE(g, cf)
    model.to(cf.device)
    print(model)
    optimizer = th.optim.Adam(
        model.parameters(), lr=cf.lr, weight_decay=cf.weight_decay)
    stopper = EarlyStopping(patience=cf.early_stop, path=cf.checkpoint_file)

    # ! Train Phase 1: Pretrain
    print(f'>>>> PHASE 1 <<<<< Pretraining graph meta structure...')
    if os.path.exists(cf.pretrain_model_ckpt):
        model.gs_generator.KGE.model.load_state_dict(th.load(cf.pretrain_model_ckpt, map_location=cf.device))
        print(f'Pretrain embedding loaded from {cf.pretrain_model_ckpt}')
    else:
        # Construct virtual relation triples
        pretrain_loader = th.utils.data.DataLoader(
            dataset=GraphMetaTripleDataset(g, cf),
            batch_size=cf.batch_size,
            collate_fn=ke_triple_collate_fn(),
            num_workers=cf.num_workers,
            worker_init_fn=worker_init_fn
        )
        for batch_id, batch in enumerate(pretrain_loader):
            t0 = time()
            model.train()
            loss = model.gs_generator(convert_to_openke_batch(batch, cf.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print_log(batch_id, {'Time': time() - t0, 'loss': loss.item()})
        th.save(model.gs_generator.KGE.model.state_dict(), cf.pretrain_model_ckpt)
    print(model.gs_generator.KGE.model.ent_embeddings.weight[0, :5])

    # ! Train Phase 2: Graph Structure Coarse Filtering
    model.coarse_filter_cand_graphs()
    # ! Train Phase 3: Finetune
    print(f'>>>> PHASE 2 <<<<< Finetuning Graph Meta Structure')
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
    parser.add_argument("-g", "--gpu", default=0, type=int, help="GPU id to use.")
    parser.add_argument("-d", "--dataset", type=str, default=dataset)
    parser.add_argument("-b", "--block_log", action="store_true", help="block log or not")
    parser.add_argument("-t", "--train_percentage", default=15, type=int)
    parser.add_argument("-e", "--early_stop", default=100, type=int)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--meta_types", default='FSL', type=str)
    parser.add_argument("--seed", default=0)
    args = parser.parse_args()
    if '192.168.0' in uf.get_ip():
        args.gpu = -1
    # ! Train
    cf = train_MGSL_KGE(args)
