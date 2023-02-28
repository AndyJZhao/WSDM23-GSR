import torch
import argparse
import dgl
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import os
import random
import time
import numpy as np

from emb.LINE_dgl.reading_data import LineDataset
from emb.LINE_dgl.model import SkipGramModel
from emb.LINE_dgl.utils import thread_wrapped_func, sum_up_params, check_args


class LineTrainer:
    def __init__(self, args):
        """ Initializing the trainer with the input arguments """
        self.args = args
        self.dataset = LineDataset(
            G = args.G,
            batch_size=args.se_batch_size,
            negative=args.se_negative,
            gpus=args.se_gpus,
            fast_neg=args.se_fast_neg,
            num_samples=args.se_num_samples * 1000000,
        )

        self.emb_size = self.dataset.G.number_of_nodes()
        self.emb_model = None

    def init_device_emb(self):
        """ set the device before training
        will be called once in fast_train_mp / fast_train
        """
        choices = sum([self.args.se_only_gpu, self.args.se_only_cpu, self.args.se_mix])
        assert choices == 1, "Must choose only *one* training mode in [only_cpu, only_gpu, mix]"

        # initializing embedding on CPU
        self.emb_model = SkipGramModel(
            emb_size=self.emb_size,
            emb_dimension=self.args.se_dim,
            batch_size=self.args.se_batch_size,
            only_cpu=self.args.se_only_cpu,
            only_gpu=self.args.se_only_gpu,
            only_fst=self.args.se_only_fst,
            only_snd=self.args.se_only_snd,
            mix=self.args.se_mix,
            neg_weight=self.args.se_neg_weight,
            negative=self.args.se_negative,
            lr=self.args.se_lr,
            lap_norm=self.args.se_lap_norm,
            fast_neg=self.args.se_fast_neg,
            record_loss=self.args.se_print_loss,
            async_update=self.args.se_async_update,
            num_threads=self.args.se_num_threads,
        )

        torch.set_num_threads(self.args.se_num_threads)
        if self.args.se_only_gpu:
            print("Run in 1 GPU")
            assert self.args.se_gpus[0] >= 0
            self.emb_model.all_to_device(self.args.se_gpus[0])
        elif self.args.se_mix:
            print("Mix CPU with %d GPU" % len(self.args.gpus))
            if len(self.args.se_gpus) == 1:
                assert self.args.se_gpus[0] >= 0, 'mix CPU with GPU should have avaliable GPU'
                self.emb_model.set_device(self.args.gpus[0])
        else:
            print("Run in CPU process")

    def train(self):
        """ train the embedding """
        self.fast_train()


    def fast_train(self):
        """ fast train with dataloader with only gpu / only cpu"""
        self.init_device_emb()

        if self.args.se_async_update:
            self.emb_model.share_memory()
            self.emb_model.create_async_update()

        sum_up_params(self.emb_model)

        sampler = self.dataset.create_sampler(0)

        dataloader = DataLoader(
            dataset=sampler.seeds,
            batch_size=self.args.se_batch_size,
            collate_fn=sampler.sample,
            shuffle=False,
            drop_last=False,
            num_workers=self.args.se_num_sampler_threads,
        )

        num_batches = len(dataloader)
        print("num batchs: %d\n" % num_batches)

        start_all = time.time()
        start = time.time()
        with torch.no_grad():
            for i, edges in enumerate(dataloader):
                if self.args.se_fast_neg:
                    self.emb_model.fast_learn(edges)
                else:
                    # do negative sampling
                    bs = edges.size()[0]
                    neg_nodes = torch.LongTensor(
                        np.random.choice(self.dataset.neg_table,
                                         bs * self.args.se_negative,
                                         replace=True))
                    self.emb_model.fast_learn(edges, neg_nodes=neg_nodes)

                # if i > 0 and i % self.args.se_print_interval == 0:
                #     if self.args.se_print_loss:
                #         if self.args.se_only_fst:
                #             print("Batch %d time: %.2fs fst-loss: %.4f" \
                #                   % (i, time.time() - start, -sum(self.emb_model.loss_fst) / self.args.se_print_interval))
                #         elif self.args.se_only_snd:
                #             print("Batch %d time: %.2fs snd-loss: %.4f" \
                #                   % (i, time.time() - start, -sum(self.emb_model.loss_snd) / self.args.se_print_interval))
                #         else:
                #             print("Batch %d time: %.2fs fst-loss: %.4f snd-loss: %.4f" \
                #                   % (i, time.time() - start, \
                #                      -sum(self.emb_model.loss_fst) / self.args.se_print_interval, \
                #                      -sum(self.emb_model.loss_snd) / self.args.se_print_interval))
                #         self.emb_model.loss_fst = []
                #         self.emb_model.loss_snd = []
                #     else:
                #         print("Batch %d, training time: %.2fs" % (i, time.time() - start))
                #     start = time.time()

            if self.args.se_async_update:
                self.emb_model.finish_async_update()

        self.emb = self.emb_model.get_embedding()
        self.emb = torch.Tensor(self.emb).cpu()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Implementation of LINE.")
    # input files
    ## personal datasets
    parser.add_argument('--data_file', type=str, default='cora-graph.bin',
                        help="path of dgl graphs")
    ## ogbl datasets
    parser.add_argument('--ogbl_name', type=str, default='ogbl-collab-graph.bin',
                        help="name of ogbl dataset, e.g. ogbl-ddi")
    parser.add_argument('--load_from_ogbl', default=False, action="store_true",
                        help="whether load dataset from ogbl")
    ## ogbn datasets
    parser.add_argument('--ogbn_name', type=str,
                        help="name of ogbn dataset, e.g. ogbn-proteins")
    parser.add_argument('--load_from_ogbn', default=False, action="store_true",
                        help="whether load dataset from ogbn")

    # output files
    parser.add_argument('--save_in_pt', default=False, action="store_true",
                        help='Whether save dat in pt format or npy')
    parser.add_argument('--output_emb_file', type=str, default="emb.npy",
                        help='path of the output npy embedding file')

    # model parameters
    parser.add_argument('--dim', default=32, type=int,
                        help="embedding dimensions")
    parser.add_argument('--num_samples', default=1, type=int,
                        help="number of samples during training (million)")
    parser.add_argument('--negative', default=1, type=int,
                        help="negative samples for each positve node pair")
    parser.add_argument('--batch_size', default=128, type=int,
                        help="number of edges in each batch")
    parser.add_argument('--neg_weight', default=1., type=float,
                        help="negative weight")
    parser.add_argument('--lap_norm', default=0.01, type=float,
                        help="weight of laplacian normalization")

    # training parameters
    parser.add_argument('--only_fst', default=False, action="store_true",
                        help="only do first-order proximity embedding")
    parser.add_argument('--only_snd', default=False, action="store_true",
                        help="only do second-order proximity embedding")
    parser.add_argument('--print_interval', default=100, type=int,
                        help="number of batches between printing")
    parser.add_argument('--print_loss', default=True, action="store_true",
                        help="whether print loss during training")
    parser.add_argument('--lr', default=0.2, type=float,
                        help="learning rate")

    # optimization settings
    parser.add_argument('--mix', default=True, action="store_true",
                        help="mixed training with CPU and GPU")
    parser.add_argument('--gpus', type=int, default=[0], nargs='+',
                        help='a list of active gpu ids, e.g. 0, used with --mix')
    parser.add_argument('--only_cpu', default=False, action="store_true",
                        help="training with CPU")
    parser.add_argument('--only_gpu', default=False, action="store_true",
                        help="training with a single GPU (all of the parameters are moved on the GPU)")

    parser.add_argument('--async_update', default=False, action="store_true",
                        help="mixed training asynchronously, recommend not to use this")
    parser.add_argument('--fast_neg', default=True, action="store_true",
                        help="do negative sampling inside a batch")
    parser.add_argument('--num_threads', default=2, type=int,
                        help="number of threads used for each CPU-core/GPU")
    parser.add_argument('--num_sampler_threads', default=2, type=int,
                        help="number of threads used for sampling")

    args = parser.parse_args()

    if args.async_update:
        assert args.mix, "--async_update only with --mix"

    start_time = time.time()
    trainer = LineTrainer(args)
    trainer.train()
    print("Total used time: %.2f" % (time.time() - start_time))
