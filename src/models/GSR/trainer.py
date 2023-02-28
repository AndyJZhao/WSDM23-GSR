from abc import abstractmethod, ABCMeta
from utils.util_funcs import exp_init, time_logger, print_log
from time import time
import torch as th
from utils.evaluation import eval_logits, eval_classification
from utils.evaluation import eval_logits, save_results
from models.GSR.data_utils import get_stochastic_loader
from ogb.nodeproppred import Evaluator
# Copyright

class NodeClassificationTrainer(metaclass=ABCMeta):
    def __init__(self, model, g, features, optimizer, stopper, loss_func, sup, cf):
        self.trainer = None
        self.model = model
        self.g = g.cpu()
        self.features = features
        self.optimizer = optimizer
        self.stopper = stopper
        self.loss_func = loss_func
        self.cf = cf
        self.device = cf.device
        self.epochs = cf.epochs
        self.n_class = cf.n_class
        self.__dict__.update(sup.__dict__)
        self.train_x, self.val_x, self.test_x = \
            [_.to(cf.device) for _ in [sup.train_x, sup.val_x, sup.test_x]]
        self.labels = sup.labels.to(cf.device)
        self._evaluator = Evaluator(name='ogbn-arxiv')
        self.evaluator = lambda pred, labels: self._evaluator.eval(
            {"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels.view(-1, 1)}
        )["acc"]

    @abstractmethod
    def _train(self):
        return None, None

    @abstractmethod
    def _evaluate(self):
        return None, None

    def run(self):
        for epoch in range(self.epochs):
            t0 = time()
            loss, train_acc = self._train()
            val_acc, test_acc = self._evaluate()
            print_log({'Epoch': epoch, 'Time': time() - t0, 'loss': loss,
                       'TrainAcc': train_acc, 'ValAcc': val_acc, 'TestAcc': test_acc})
            if self.stopper is not None:
                if self.stopper.step(val_acc, self.model, epoch):
                    print(f'Early stopped, loading model from epoch-{self.stopper.best_epoch}')
                    break
        if self.stopper is not None:
            self.model.load_state_dict(th.load(self.stopper.path))
        return self.model

    def eval_and_save(self):
        val_acc, test_acc = self._evaluate()
        res = {'test_acc': f'{test_acc:.4f}', 'val_acc': f'{val_acc:.4f}'}
        if self.stopper is not None: res['best_epoch'] = self.stopper.best_epoch
        save_results(self.cf, res)


class FullBatchTrainer(NodeClassificationTrainer):
    def __init__(self, **kwargs):
        super(FullBatchTrainer, self).__init__(**kwargs)
        self.g = self.g.to(self.device)
        self.features = self.features.to(self.device)

    def _train(self):
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(self.g, self.features)
        loss = self.loss_func(logits[self.train_x], self.labels[self.train_x])
        train_acc = self.evaluator(logits[self.train_x], self.labels[self.train_x])
        loss.backward()
        self.optimizer.step()
        return loss.item(), train_acc

    @th.no_grad()
    def _evaluate(self):
        self.model.eval()
        logits = self.model(self.g, self.features)
        val_acc = self.evaluator(logits[self.val_x], self.labels[self.val_x])
        test_acc = self.evaluator(logits[self.test_x], self.labels[self.test_x])
        return val_acc, test_acc


class StochasticTrainer(NodeClassificationTrainer):
    def __init__(self, **kwargs):
        super(StochasticTrainer, self).__init__(**kwargs)
        self.train_loader = get_stochastic_loader(self.g, self.train_x.cpu(), self.cf.cla_batch_size, self.cf.num_workers)
        self.val_loader = get_stochastic_loader(self.g, self.val_x.cpu(), self.cf.cla_batch_size, self.cf.num_workers)
        self.test_loader = get_stochastic_loader(self.g, self.test_x.cpu(), self.cf.cla_batch_size, self.cf.num_workers)

    def _train(self):
        self.model.train()
        loss_list = []
        pred = th.ones_like(self.labels).to(self.device) * -1
        for batch_id, (input_nodes, output_nodes, blocks) in enumerate(self.train_loader):
            blocks = [b.to(self.cf.device) for b in blocks]
            input_features = self.features[input_nodes].to(self.device)
            output_labels = self.labels[output_nodes].to(self.device)
            out_logits = self.model(blocks, input_features, stochastic=True)
            loss = self.loss_func(out_logits, output_labels)
            pred[output_nodes] = th.argmax(out_logits, dim=1)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch_id + 1 < len(self.train_loader) or len(self.train_loader) == 1:
                # Metrics of the last batch (high variance) shouldn't be added
                loss_list.append(loss.item())
        train_acc, train_f1, train_mif1 = eval_classification(pred[self.train_x], self.train_y, n_class=self.n_class)

        return sum(loss_list) / len(loss_list), train_acc

    @th.no_grad()
    def _evaluate(self):
        def _eval_model(loader, val_x, val_y):
            pred = th.ones_like(self.labels).to(self.device) * -1
            for batch_id, (input_nodes, output_nodes, blocks) in enumerate(loader):
                blocks = [b.to(self.cf.device) for b in blocks]
                input_features = self.features[input_nodes].to(self.device)
                out_logits = self.model(blocks, input_features, stochastic=True)
                pred[output_nodes] = th.argmax(out_logits, dim=1)
            acc, val_f1, val_mif1 = eval_classification(pred[val_x], val_y, n_class=self.n_class)
            return acc

        self.model.eval()
        val_acc = _eval_model(self.val_loader, self.val_x, self.val_y)
        test_acc = _eval_model(self.test_loader, self.test_x, self.test_y)
        return val_acc, test_acc
