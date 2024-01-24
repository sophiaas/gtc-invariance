from torch_tools.trainer import Trainer
import torch
from collections import OrderedDict


class GTrainer(Trainer):
    
    def __init__(
        self,
        model,
        loss,
        optimizer,
        logger=None,
        scheduler=None,
        regularizer=None,
        normalizer=None,
    ):
        super().__init__(
            model=model,
            loss=loss,
            optimizer=optimizer,
            logger=logger,
            scheduler=scheduler,
            regularizer=regularizer,
            normalizer=normalizer
        )
        
    def __getstate__(self):
        d = self.__dict__
        self_dict = {k : d[k] for k in d if k != '_modules'}
        module_dict = OrderedDict({'loss': self.loss})
        if self.regularizer is not None:
            module_dict["regularizer"] = self.regularizer
        if self.normalizer is not None:
            module_dict["normalizer"] = self.normalizer
        if self.scheduler is not None:
            module_dict["scheduler"] = self.scheduler
        self_dict['_modules'] = module_dict
        return self_dict

    def __setstate__(self, state):
        self.__dict__ = state
        
    def step(self, data_loader, grad=True):
        """Compute a single step of training.

        Parameters
        ----------
        data_loader : torch.utils.data.dataloader.DataLoader
        grad : boolean
            required argument to switch between training and evaluation

        Returns
        -------
        log_dict : dictionary with losses to be logged by the trainer/logger
            format - {'total_loss': total_loss, 'l1_penalty': l1_penalty, ...}
            Dictionary must contain a key called `total_loss`

        """
        log_dict = {"loss": 0, 
                    "reg_loss": 0, 
                    "total_loss": 0,
                    "accuracy": 0}
        for i, (x, labels) in enumerate(data_loader):
            loss = 0
            reg_loss = 0
            total_loss = 0
            accuracy = 0

            x = x.to(self.model.device)
            labels = labels.to(self.model.device)

            if grad:
                self.optimizer.zero_grad()
                out = self.model.forward(x)
            else:
                with torch.no_grad():
                    out = self.model.forward(x)

            # Compute loss term without regularization terms (e.g. classification loss)
            _, prediction = torch.max(out.data, 1)
            accuracy += (prediction == labels).sum().item() / len(x)
            loss += self.loss(out, labels)
            log_dict["loss"] += loss
            log_dict["accuracy"] += accuracy
            total_loss += loss

            # Compute regularization penalty terms (e.g. sparsity, l2 norm, etc.)
            if self.regularizer is not None:
                reg_variable_dict = {
                    "x": x,
                    "out": out,
                } | dict(self.model.named_parameters()) # Must use named parameters rather than state_dict to preserve grads

                reg_loss += self.regularizer(reg_variable_dict)
                log_dict["reg_loss"] += reg_loss
                total_loss += reg_loss

            if grad:
                total_loss.backward()
                self.optimizer.step()
                
            if self.normalizer is not None:
                self.normalizer(dict(self.model.named_parameters()))

            log_dict["total_loss"] += total_loss

        # Normalize loss terms for the number of samples/batches in the epoch (optional)
        n_samples = len(data_loader)
        for key in log_dict.keys():
            log_dict[key] /= n_samples

        plot_variable_dict = {"model": self.model}

        return log_dict, plot_variable_dict
    
    def print_update(self, result_dict_train, result_dict_val=None):

        update_string = "Epoch {} ||  N Examples {} || Train Total Loss {:0.5f} || Train Accuracy {:0.5f}".format(
            self.epoch, self.n_examples, result_dict_train["total_loss"], result_dict_train["accuracy"]
        )
        if result_dict_val:
            update_string += " || Validation Total Loss {:0.5f} || Validation Accuracy {:0.5f}".format(
                result_dict_val["total_loss"], result_dict_val["accuracy"]
            )
        print(update_string)
