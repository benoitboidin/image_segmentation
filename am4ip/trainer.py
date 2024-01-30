
from typing import Callable, List
import torch
import torch.utils.data as data


class BaselineTrainer:
    def __init__(self, model: torch.nn.Module,
                 loss: Callable,
                 optimizer: torch.optim.Optimizer,
                 use_cuda=False,
                 use_mps=False):
        self.loss = loss
        self.use_cuda = use_cuda
        self.use_mps = use_mps
        self.optimizer = optimizer

        if use_cuda:
            self.model = model.to(device="cuda:0")
        if use_mps:
            torch.set_default_device("mps")
            mps_device = torch.device("mps")
            self.model = model.to(device=mps_device)
        else:
            self.model = model

    def print_progress(iteration, total, prefix='', suffix='', length=50, fill='█'):
        percent = ("{0:.1f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
        if iteration == total: 
            print()

    def fit(self, train_data_loader: data.DataLoader,
            epoch: int):
        avg_loss = 0.
        self.model.training = True
        for e in range(epoch):
            print(f"Start epoch {e+1}/{epoch}")
            n_batch = 0

            for i, (input_image, target) in enumerate(train_data_loader):
                # Reset previous gradients
                self.optimizer.zero_grad()  

                # Move data to cuda is necessary:
                if self.use_cuda:
                    input_image = input_image.cuda()
                    target = target.cuda()
                if self.use_mps:
                    input_image = input_image.to(device="mps")
                    target = target.to(device="mps")

                # Make forward
                # TODO change this part to fit your loss function
                loss = self.loss(self.model.forward(input_image), target)
                loss.backward()

                # Adjust learning weights
                self.optimizer.step()
                avg_loss += loss.item()
                n_batch += 1

                print(f"\r{i+1}/{len(train_data_loader)}: loss = {avg_loss / n_batch}", end='')
            print()

        return avg_loss
