from dataset import CropSegmentationDataset
from am4ip.models import CBDNetwork
from am4ip.trainer import BaselineTrainer
from am4ip.losses import TotalLoss
from am4ip.metrics import nMAE

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import Compose, PILToTensor
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage


def train_model(train_loader, epoch, cuda=False, model_path=None):

    model = torch.nn.Sequential(torch.nn.Conv2d(3, 64, 3, padding=1),
                            torch.nn.BatchNorm2d(64),
                            torch.nn.ReLU(inplace=True),
                            torch.nn.Dropout2d(0.2),
                            torch.nn.Conv2d(64, 64, 3, padding=1),
                            torch.nn.BatchNorm2d(64),
                            torch.nn.ReLU(inplace=True),
                            torch.nn.Dropout2d(0.2),
                            torch.nn.Conv2d(64, dataset.get_class_number(), 3, padding=1))
    
    if model_path is None:
        loss = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        trainer = BaselineTrainer(model=model, loss=loss, optimizer=optimizer, use_cuda=cuda)
        trainer.fit(train_loader, epoch=epoch)
        return model
    
    else:
        model.load_state_dict(torch.load(model_path))
        return model
    

if __name__ == "__main__":

    # Parameters
    batch_size = 10
    lr = 1e-3
    epoch = 1
    cuda = False

    # Dataset
    transform = Compose([PILToTensor(),
                        lambda z: z.to(dtype=torch.float32) / 127.5 - 1  # Normalize between -1 and 1
                        ])

    target_transform = Compose([PILToTensor(),
                                lambda z: z.to(dtype=torch.long).squeeze(0)  # Remove channel dimension
                                ])

    dataset = CropSegmentationDataset(transform=transform, target_transform=target_transform, augmentations=True) # Augmentation
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print("Dataset size:", len(dataset))

    # Training (use model_path to load pretrained model)
    model = train_model(train_loader, epoch, cuda=cuda) #, model_path="model.pth")

    # Save model
    torch.save(model.state_dict(), "model.pth") 

    # Load test dataset
    test_dataset = CropSegmentationDataset(set_type="test", transform=transform, target_transform=target_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(model.eval())

    # print("Computing metrics...")
    # Compute metrics
    # nmae = nMAE()
    # for i, (x, y) in enumerate(test_loader):
    #     if cuda:
    #         x = x.cuda()
    #         y = y.cuda()
    #     x = x.unsqueeze(0)
    #     y = y.unsqueeze(0)
    #     y_pred = model(x)
    #     nmae.update(y_pred, y)
    # print(f"nMAE: {nmae.compute()}")
    # print("Metrics computed.")

    print("job's done.")
