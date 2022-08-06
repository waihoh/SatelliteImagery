import os
import torch
from torch.utils.data import DataLoader
from data import ImageFolder
from loss import dice_bce_loss
from framework import MyFrame
from time import time
from networks.unet import Unet
from networks.dunet import Dunet
from networks.dlinknet import DLinkNet34, DLinkNet34_less_pool, DLinkNet50, DLinkNet101, LinkNet34

# configurations
SHAPE = (1024, 1024)
ROOT = 'dataset/train/'
FOLDER_PATH = os.path.join(os.getcwd(), ROOT)
imagelist = filter(lambda x: x.find('sat') != -1, os.listdir(FOLDER_PATH))
trainlist = list(map(lambda x: x[:-8], imagelist))
device = torch.device('cpu')
batchsize = 4

dataset = ImageFolder(trainlist, FOLDER_PATH)
data_loader = DataLoader(dataset=dataset, batch_size=batchsize, shuffle=True)

# net = Unet, Dunet, DLinkNet34, DLinkNet34_less_pool, DLinkNet50, DLinkNet101, LinkNet34
solver = MyFrame(net=DLinkNet34, loss=dice_bce_loss, device=device, lr=2e-4)

total_epoch = 1
tic = time()
for epoch in range(1, total_epoch+1):
    data_loader_iter = iter(data_loader)
    for img, mask in data_loader_iter:
        solver.set_input(img_batch=img, mask_batch=mask)
        train_loss = solver.optimize()
        print("*"*100)
        print('epoch:', epoch, '    time:', int(time() - tic))
        print('loss:', train_loss)
