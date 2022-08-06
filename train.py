import os
import torch
from torch.utils.data import DataLoader
from networks.dlinknet import DLinkNet34
from framework import MyFrame
from loss import dice_bce_loss
from data import ImageFolder
from time import time

SHAPE = (1024, 1024)
ROOT = 'dataset/train/'
FOLDER_PATH = os.path.join(os.getcwd(), ROOT)
imagelist = filter(lambda x: x.find('sat') != -1, os.listdir(FOLDER_PATH))
trainlist = list(map(lambda x: x[:-8], imagelist))
NAME = 'log01_dink34_new'
device = torch.device('cpu')
batchsize = 4

dataset = ImageFolder(trainlist, FOLDER_PATH)
data_loader = DataLoader(dataset=dataset, batch_size=batchsize, shuffle=True) # num_workers=4

solver = MyFrame(net=DLinkNet34, loss=dice_bce_loss, device=device, lr=2e-4)
log_folder = "logs"
if not os.path.exists(log_folder):
    os.mkdir(log_folder)
mylog = open(os.path.join(log_folder, (NAME + '.log')), 'w')

tic = time()
no_optim = 0  # counter for tracking no improvement in training

# Configure training
total_epoch = 300
train_epoch_best_loss = 100

for epoch in range(1, total_epoch + 1):
    data_loader_iter = iter(data_loader)
    train_epoch_loss = 0  # cumulative
    for img, mask in data_loader_iter:
        solver.set_input(img_batch=img, mask_batch=mask)
        train_loss = solver.optimize()
        train_epoch_loss += train_loss

    train_epoch_loss /= len(data_loader_iter)

    print("*" * 20, file=mylog)
    print("Epoch:", epoch, '     time:', int(time() - tic), file=mylog)
    print("train_loss:", train_epoch_loss, file=mylog)

    if train_epoch_loss >= train_epoch_best_loss:
        no_optim += 1
    else:
        no_optim = 0  # reset counter
        train_epoch_best_loss = train_epoch_loss
        solver.save('weights/' + NAME + ".th")

    if no_optim > 6:
        print(f'Early stop at {epoch} epoch', file=mylog)
        break

    if no_optim > 3:
        if solver.old_lr < 5e-7:
            break
        solver.load('weights/' + NAME + ".th")
        solver.update_lr(new_lr=5, mylog=mylog, factor=True)

    mylog.flush()

print("COMPLETED")
mylog.close()
