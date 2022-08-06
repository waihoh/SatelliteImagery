import numpy as np
import torch
import cv2
from collections import OrderedDict


class MyFrame:
    def __init__(self, net, loss, device, lr=2e-4, evalmode=False):
        self.device = device
        self.net = net().to(self.device)
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)
        self.loss = loss()
        self.old_lr = lr

        # other initialization
        self.img = None
        self.mask = None
        self.img_id = None

        if evalmode:
            for i in self.net.nn_modules():
                if isinstance(i, torch.nn.BatchNorm2d):
                    i.eval()

    def set_input(self, img_batch, mask_batch=None, img_id=None):
        self.img = img_batch
        self.mask = mask_batch
        self.img_id = img_id

    def test_one_img(self, img):
        pred = self.net.forward(img)
        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0
        mask = pred.squeeze().cpu().data.numpy()
        return mask

    def test_batch(self):
        self.img.to(self.device)
        mask = self.net.forward(self.img).cpu().data.numpy().squeeze(1)
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        return mask, self.img_id

    def test_one_img_from_path(self, path):
        img = cv2.imread(path)
        img = np.array(img, np.float32)/255.0 * 3.2 - 1.6
        img = torch.Tensor(img).to(self.device)
        mask = self.net.forward(img).squeeze().cpu().data.numpy()
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        return mask

    def optimize(self):
        self.img.to(self.device)
        self.mask.to(self.device)
        self.optimizer.zero_grad()
        pred = self.net.forward(self.img)
        loss = self.loss(self.mask, pred)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        # Note that the provided trained model was saved with DataParallel.
        # Remove "module." before loading to self.net
        temp = torch.load(path, map_location=self.device)
        new_state_dict = OrderedDict()
        for k, v in temp.items():
            new_state_dict[k.replace('module.', '')] = v

        self.net.load_state_dict(new_state_dict)

    def update_lr(self, new_lr, mylog, factor=False):
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        str = f"update learning rate: {self.old_lr} -> {new_lr}"
        print >> mylog, str
        print(str)
        self.old_lr = new_lr
