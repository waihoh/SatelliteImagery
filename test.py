import torch
import os
import cv2
import numpy as np
from networks.dlinknet import DLinkNet34
from time import time
from collections import OrderedDict

device = torch.device('cpu')
batchsize = 4


class TTAFrame:
    def __init__(self, net):
        self.net = net().to(device)
        # self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))

    def test_one_img_from_path(self, path, evalmode=True):
        if evalmode:
            self.net.eval()

        if batchsize >= 8:
            return self.test_one_img_from_path_1(path)
        elif batchsize >= 4:
            return self.test_one_img_from_path_2(path)
        elif batchsize >= 2:
            return self.test_one_img_from_path_4(path)

    def test_one_img_from_path_4(self, path):
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.array(img1)[:, :, ::-1]
        img4 = np.array(img2)[:, :, ::-1]

        img1 = img1.transpose(0, 3, 1, 2)
        img2 = img2.transpose(0, 3, 1, 2)
        img3 = img3.transpose(0, 3, 1, 2)
        img4 = img4.transpose(0, 3, 1, 2)

        img1 = torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).to(device)
        img2 = torch.Tensor(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6).to(device)
        img3 = torch.Tensor(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6).to(device)
        img4 = torch.Tensor(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6).to(device)

        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, ::-1] + maskc[:, :, ::-1] + maskd[:, ::-1, ::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]

        return mask2

    def test_one_img_from_path_2(self, path):
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        img5 = img3.transpose(0, 3, 1, 2)
        img5 = np.array(img5, np.float32) / 255.0 * 3.2 - 1.6
        img5 = torch.Tensor(img5).to(device)
        img6 = img4.transpose(0, 3, 1, 2)
        img6 = np.array(img6, np.float32) / 255.0 * 3.2 - 1.6
        img6 = torch.Tensor(img6).to(device)

        maska = self.net.forward(img5).squeeze().cpu().data.numpy()  # .squeeze(1)
        maskb = self.net.forward(img6).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, :, ::-1]
        mask2 = mask1[:2] + mask1[2:, ::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]

        return mask3

    def test_one_img_from_path_1(self, path):
        img = cv2.imread(path)  # .transpose(2,0,1)[None]

        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        img5 = np.concatenate([img3, img4]).transpose(0, 3, 1, 2)
        img5 = np.array(img5, np.float32) / 255.0 * 3.2 - 1.6
        img5 = torch.Tensor(img5).to(device)

        mask = self.net.forward(img5).squeeze().cpu().data.numpy()  # .squeeze(1)
        mask1 = mask[:4] + mask[4:, :, ::-1]
        mask2 = mask1[:2] + mask1[2:, ::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]

        return mask3

    def load(self, path):
        # Note that the provided trained model was saved with DataParallel.
        # Remove "module." before loading to self.net
        temp = torch.load(path, map_location=device)
        new_state_dict = OrderedDict()
        for k, v in temp.items():
            new_state_dict[k.replace('module.', '')] = v

        self.net.load_state_dict(new_state_dict)


# source = 'dataset/test/'
source = 'dataset/valid/'
val = os.listdir(source)
solver = TTAFrame(DLinkNet34)
solver.load(path='weights/log01_dink34.th')
tic = time()
target = 'outputs/log01_dink34/'
if not os.path.exists(target):
    os.mkdir(target)
for i, name in enumerate(val):
    if i % 10 == 0:
        print(i / 10, '    ', '%.2f' % (time() - tic))
    mask = solver.test_one_img_from_path(source + name)
    mask[mask > 4.0] = 255
    mask[mask <= 4.0] = 0
    mask = np.concatenate([mask[:, :, None], mask[:, :, None], mask[:, :, None]], axis=2)
    cv2.imwrite(target + name[:-7] + 'mask.png', mask.astype(np.uint8))
