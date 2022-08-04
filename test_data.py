from torch.utils.data import DataLoader
import os
from data import ImageFolder

# configurations
SHAPE = (1024, 1024)
ROOT = 'dataset/train/'
FOLDER_PATH = os.path.join(os.getcwd(), ROOT)
imagelist = filter(lambda x: x.find('sat') != -1, os.listdir(FOLDER_PATH))
trainlist = list(map(lambda x: x[:-8], imagelist))

batchsize = 4

dataset = ImageFolder(trainlist, FOLDER_PATH)
data_loader = DataLoader(dataset=dataset, batch_size=batchsize, shuffle=True)

for i in range(1):
    data_loader_iter = iter(data_loader)
    for img, mask in data_loader_iter:
        print("*"*100)
        print(img.shape)
        print(mask.shape)