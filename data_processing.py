from torch.utils.data import DataLoader
import torch 
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
import os 
from PIL import Image
import matplotlib.pyplot as plt

torch.manual_seed(43)

train_folder = '\\Users\\oscar\\OneDrive\\Documents\\pytorch\\CNNs\\data\\archive\\train'

test_folder = '\\Users\\oscar\\OneDrive\\Documents\\pytorch\\CNNs\\data\\archive\\test'

class DataIter:
    
    def __init__(self,train_folder: str, test_folder: str, batch_size: int = 4, shuffle: bool = False, show_data: bool = False, img_size:int = 244):
        self.train_folder = train_folder
        self.test_folder = test_folder
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.list_length = len(self.train_folder)
        self.dictionario = dict(zip(range(len(os.listdir(self.train_folder))), os.listdir(self.train_folder)))
        self.show_data = show_data
        self.img_size = img_size
        if self.show_data:
            print(f"The length of the list of training images is: {self.list_length}")
            print(f"The dictionary of training images is: \n {self.dictionario}")
                
    def __getitem__(self,idx: int):
        img_path_train = os.listdir(self.train_folder)[idx]
        img = os.path.join(self.train_folder,img_path_train)
        return img
    
    def get_image(self,id_img: int,img_path: str):
        img_in = os.listdir(img_path)[id_img]
        img = os.path.join(img_path,img_in)
        img = Image.open(img)
        # transform = transforms.ToTensor()
        # ten_img = transform(img)
        # print(ten_img.shape)
        img.show()
        plt.show()
    
    def dataloader(self):
        stats = (0.5,0.5,0.5), (0.5,0.5,0.5)
        transform = transforms.Compose(
            [transforms.Resize(self.img_size,antialias=True),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(stats[0],stats[1])
            ])
        train_img = ImageFolder(self.train_folder, transform=transform)
        test_img =  ImageFolder(self.test_folder, transform=transform)
        train_iter = DataLoader(train_img, batch_size = self.batch_size, shuffle= self.shuffle, num_workers=1, pin_memory=True)
        test_iter = DataLoader(test_img, batch_size = self.batch_size)
        return train_iter,test_iter
    
