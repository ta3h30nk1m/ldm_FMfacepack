import os
from abc import abstractmethod
from PIL import Image
from torch.utils.data import Dataset, IterableDataset, DataLoader
from torchvision import transforms
from pytorch_lightning import LightningDataModule

# fix torch random seed
#torch.manual_seed(0)

# class ImgIterableBaseDataset(IterableDataset):
#     '''
#     Define an interface to make the IterableDatasets for text2img data chainable
#     '''
#     def __init__(self, num_records=0, valid_ids=None, size=256):
#         super().__init__()
#         self.num_records = num_records
#         self.valid_ids = valid_ids
#         self.sample_ids = valid_ids
#         self.size = size

#         print(f'{self.__class__.__name__} dataset contains {self.__len__()} examples.')

#     def __len__(self):
#         return self.num_records

#     @abstractmethod
#     def __iter__(self):
#         pass

class FaceDataset(Dataset):
    def __init__(self, data_dir="/content/DF11", train=True):
        # get a list of images
        filenames = os.listdir(data_dir)

        # get the full path to images
        self.full_filenames = []
        for f in filenames:
            if f.endswith('.png'):
                self.full_filenames.append(os.path.join(data_dir, f))
        #self.full_filenames = [os.path.join(data_dir, f) for f in filenames]

        if train:
            self.transform = transforms.Compose([
                transforms.Resize((256,256)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), # Scales data into [0,1] 
                transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((256,256)),
                transforms.ToTensor(), # Scales data into [0,1] 
                transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
            ])

    def __len__(self):
        # return size of dataset
        return len(self.full_filenames)

    def __getitem__(self, idx):
        # open image, apply transforms and return with label
        image = Image.open(self.full_filenames[idx]).convert("RGB")
        new_img = Image.new(image.mode, (310, 310), (255,255,255))
        new_img.paste(image, (25, 0))
        new_img = self.transform(new_img)
        return new_img

class FaceDatasetTrain(FaceDataset):
    def __init__(self, **kwargs):
        super().__init__(data_dir="/content/DF11", train=True)

class FaceDatasetTest(FaceDataset):
    def __init__(self, **kwargs):
        super().__init__(data_dir="", train=False)


class LitDataModule(LightningDataModule):
    def __init__(self, dataset, batch_size):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)