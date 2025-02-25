import os

from PIL import Image
from torch.utils.data import Dataset
class PACS(Dataset):

    def __init__(self, root: str, domain: str, transform=None, **kwargs):
        """
        Arguments:
            root: The dataset must be located at ```<root>/domainnet```
            domain: One of the 4 domains
            train: Whether or not to use the training set.
            transform: The image transform applied to each sample.
        """
        super().__init__()
        # self.train = train
        self.domain = domain
        self.set_paths_and_labels(root)
        self.transform = transform

    def set_paths_and_labels(self, root):
        # name = "train" if self.train else "test"
        labels_file = os.path.join(root, "PACS", f"{self.domain}.txt")
        img_dir = os.path.join(root, "PACS")

        with open(labels_file) as f:
            content = [line.rstrip().split(" ") for line in f]
        self.img_paths = [os.path.join(img_dir, x[0]) for x in content]
        self.labels = [int(x[1]) for x in content]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        img = Image.open(self.img_paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label