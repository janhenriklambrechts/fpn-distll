

import torchvision
class MultiTransformDataset(torchvision.datasets.CIFAR100):
    def __init__(self, *transforms):
        super(torchvision.datasets.CIFAR100, self).__init__()
        self.transforms = transforms

    def __getitem__(self, i):
        img, target = self.data[i], self.target[i] 

        img = Image.fromarray(img)

        imgs_transformed = tuple(transform(img) for transform in self.transforms) 
        return imgs_transformed, target

