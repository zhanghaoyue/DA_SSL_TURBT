from data_loader.custom_transform.augmentors.augmentor import FeatureMap, Augmentor
from data_loader.custom_transform.augmentors.functional import drop_instance
import torch


class InstanceDrop(Augmentor):
    def __init__(self, pf: float):
        super(InstanceDrop, self).__init__()
        self.pf = pf

    def augment(self, f: FeatureMap) -> FeatureMap:
        x = f.unfold()
        if self.pf == 0.0:
            return FeatureMap(x=x)

        x = drop_instance(x, self.pf)
        
        return FeatureMap(x=x)
    