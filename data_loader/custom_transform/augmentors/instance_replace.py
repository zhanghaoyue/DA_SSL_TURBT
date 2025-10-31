from data_loader.custom_transform.augmentors.augmentor import FeatureMap, Augmentor
from data_loader.custom_transform.augmentors.functional import replace_feature
import torch


class InstanceFeatureReplace(Augmentor):
    def __init__(self, pf: float):
        super().__init__()
        self.pf = pf

    def augment(self, f: FeatureMap) -> FeatureMap:
        x = f.unfold()
        if self.pf == 0.0:
            return FeatureMap(x=x)
        
        x = replace_feature(x, self.pf)
        
        return FeatureMap(x=x)