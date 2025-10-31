from data_loader.custom_transform.augmentors.augmentor import FeatureMap, Augmentor
from data_loader.custom_transform.augmentors.functional import dropout_feature
import torch


class InstanceFeatureDropout(Augmentor):
    def __init__(self, pf: float):
        super(InstanceFeatureDropout, self).__init__()
        self.pf = pf

    def augment(self, f: FeatureMap) -> FeatureMap:
        x = f.unfold()
        if self.pf == 0.0:
            return FeatureMap(x=x)

        x = dropout_feature(x, self.pf)
        
        return FeatureMap(x=x)
    