from data_loader.custom_transform.augmentors.augmentor import FeatureMap, Augmentor
from data_loader.custom_transform.augmentors.functional import add_random_gaussian_noise

class InstanceFeatureNoise(Augmentor):
    def __init__(self, pf: float):
        super(InstanceFeatureNoise).__init__()
        self.pf = pf

    def augment(self, f: FeatureMap) -> FeatureMap:
        x = f.unfold()
        if self.pf == 0.0:
            return FeatureMap(x=x)
        
        x = add_random_gaussian_noise(x, self.pf)
        
        return FeatureMap(x=x)