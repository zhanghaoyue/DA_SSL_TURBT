from data_loader.custom_transform.augmentors.augmentor import FeatureMap, Augmentor


class Identity(Augmentor):
    def __init__(self):
        super(Identity, self).__init__()

    def augment(self, f: FeatureMap) -> FeatureMap:
        return f