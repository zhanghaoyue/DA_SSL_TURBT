from .augmentor import Augmentor, Compose, RandomChoice
from .identity import Identity

from .instance_mask import InstanceMasking
from .instance_replace import InstanceFeatureReplace
from .instance_rand import InstanceReplace
from .instance_drop import InstanceDrop
from .instance_noise import InstanceFeatureNoise
from .instance_feature_drop import InstanceFeatureDrop
from .instance_feature_dropout import InstanceFeatureDropout

__all__ = [
    'Augmentor',
    'Compose',
    'RandomChoice',
    'Identity',
    'InstanceMasking',
    'InstanceFeatureReplace',
    'InstanceReplace',
    'InstanceDrop',
    'InstanceFeatureNoise',
    'InstanceFeatureDrop',
    'InstanceFeatureDropout'
]

classes = __all__