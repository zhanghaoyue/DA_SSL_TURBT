from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from typing import Optional, Tuple, NamedTuple, List


class FeatureMap(NamedTuple):
    x: torch.FloatTensor  # [instance, feature-length]

    def unfold(self) -> torch.FloatTensor:
        return self.x


class Augmentor(ABC):
    """Base class for feature map augmentors."""
    def __init__(self):
        pass

    @abstractmethod
    def augment(self, f: FeatureMap) -> FeatureMap:
        raise NotImplementedError("Augmentor.augment should be implemented.")

    def __call__(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.augment(FeatureMap(x)).unfold()


class Compose(Augmentor):
    def __init__(self, augmentors: List[Augmentor]):
        super().__init__()
        self.augmentors = augmentors

    def augment(self, f: FeatureMap) -> FeatureMap:
        for aug in self.augmentors:
            f = aug.augment(f)
        return f


class RandomChoice(Augmentor):
    def __init__(self, augmentors: List[Augmentor], num_choices: int):
        super().__init__()
        assert num_choices <= len(augmentors), "num_choices must be <= number of augmentors"
        self.augmentors = augmentors
        self.num_choices = num_choices

    def augment(self, f: FeatureMap) -> FeatureMap:
        idx = torch.randperm(len(self.augmentors))[:self.num_choices]
        for i in idx:
            f = self.augmentors[i].augment(f)
        return f