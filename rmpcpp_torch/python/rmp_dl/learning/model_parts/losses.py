
import abc
from typing import Optional, final
import torch
import torch.nn as nn

class MaskedLoss(nn.Module, abc.ABC):
    def __init__(self, force_unit_norm=False):
        super().__init__()
        self.force_unit_norm = force_unit_norm
        if self.force_unit_norm:
            self.mse = nn.MSELoss()

    def _add_unit_norm_loss(self, loss, y_pred):
        norm = torch.norm(y_pred, dim=-1)
        return loss + self.mse(norm, torch.ones_like(norm))

    @final
    def forward(self, y_pred, y_true, mask: Optional[torch.Tensor]=None): 
        if mask is None:
            mask = torch.tensor([1.0], device=y_pred.device)
        loss = torch.mean(self._get_non_reduced_loss(y_pred, y_true) * mask.unsqueeze(-1))
        if self.force_unit_norm:
            loss = self._add_unit_norm_loss(loss, y_pred)
        return loss

    @abc.abstractmethod
    def _get_non_reduced_loss(self, y_pred, y_true): ...


class BceWithLogitsLoss(MaskedLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = nn.BCEWithLogitsLoss(reduce=False)

    def _get_non_reduced_loss(self, y_pred, y_true):
        return self.loss(y_pred, y_true)

class CrossEntropyLoss(MaskedLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = nn.CrossEntropyLoss(reduce=False)

    def _get_non_reduced_loss(self, y_pred, y_true):
        # Sadly this does not accept multi class labels in pytorch 1.9
        # Check if there is only a single y set to 1. Throw exception otherwise
        sums = torch.sum(y_true, dim=-1)
        if not (torch.isclose(sums, torch.ones_like(sums))).all():
            raise ValueError("Cross entropy loss expects a single y set to 1")
        y_true = torch.argmax(y_true, dim=-1)
        # We need to reshape the loss, as it expects a 1D tensor
        return self.loss(y_pred.reshape(-1, y_pred.shape[-1]), y_true.reshape(-1)).reshape(y_pred.shape[0], -1, 1)
    
class MSELoss(MaskedLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = nn.MSELoss(reduce=False)

    def _get_non_reduced_loss(self, y_pred, y_true):
        return self.loss(y_pred, y_true)
    
class CosineSimilarityLoss(MaskedLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cossim = nn.CosineSimilarity(dim=-1, eps=1e-5)

    def _get_non_reduced_loss(self, y_pred, y_true):
        return (1 - self.cossim(y_pred, y_true)).unsqueeze(-1)  # We want to keep the last dimension, so we can multiply with the mask
    
class AngularLoss(MaskedLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cossim = nn.CosineSimilarity(dim=-1, eps=1e-5)
        # We clamp the input to arccos, see issue here: https://github.com/pytorch/pytorch/issues/8069
        self.epsilon = 1e-7
    
    def _get_non_reduced_loss(self, y_pred, y_true):
        return torch.arccos(torch.clamp(self.cossim(y_pred, y_true), -1 + self.epsilon, 1 - self.epsilon)).unsqueeze(-1)  # We want to keep the last dimension, so we can multiply with the mask
    
