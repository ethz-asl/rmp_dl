
from typing import Any, List, Optional, Tuple
from rmp_dl.learning.data.utils.validation_dataset_names import ValidationDatasetNames
from rmp_dl.learning.model_parts.losses import AngularLoss, BceWithLogitsLoss, CosineSimilarityLoss, CrossEntropyLoss, MSELoss
from rmp_dl.learning.model_parts.state_encoding.halton_decoding import Halton2dDecoder
from rmp_dl.learning.model_parts.state_encoding.halton_encoding import Halton2dEncoder
from rmp_dl.learning.model_io.model_util import ModelUtil
import torch
import torch.nn as nn
import pytorch_lightning as pl

class LossModule(pl.LightningModule):
    def __init__(self,
                 cartesian_loss_type: str, 
                 cartesian_loss_parameters: dict, 
                 using_ray_decoding: bool,
                 log_callback, 
                 ray_loss_type: Optional[str]=None, 
                 ray_loss_parameters: Optional[dict]=None,
                 n: Optional[int]=None, 
                 wandb_model_comparison_ids: List[str] = [],
                 ):
        """This module is in charge of computing the loss
        Note that we have 2 main loss types: Cartesian loss and ray loss.
        This is because we can have a decoder that directly outputs a cartesian direction, or a decoder that outputs a distribution over rays.
        The cartesian loss is usually cosine similarity with the geodesic, while the ray loss is usually cross entropy with the geodesic converted to a ray. 
        The ray loss is only used when using_ray_decoding is set to True.
        In case of using ray loss, we also want to log the cartesian loss, so we convert the distribution over rays to a single direction and compute the cartesian loss.
        This is so we can easily compare the performance of a model that directly outputs a cartesian direction with a model that outputs a distribution over rays.

        The wandb_model_comparison_ids is used to compare the loss of different models. This is useful when training using DAgger, as 
        this allows comparisons between models over the exact same data. The loss is logged with the wandb id as a suffix. 

        """
        super().__init__()
        self.using_ray_decoding = using_ray_decoding

        self.cartesian_loss = self._resolve_cartesian_loss(cartesian_loss_type, cartesian_loss_parameters)

        if using_ray_decoding:
            if n is None or ray_loss_type is None or ray_loss_parameters is None:
                raise ValueError("When using ray decoding, the ray decoding loss parameters must be specified")
            self.halton2d_label_encoder = Halton2dEncoder(n, **ray_loss_parameters["label_halton_encoding"])
            self.ray_loss = self._resolve_ray_loss(ray_loss_type, ray_loss_parameters)

            # We allow for multiple ray -> cartesian decoders. These don't affect training, but are only here for logging purposes
            # so we can easily compare against models that directly output a single cartesian direction
            self.ray_output_decoders = {}
            for params in ray_loss_parameters["ray_output_decoding"]:
                name, decoder = self.get_ray_output_decoder(**params, n=n)
                self.ray_output_decoders[name] = decoder

        self.log_callback = log_callback

        self.comparison_models: List[Tuple[str, nn.Module]] = []  
        for wandb_model_comparison_id in wandb_model_comparison_ids:
            model = ModelUtil.load_model(wandb_model_comparison_id)
            self.comparison_models.append((wandb_model_comparison_id, model))

    def get_ray_output_decoder(self, name, method, method_parameters, n):
        return name, Halton2dDecoder(n=n, method=method, method_parameters=method_parameters)

    def compute_mask(self, lengths):
        """Compute a mask to disable gradients for padded sequences

        Args:
            lengths (List[int]): The lengths of the sequences

        Returns:
            torch.Tensor: A mask of shape (max_length) where 1 means that the value should be used and 0 means that it should be ignored
        """
        if lengths is None:
            return torch.tensor([1.0], device=torch.device("cuda"))
        
        batch_size = len(lengths)
        max_length = max(lengths)
        # [[0, 0, 0, ...], [1, 1, 1, ...], ...] 
        # Basically a column vector of (0 .. max_length - 1) repeated batch_size times
        length_count_tensor = torch.arange(max_length).expand((batch_size, max_length)).T.to(torch.device("cuda"))
        lengths_tensor = torch.tensor(lengths).unsqueeze(0).to(torch.device("cuda"))
        mask = (length_count_tensor < lengths_tensor)
        return mask

    def get_and_log_train_loss(self, *args, **kwargs):
        if self.using_ray_decoding: 
            return self._get_and_log_ray_loss_train(*args, **kwargs)
        else:
            return self._get_and_log_cartesian_loss_train(*args, **kwargs)
    
    def get_and_log_validation_loss(self, *args, **kwargs):
        if self.using_ray_decoding: 
            return self._get_and_log_ray_loss_validation(*args, **kwargs)
        else:
            return self._get_and_log_cartesian_loss_validation(*args, **kwargs)

    def _get_and_log_cartesian_loss_train(self, y_hat, y, lengths=None, inputs: Optional[Tuple] = None):
        """Get and log the cartesian loss during training

        Args:
            lengths (_type_, optional): In case of sequenced data, lengths is used to determine the mask, to disable gradients for padded sequences. 
                In case of None, no mask is used. Defaults to None.
        """
        mask = self.compute_mask(lengths)
        norm = torch.mean(torch.norm(y_hat, dim=-1))
        
        loss = self.cartesian_loss(y_hat, y, mask)

        self.log_callback('y_norm', norm, on_step=True, on_epoch=True)
        self.log_callback('train_loss', loss, on_step=True, on_epoch=True)
        
        with torch.no_grad():
            # Compare against other models 
            for wandb_id, model in self.comparison_models:
                if inputs is None:
                    # I can also make inputs non-default, however I don't want to mess with the signature as there are current 
                    # runs in the queue on euler expecting the old signature, so I think they may crash if I change it.  
                    raise ValueError("When comparing against other models, data inputs have to be passed to the loss log module")
                y_comp, hiddens = model(*inputs)
                comp_loss = self.ray_loss(y_comp, y, mask)
                self.log_callback(f"train_loss-{wandb_id}", comp_loss, on_epoch=True)

        return loss

    def _get_and_log_cartesian_loss_validation(self, y_hat, y, dataloader_idx, lengths=None, inputs: Optional[Tuple]=None):
        """Los the cartesian loss during validation

        Args:
            lengths (_type_, optional): In case of sequenced data, lengths is used to determine the mask, to disable gradients for padded sequences. 
                In case of None, no mask is used. Defaults to None.
        """
        mask = self.compute_mask(lengths)
        loss = self.cartesian_loss(y_hat, y, mask)

        name = ValidationDatasetNames.resolve_validation_dataset_name(dataloader_idx)
        self.log_callback(name, loss, on_epoch=True)
        
        
        for wandb_id, model in self.comparison_models:
            if inputs is None:
                # I can also make inputs non-default, however I don't want to mess with the signature as there are current 
                # runs in the queue on euler expecting the old signature, so I think they may crash if I change it.  
                raise ValueError("When comparing against other models, data inputs have to be passed to the loss log module")
            y_comp, hiddens = model(*inputs)
            comp_loss = self.cartesian_loss(y_comp, y, mask)
            self.log_callback(f"{name}-{wandb_id}", comp_loss, on_epoch=True)

    def _get_and_log_ray_loss_train(self, y_hat, y, lengths=None, inputs: Optional[Tuple] = None):
        """Get and log the ray loss during train. Also logs cartesian loss

        Args:
            lengths (_type_, optional): In case of sequenced data, lengths is used to determine the mask, to disable gradients for padded sequences. 
                In case of None, no mask is used. Defaults to None.
        """
        mask = self.compute_mask(lengths)
        # First we need to convert y to ray representation
        # The label encoder does not expect the (sequence_length, .. ) dimension, so we reshape it to include it in batch size
        # and then reshape back
        y_ray = self.halton2d_label_encoder(y.reshape(-1, *y.shape[2:])).reshape(*y.shape[:2], -1)
        ray_loss = self.ray_loss(y_hat, y_ray, mask)

        self.log_callback('train_ray_loss', ray_loss, on_step=True, on_epoch=True)

        with torch.no_grad():
            # Compare against other models 
            for wandb_id, model in self.comparison_models:
                if inputs is None:
                    # I can also make inputs non-default, however I don't want to mess with the signature as there are current 
                    # runs in the queue on euler expecting the old signature, so I think they may crash if I change it.  
                    raise ValueError("When comparing against other models, data inputs have to be passed to the loss log module")
                y_comp, hiddens = model(*inputs)
                comp_loss = self.ray_loss(y_comp, y_ray, mask)
                self.log_callback(f"train-ray_loss-{wandb_id}", comp_loss, on_epoch=True)


        with torch.no_grad():
            # We also convert the distribution over the ray directions to a single direction and compute the cartesian loss
            # (which is usually cosine similarity with the geodesic), so we can more easily compare with the cartesian decoders
            # and get a better idea of what a planning run using the ray decoder would look like. 
            # We allow the option for multiple ray_distribution -> cartesian_direction decoders and log all the losses
            for decoder_name, decoder in self.ray_output_decoders.items():
                # The decoder does not expect the (sequence_length, .. ) dimension, so we reshape it to include it in batch size
                # and then reshape back
                direction_hat = decoder(y_hat.reshape(-1, *y_hat.shape[2:])).reshape(*y_hat.shape[:2], -1)
                cartesian_loss = self.cartesian_loss(direction_hat, y, mask)
        
                self.log_callback(f'train_loss-{decoder_name}', cartesian_loss, on_step=False, on_epoch=True)

        return ray_loss
        
    def _get_and_log_ray_loss_validation(self, y_hat, y, dataloader_idx, lengths=None, inputs: Optional[Tuple]=None):
        """Log the ray loss during train. Also logs cartesian loss

        Args:
            lengths (List[int], optional): In case of sequenced data, lengths is used to determine the mask, to disable gradients for padded sequences. 
                In case of None, no mask is used. Defaults to None.
        """
        mask = self.compute_mask(lengths)
        # First we need to convert y to ray representation
        # The label encoder does not expect the (sequence_length, .. ) dimension, so we reshape it to include it in batch size
        # and then reshape back
        y_ray = self.halton2d_label_encoder(y.reshape(-1, *y.shape[2:])).reshape(*y.shape[:2], -1)
        ray_loss = self.ray_loss(y_hat, y_ray, mask)

        name = ValidationDatasetNames.resolve_validation_dataset_name(dataloader_idx)

        self.log_callback(f"{name}-ray_loss", ray_loss, on_epoch=True)

        # Compare against other models 
        for wandb_id, model in self.comparison_models:
            if inputs is None:
                # I can also make inputs non-default, however I don't want to mess with the signature as there are current 
                # runs in the queue on euler expecting the old signature, so I think they may crash if I change it.  
                raise ValueError("When comparing against other models, data inputs have to be passed to the loss log module")
            y_comp, hiddens = model(*inputs)
            comp_loss = self.ray_loss(y_comp, y_ray, mask)
            self.log_callback(f"{name}-ray_loss-{wandb_id}", comp_loss, on_epoch=True)

        with torch.no_grad(): # Actually we are in validation mode so this probably does not do much
            # We also convert the distribution over the ray directions to a single direction and compute the cartesian loss
            # (which is usually cosine similarity with the geodesic), so we can more easily compare with the cartesian decoders
            # and get a better idea of what a planning run using the ray decoder would look like. 
            # We allow the option for multiple ray_distribution -> cartesian_direction decoders and log all the losses
            for decoder_name, decoder in self.ray_output_decoders.items():
                # The decoder does not expect the (sequence_length, .. ) dimension, so we reshape it to include it in batch size
                # and then reshape back
                direction_hat = decoder(y_hat.reshape(-1, *y_hat.shape[2:])).reshape(*y_hat.shape[:2], -1)
                cartesian_loss = self.cartesian_loss(direction_hat, y, mask)
        
                self.log_callback(f"{name}-{decoder_name}", cartesian_loss, on_epoch=True)

    def _resolve_ray_loss(self, ray_loss_type, ray_loss_parameters):
        if ray_loss_type == "bce_with_logits":
            return BceWithLogitsLoss()
        if ray_loss_type == "cross_entropy_new":
            return CrossEntropyLoss()
        if ray_loss_type == "mse":
            return MSELoss()
        else: 
            raise ValueError(f"Unsupported ray_loss_type string {ray_loss_type}")

    def _resolve_cartesian_loss(self, cartesian_loss_type, cartesian_loss_parameters):
        if cartesian_loss_type == "cosine_similarity":
            return CosineSimilarityLoss(force_unit_norm=cartesian_loss_parameters["force_unit_norm"])
        elif cartesian_loss_type == "mse":
            return MSELoss(force_unit_norm=cartesian_loss_parameters["force_unit_norm"])
        elif cartesian_loss_type == "angular":
            return AngularLoss(force_unit_norm=cartesian_loss_parameters["force_unit_norm"])
        else:
            raise ValueError(f"Unsupported cartesian_loss_type string {cartesian_loss_type}")        