from __future__ import annotations
import copy

from typing import Any, List, Optional, Tuple
import pytorch_lightning as pl
from rmp_dl.learning.model import RayModel
from rmp_dl.learning.model_parts.loss_log_module import LossModule
from rmp_dl.learning.model_parts.state_encoding.halton_decoding import Halton2dDecoderFactory
import torch
import torch.nn.functional as F



class RayLightningModule(pl.LightningModule):
    def __init__(self, 
                 model_params: dict,
                 loss: dict, 
                 weight_decay: float,
                 learning_rate: float, 
                 # During sequential training, the first dimensino is sequence length, and the logger
                 # has trouble inferring the batch size, so we explicitly set it. 
                 # So this parameter only matters during training
                 batch_size: Optional[int] = None, 
                 truncated_bptt_steps: int = 0,  # 0 means no tbptt
                 custom_recurrent_optimizer_parameters: Optional[dict] = None,
                 training=False,
                ):
        """Wrapper around the RayModel that adds a loss module and some other functionality
        This model is also used in some parts of the code for inference only, so we add an explicit training flag. 
        If that is disabled, the loss module is not initialized. We specifically do not want to initialize it,
        because it loads other models to compare against, based off of how the parameters are configured, which is not something we
        want during inference. Slightly hacky, and probably the loss module should be moved somewhere else or something, 
        but not sure right now of a clean way of doing this. 
        """
        super().__init__()
        self.model_args = copy.deepcopy(model_params)
        self.truncated_bptt_steps = truncated_bptt_steps

        self.model = RayModel(**model_params)

        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.custom_recurrent_optimizer_parameters = custom_recurrent_optimizer_parameters

        log_callback = self.log if batch_size is None else lambda *args, **kwargs: self.log(*args, **kwargs, batch_size=batch_size)
        self.loss_module = None
        if training:
            self.loss_module = LossModule(**loss, 
                                        using_ray_decoding=self.model.ray_decoding_learned, 
                                        n=self.model.output.get_output_size(), 
                                        log_callback=log_callback)

    def get_model_args_and_state_dict(self):
        return copy.deepcopy(self.model_args), self.model.state_dict()

    @property
    def ray_decoding_learned(self):
        return self.model.ray_decoding_learned

    @staticmethod
    def extract_label_from_observation(observation):
        y = observation["expert_policy"]["geodesic"].float().permute(1, 0, 2) # (sequence_length, batch_size, 3)
        y = F.normalize(y, dim=-1, eps=1e-5)  

        return y

    @staticmethod
    def extract_input_from_observation(observation) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        rays = observation["rays"]["rays"].float().permute(1, 0, 2) # (sequence_length, batch_size, num_rays)
        pos = observation["state"]["pos"].float().permute(1, 0, 2) # (sequence_length, batch_size, 3)
        vel = observation["state"]["vel"].float().permute(1, 0, 2) # (sequence_length, batch_size, 3)]
        goal = observation["info"]["goal"].float().permute(1, 0, 2) # (sequence_length, batch_size, 3)

        robot_radius = None
        if "geodesic_inflation" in observation["info"]:
            robot_radius = observation["info"]["geodesic_inflation"].float().unsqueeze(-1).permute(1, 0, 2)

        rel_pos = goal - pos # Normalize position wrt goal

        return rays, rel_pos, vel, robot_radius

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def get_output_size(self):
        return self.model.get_output_size()

    @staticmethod
    def filter_observation(observation: dict[str, Any]) -> dict:
        """Pytorch uses a collate function to batch multiple observations together. To do this, 
        it recursively calls the collate function on entries in the batch, see documentation here:
        https://pytorch.org/docs/stable/data.html#torch.utils.data.default_collate

        During rollouts, we may gather more observations types than we actually use for training, for e.g. logging
        and debugging. This can also depend on the type of rollout (e.g. different policies), and because we use different
        datasets (and also presaved ones), it is very easy for dictionaries to end up with different entries within a training dataset, 
        also because we combine expert and model rollouts. 
        The default collate function will throw an exception if that happens when trying to batch these dictionaries together. 
        This is not always necessary, as we may only use a subset of the data for training.
        So we define this filter here, which filters out unnecessary observations before it is being passed to the collate function. 

        If there is still an exception after this, it means some strictly required data is missing. 
        Args:
            observations (List[dict]): List of observations
        """

        # We just filter on the first level of the dict!
        keep = set(["rays", "state", "info", "expert_policy"])

        return {k: v for k, v in observation.items() if k in keep}


    def training_step(self, batch, batch_idx):
        # Batch is a dictionary of observations
        # E.g. batch["state"]["pos"] contains the position tensor
        # and has shape (sequence_length, batch_size, 3), where sequence length is only > 1 if we have sequenced input data. 
        # We want to be able to deal with both sequenced data and non-sequenced data, as our validation sets may be non-sequenced.
        
        # The batch also contains sequence length information, which we need to extract
        batch, lengths = batch

        rays, rel_pos, vel, robot_radius, y = batch
        
        y_hat, hiddens = self(rays, rel_pos, vel, robot_radius=robot_radius, hiddens=None)

        if self.loss_module is None:
            raise RuntimeError("RayLightningModule was not initialized with training=True, so it cannot be used for training")
        loss = self.loss_module.get_and_log_train_loss(y_hat, y, lengths=lengths, inputs=(rays, rel_pos, vel))
        
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # In case of sequenced data, the batch also contains sequence length information, which we need to extract
        batch, lengths = batch
        rays, rel_pos, vel, robot_radius, y = batch
        
        y_hat, hiddens = self(rays, rel_pos, vel, robot_radius=robot_radius, hiddens=None)        

        if self.loss_module is None:
            raise RuntimeError("RayLightningModule was not initialized with training=True, so it cannot be used for training")
        self.loss_module.get_and_log_validation_loss(y_hat, y, dataloader_idx=dataloader_idx, lengths=lengths, inputs=(rays, rel_pos, vel))
    

    def configure_optimizers(self):
        # The hasattr check is there because there may be some runs on euler still using old code when we push this, 
        # and with multiprocessing this module may be copied but without that attribute. I don't think it should happen
        # but better safe than sorry. TODO: Remove this in a few days (or not, it doesn't hurt)
        if hasattr(self, "custom_recurrent_optimizer_parameters") and self.custom_recurrent_optimizer_parameters is not None:
            # You can set custom optimizer parameters for the recurrent model components.
            

            # We get the parameters from the recurrent models in the combined model. The nested list comprehension is to flatten the list of parameters
            # We may throw an error below if the model is set up in such a way that there are no recurrent models. I think that is okay, 
            # as we may have made a mistake in the config in that case.
            recurrent_parameters = [param for recurrent_model in self.model.output.combined.model.recurrent_models for param in recurrent_model.parameters()]

            # We filter out the recurrent parameters from the parameters to optimize
            non_recurrent_parameters = [param for param in self.parameters() if not any(param is recurrent_param for recurrent_param in recurrent_parameters)]

            # We create the optimizer with 2 different optimizer groups
            return torch.optim.Adam(
                [
                    {"params": non_recurrent_parameters}, # The non recurrent parameters take the default wd and lr defined below
                    # We overwrite parameters defined in the custom_recurrent_optimizer_parameters dict
                    {**{"params": recurrent_parameters},  **self.custom_recurrent_optimizer_parameters}
                ], 
                weight_decay=self.weight_decay,
                lr = self.learning_rate
            )

        return torch.optim.Adam(self.parameters(), weight_decay=self.weight_decay, lr=self.learning_rate)

 