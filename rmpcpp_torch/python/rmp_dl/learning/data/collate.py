from rmp_dl.learning.data.pipeline.nodes.samplers.sampler_output_base import SamplerOutputBase
from rmp_dl.learning.lightning_module import RayLightningModule
import torch

import numpy as np

def numpy_collate(batch):
    r"""Puts each data field into a numpy array with outer dimension batch size"""

    elem = batch[0]
    if isinstance(elem, np.ndarray):
        return np.stack(batch, 0)
    elif isinstance(elem , dict):
        return {key: numpy_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, np.floating):
        return np.array(batch)

    raise TypeError("Expected np.ndarray or dict, got {}".format(type(elem).__name__))

#################################################################################################
def filter_collate_fn(batch):
    # See comments in RayModel.filter_observation

    if not isinstance(batch[0], dict):
        raise RuntimeError("The datapipeline should return a dict for each observation")

    # Function that extracts the length of the first array it finds in the dictionary
    # All arrays should be of the same length, so the first one is fine
    len_of_arrays_in_dict = SamplerOutputBase.length_of_first_numpy_array_in_dict

    # If we have non-sequenced data, the length of the first dimension of all arrays should be 1
    if all(len_of_arrays_in_dict(item) == 1 for item in batch):
        batch = [RayLightningModule.filter_observation(observation) for observation in batch]
        lengths = None
    else:
        # Sequenced data otherwise

        # Batch is now a list of lists of observations (batch_size, seq_len, DICT)
        # First we filter out the entries inside the dictionaries we don't need 
        batch = [RayLightningModule.filter_observation(d) for d in batch]
        # We sort the sequences based on length
        batch = sorted(batch, key=len_of_arrays_in_dict, reverse=True)
        # We save the length of each sequence
        lengths = np.array([len_of_arrays_in_dict(lst) for lst in batch])

        # We pad all sequences to the same length by padding with zeros
        def pad(d):
            if isinstance(d, dict):
                return {k: pad(v) for k, v in d.items()}
            elif isinstance(d, np.ndarray):
                return np.pad(d, ((0, lengths[0] - len(d)), (0, 0)), 'edge')
            else: 
                raise RuntimeError("Expected dict or np.ndarray, got {}".format(type(d).__name__))
        batch = [pad(d) for d in batch]

    data = torch.utils.data.dataloader.default_collate(batch)

    rays, rel_pos, vel, robot_radius = RayLightningModule.extract_input_from_observation(data)
    y = RayLightningModule.extract_label_from_observation(data)

    return (rays, rel_pos, vel, robot_radius, y), lengths
