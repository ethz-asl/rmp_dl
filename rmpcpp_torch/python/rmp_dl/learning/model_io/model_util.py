
import copy
from typing import OrderedDict
from rmp_dl.learning.util import WandbUtil
import torch

class ModelUtil:
    @staticmethod
    def load_model(wandb_id, version="latest", download_dir="data/temp/models"):
        file = WandbUtil.download_model(wandb_id, version, download_dir)
        return ModelUtil.get_model_from_checkpoint_file(file, wandb_id)


    @staticmethod
    def get_model_from_checkpoint_file(file, wandb_id):
        checkpoint = torch.load(file)

        config = WandbUtil.get_config(wandb_id)
        old_config_version = config["model"]["version"]

        ModelUtil.fix_config(config)

        from rmp_dl.learning.lightning_module import RayLightningModule  # To avoid circular import issues we import it here
        model = RayLightningModule(**config["model"]["module_parameters"])

        sd = checkpoint["state_dict"]
        fixed_sd = ModelUtil.fix_state_dict(sd, old_config_version)
        model.load_state_dict(fixed_sd)
        model.to(torch.device("cuda"))
        
        model.eval()

        print(model)

        return model

    @staticmethod   
    def fix_state_dict(sd, old_config_version) -> OrderedDict[str, torch.Tensor]:
        """Fixes old state dicts that have been saved with an outdated version of the code.
        Note 5th July 2023: Some really old versions are not backwards compatible anymore due to significant changes in the code. 
        Though most of them contained some bugs still anyway, and I'll remove them from wandb as well in due time. """
        
        new_sd = {}

        # Because we moved the raynetwork and statenetwork into separate modules, we need to fix the state dict
        for key, value in sd.items():
            if  key.startswith("raynetwork.model") or key.startswith("statenetwork.model") or \
                key.startswith("raynetwork.encoder.model") or key.startswith("statenetwork.state_encoder.model"):  # This is the newest version, nothing to do
                new_key = key
            elif key.startswith("raynetwork.") or key.startswith("statenetwork."):
                new_key = key.replace("raynetwork.", "raynetwork.model.")
                new_key = new_key.replace("statenetwork.", "statenetwork.model.")
            else:
                new_key = key
            new_sd[new_key] = value

        sd = new_sd
        new_sd = {}

        # 13th of July 2023: we moved the raynetwork into a separate "encoder" module
        # i.e. we change raynetwork -> raynetwork.encoder
        for key, value in sd.items():
            if key.startswith("raynetwork.encoder."): # This is the new version, nothing to do
                new_key = key
            elif key.startswith("raynetwork."): # This is the old version
                new_key = key.replace("raynetwork.", "raynetwork.encoder.")
            else:
                new_key = key
            new_sd[new_key] = value


        sd = new_sd
        new_sd = {}
        
        # I saved the shared rayencoder as a module inside the main module but removed it later. 

        # We moved the raynetwork into a shared encoder module on the outer level
        # We actually need to copy the state dict items now. Even though the models share the weights, 
        # pytorch still wants the keys. Pytorch does not know that they are shared, and just loads the weights. 
        # I guess it means that it is writing to the same memory space twice, but I don't think that is a problem.
        for key, value in sd.items():
            if key.startswith("shared_ray_encoder"):
                pass  # We don't add it
            else:
                new_sd[key] = value

        sd = new_sd
        new_sd = {}

        # We moved the combined network into a separate module, and also put the last layer into a separate module called the decoder
        # Type of error that we are fixing:
        # RuntimeError: Error(s) in loading state_dict for RayModel:
        # Missing key(s) in state_dict: "output.combined.0.weight", "output.combined.0.bias", "output.combined.2.weight", "output.combined.2.bias", "output.combined.4.weight", "output.combined.4.bias", "output.combined.6.weight", "output.combined.6.bias", "output.decoder.connection.weight", "output.decoder.connection.bias". 
        # Unexpected key(s) in state_dict: "combined.8.bias", "combined.0.weight", "combined.0.bias", "combined.2.weight", "combined.2.bias", "combined.4.weight", "combined.4.bias", "combined.6.weight", "combined.6.bias", "combined.8.weight". 
        # So combined.n.* -> output.combined.n.* 
        # And for the highest n: 
        # combined.n -> output.decoder.connection
        highest_n = 0
        for key, value in sd.items():
            if key.startswith("output.combined."): # New version, nothing to do
                new_key = key
            elif key.startswith("combined."):  # Old version, keep track of n and already change it
                n = int(key.split(".")[1])  
                highest_n = max(highest_n, n)
                new_key = key.replace("combined.", "output.combined.")
            else:
                new_key = key
            new_sd[new_key] = value

        
        # We still have to change the highest n to the decoder
        if highest_n != 0:
            sd = new_sd
            new_sd = {}
            # This means that we had an old version that we updated
            for key, value in sd.items():
                if key.startswith(f"output.combined.{highest_n}."):
                    new_key = key.replace(f"output.combined.{highest_n}.", "output.decoder.connection.")
                else:
                    new_key = key
                new_sd[new_key] = value


        # 2023 - 08 - 24, moved all modules into a separate model module, such that the outer lightningmodule is just in charge of training and loss. 
        if old_config_version < 8:
            sd = new_sd
            new_sd = {}
            for key, value in sd.items():
                if key.startswith("_metadata"):
                    new_key = key
                else:
                    new_key = f"model.{key}"
                new_sd[new_key] = value

        # 2023 - 09 - 05, moved the combined network into a separate module, as we are going to be adding more complexity (e.g. lstm) here. 
        if old_config_version < 10:
            sd = new_sd
            new_sd = {}
            for key, value in sd.items():
                if key.startswith("model.output.combined"):
                    new_key = key.replace("model.output.combined", "model.output.combined.model")
                else:
                    new_key = key
                new_sd[new_key] = value

        # 2023 - 09 -21. Removed the duplication of hidden_state_module and model in the combined network. 
        # We just remove the state dict, as the module was assigned to 2 attributes, meaning that it would be saved twice. 
        # We just remove the keys for the duplicate attribute that was removed. 
        if old_config_version < 11:
            sd = new_sd
            new_sd = {}
            for key, value in sd.items():
                if key.startswith("model.output.combined.hidden_state_module"):
                    pass
                else:
                    new_key = key
                new_sd[new_key] = value

        return new_sd

    @staticmethod
    def fix_config(config):
        """Fixes old config files that have been saved with an outdated version of the code.
        Note 5th July 2023: Some really old versions are not backwards compatible anymore due to significant changes in the code. 
        Though most of them contained some bugs still anyway, and I'll remove them from wandb as well in due time. """

        # 04-08-2023: Switched the data pipeline, and moved configs into 4 different files; 
        # general, model, dataset_train, dataset_validation. We check whether "general" exists to determine whether it is a new or old
        # config. 

        if not "general" in config:
            ModelUtil._fix_old_config(config)
            old_config = copy.deepcopy(config)
            config.clear()
            config.update({
                "general": {
                    "planner": old_config["planner"], 
                    "policies": old_config["policies"],
                    "worldgen": old_config["worldgen"], 
                    "voxel_size": old_config["worldgen"]["voxel_size"],
                    "truncation_distance_vox": old_config["truncation_distance_vox"]
                },
                "model": {"model": old_config["model"]}
                # The datatrain and dataval are only used during training so it doesnt matter that they are not defiend
            })
        else:
            # Forgot to put version in for a while
            if "version" not in config["model"]:
                config["model"]["version"] = 4

        ModelUtil.fix_new_config(config)



    @staticmethod
    def fix_new_config(config):
        if config["model"]["version"] == 4:
            config["model"]["version"] = 5
            config["model"]["model"]["learning_rate"] = 0.001

        # I think I skipped a version by accident
        # 2023 - 08 - 24 I moved all modules into a separate model module, such that the outer lightningmodule is just in charge of training and loss.
        if config["model"]["version"] == 5 or config["model"]["version"] == 6:
            config["model"]["version"] = 7
            config["model"]["module_parameters"] = {
                "weight_decay": config["model"]["model"]["weight_decay"],
                "learning_rate": config["model"]["model"]["learning_rate"],
                "loss": config["model"]["model"]["loss"],
                "model_params": {
                    "disable_rays": config["model"]["model"]["disable_rays"],
                    "batch_norm": config["model"]["model"]["batch_norm"],
                    "shared_ray_encoder": config["model"]["model"]["shared_ray_encoder"],
                    "shared_ray_decoder": config["model"]["model"]["shared_ray_decoder"],
                    "raynetwork": config["model"]["model"]["raynetwork"],
                    "statenetwork": config["model"]["model"]["statenetwork"],
                    "outputnetwork": config["model"]["model"]["outputnetwork"],
                    "maximum_ray_length": 10.0, # This has moved from getting it from a different config file. Up to this point it was always 10.0, so we can hardcode it

                }
            }
            del config["model"] ["model"] 

        # Changed the ray output decoding method to support multiple decoders which are logged at the same time
        # This change pretty much only affects logging during training. 
        # TODO: I should probably only load the inner model during inference instead of this outer lightning module,
        # so that we don't have to deal with these parameters. I'm already close to being able to easily do this, 
        # there's just some methods from the outer lightning module that I need to move into the model module, 
        # but I think these methods belong moreso in the outer model, so not too sure if I should do it this way, 
        # and probably requires a different solution. 
        if config["model"]["version"] == 7:
            config["model"]["version"] = 8

            config["model"]["module_parameters"]["loss"]["ray_loss_parameters"]["ray_output_decoding"] = [
                {
                    "name": "max1",
                    "method": "max",
                    "method_parameters": {
                        "k": 1
                    }
                }
            ]

        if config["model"]["version"] == 8:
            config["model"]["version"] = 9

            # Fixed the naming of losses
            loss = config["model"]["module_parameters"]["loss"] 
            if loss["ray_loss_type"] == "cross_entropy":
                loss["ray_loss_type"] = "bce_with_logits"

        # Nothing to do from 9 -> 10 -> 11, as this only affects the state dictionary of the model 
        if config["model"]["version"] == 9 or config["model"]["version"] == 10:
            config["model"]["version"] = 11

        # Switched from a boolean batch norm, to a string norm_type option, as we add support for layer norm
        if config["model"]["version"] == 11:
            config["model"]["version"] = 12

            # Added the option to disable the velocity in the state network
            config["model"]["module_parameters"]["model_params"]["norm_type"] = "batch_norm" if config["model"]["module_parameters"]["model_params"]["batch_norm"] else "none"
            del config["model"]["module_parameters"]["model_params"]["batch_norm"]

        # Added dropout possibility
        if config["model"]["version"] == 12:
            config["model"]["version"] = 13

            # There was no dropout before, which corresponds to 0.0
            config["model"]["module_parameters"]["model_params"]["dropout"] = 0.0



    @staticmethod 
    def _fix_old_config(config):
        # Old versions used to have parameters for the model grouped together, this is now split into different modules
        if type(config["model"]["raynetwork"]) is list: # This means that it is an old version
            config["model"]["raynetwork"] = {
                "model_type": "fully_connected",  # This is the default
                "fully_connected_layer_sizes": config["model"]["raynetwork"],
                "invert_rays": config["model"]["invert_rays"],
            }

            config["model"]["statenetwork"] = {
                "model_type": "fully_connected",  # This is the default
                "fully_connected_layer_sizes": config["model"]["statenetwork"],
                "disable_velocity": config["model"]["disable_velocity"],
                "normalize_relpos_and_vel": config["model"]["normalize_relpos_and_vel"],
                "use_positional_encoding": False,  # This is new and the default
                "positional_encoding_L": 4,  # This is irrelevant if disabled 
            }

            del config["model"]["invert_rays"]
            del config["model"]["disable_velocity"]
            del config["model"]["normalize_relpos_and_vel"]


        # Add weight decay default if not there
        if "weight_decay" not in config["model"]:
            config["model"]["weight_decay"] = 0

        # The raynetwork has been moved to a separate "encoder" module
        if "encoder" not in config["model"]["raynetwork"] and "shared_ray_encoder" not in config["model"]:
            config["model"]["raynetwork"] = {
                "invert_rays": config["model"]["raynetwork"]["invert_rays"],
                "encoder": {
                    "model_type": config["model"]["raynetwork"]["model_type"],
                    "model_type_parameters": {
                        "fully_connected": {
                            "layer_sizes": copy.deepcopy(config["model"]["raynetwork"]["fully_connected_layer_sizes"])
                        },
                    }
                }
            }

        # We moved grouped the model_type parameters of the statenetwork into a separate dictionary
        # So statenetwork.fully_connected_layer_sizes -> statenetwork.model_type_parameters.fully_connected.layer_sizes
        if "model_type_parameters" not in config["model"]["statenetwork"]:
            config["model"]["statenetwork"]["model_type_parameters"] = {
                "fully_connected": {
                    "layer_sizes": copy.deepcopy(config["model"]["statenetwork"]["fully_connected_layer_sizes"])
                }
            }
            del config["model"]["statenetwork"]["fully_connected_layer_sizes"]

        
        # We made the input encoding of the state selectable with a string (as we added ray blowup encoding aside from none and nerf-positional)
        if "state_encoding" not in config["model"]["statenetwork"]:
            config["model"]["statenetwork"]["state_encoding"] = {
                "disable_velocity": config["model"]["statenetwork"]["disable_velocity"],
                "encoding_type": "none" if not config["model"]["statenetwork"]["use_positional_encoding"] else "nerf_positional",
                "encoding_type_parameters": {
                    "none": {
                        
                    },
                    "nerf_positional": {
                        "L": config["model"]["statenetwork"]["positional_encoding_L"]
                    },
                    "ray_encoding": {
                        "halton_encoding_method": "max",
                        "disable_grad": False
                    },
                }
            }
            del config["model"]["statenetwork"]["disable_velocity"]
            del config["model"]["statenetwork"]["positional_encoding_L"]
            del config["model"]["statenetwork"]["use_positional_encoding"]
            
        # We made the ray encoder a shared network, so the parameters have been moved to the outermost level
        if "shared_ray_encoder" not in config["model"]: # Check if it is in the outermost level
            config["model"]["shared_ray_encoder"] = {
                "encoder_initialization": "random",
                "encoder_initialization_params": {
                    "random": copy.deepcopy(config["model"]["raynetwork"]["encoder"])
                },
            }
            config["model"]["raynetwork"]["disable_grad"] = False
            del config["model"]["raynetwork"]["encoder"]


        
        # The 'combined' network has been moved to a separate module, containing the intermediate network and an output decoder
        # Also, we have a shared_ray_decoder now as well
        # From this point, I've started using versions to keep track of changes instead of trying to check if parameters exist or not
        # Should have thought of this earlier (duh)
        if "version" not in config:
            config["version"] = 1

            config["model"]["shared_ray_decoder"] = {}  # The exact parameters don't matter, as this is new, so old versions can't use it

            config["model"]["outputnetwork"] = {
                "combined_network": {
                    "model_type": "fully_connected",
                    "model_type_parameters": {
                        "fully_connected": {
                            "layer_sizes": copy.deepcopy(config["model"]["combined"])
                        }
                    }
                },
                "decoder": {
                    "decoding_type": "cartesian",  # Cartesian was the default
                    "decoding_type_parameters": {
                        "cartesian": {
                            "normalize_output": config["model"]["normalize_output"]
                        }
                        # The parameters for ray_decoding_learned are not relevant, as this is new
                    }
                }
            }

            del config["model"]["combined"]
            del config["model"]["normalize_output"]


        # 2023 - 07 -17 We put the loss in a separate module, and added a new loss type for the ray decoder
        if config["version"] == 1:
            config["version"] = 2

            config["model"]["loss"] = {
                "cartesian_loss_type": config["model"]["loss"],
                "cartesian_loss_parameters": {
                    "force_unit_norm": config["model"]["force_unit_norm"]
                    },
                # The ray loss stuff is not important because old versions can't have it yet
            }
        
            del config["model"]["force_unit_norm"]

        # 2023 - 07 - 20 We add new velocity and position normalization options, and also separate the normalization method of the two. 
        # Furthermore, we add the option to normalize the rays according to the distance to the goal
        if config["version"] == 2:
            config["version"] = 3

            config["model"]["statenetwork"]["rel_pos_normalization_method"] = "sigmoid_like" if config["model"]["statenetwork"]["normalize_relpos_and_vel"] else "none"
            config["model"]["statenetwork"]["vel_normalization_method"] = "sigmoid_like" if config["model"]["statenetwork"]["normalize_relpos_and_vel"] else "none"

            del config["model"]["statenetwork"]["normalize_relpos_and_vel"]

            config["model"]["raynetwork"]["ray_normalization_method"] = "invert" if config["model"]["raynetwork"]["invert_rays"] else "max"

            del config["model"]["raynetwork"]["invert_rays"]

        if config["version"] == 3:
            config["version"] = 4
    
            config["model"]["loss"]["ray_loss_parameters"] = {
                "label_halton_encoding": {
                    "method": "max",
                    "method_parameters": {
                        "max": {
                            "k": 1
                        }
                    }
                },
                "label_halton_decoding": {
                    "method": "max",
                    "method_parameters": {
                        "max": {}
                    }
                }
            }

            config["model"]["statenetwork"]["state_encoding"]["encoding_type_parameters"]["ray_encoding_learned"]["halton_encoding"] = {
                "method": "max",
                "method_parameters": {
                    "max": {
                        "k": 1
                    }
                }
            }

            del config["model"]["statenetwork"]["state_encoding"]["encoding_type_parameters"]["ray_encoding_learned"]["halton_encoding_method"]

