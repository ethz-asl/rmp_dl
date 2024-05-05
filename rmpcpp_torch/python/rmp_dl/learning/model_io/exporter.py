import os
from rmp_dl.learning.lightning_module import RayLightningModule
from rmp_dl.learning.model import RayModelDirectionConversionWrapper
from rmp_dl.learning.model_io.model_util import ModelUtil
import torch

import torch.nn as nn


class ModelExporter:
    # Used to export to file, or ONNX
    def __init__(self, model):
        self.model = model

    def export_to_onnx(self, filename):
        filename = os.path.join(self.get_model_dir(), filename)

        wrapped = RayModelDirectionConversionWrapper(self.model)

        input_rays = torch.randn(1, 1, 1024).cuda()
        input_rel_pos = torch.randn(1, 1, 3).cuda()
        input_vel = torch.randn(1, 1, 3).cuda()

        # First we do a call with hidden states set to None to get the hidden state sizes
        _, input_hiddens = wrapped(input_rays, input_rel_pos, input_vel, None)

        input_names = ["rays", "rel_pos", "vel", "input_hidden_h0", "input_hidden_c0"]
        output_names = ["output"] 

        # The 1st axis is sequence length which is dynamic, 2nd axis is batch size which is dynamic
        dynamic_axes = {
            "rays":             {0: "sequence_length", 1: "batch_size"},
            "rel_pos":          {0: "sequence_length", 1: "batch_size"},
            # For the hidden state, there is no sequence length
            "input_hidden_h0":   {0: "batch_size"},
            "input_hidden_c0":   {0: "batch_size"},
        }

        # In case of FFN the hidden state stays None, which gets ignored
        if input_hiddens is not None:
            dynamic_axes["output_hidden_h0"] = {0: "batch_size"}
            dynamic_axes["output_hidden_c0"] = {0: "batch_size"}
            output_names += ["output_hidden_h0", "output_hidden_c0"]
        
        inputs = (input_rays, input_rel_pos, input_vel, input_hiddens) if input_hiddens is not None else (input_rays, input_rel_pos, input_vel)

        torch.onnx.export(wrapped, 
                          inputs, 
                          filename,
                          export_params=True,
                          verbose=True,
                          input_names=input_names,
                          output_names=output_names,
                          training=torch.onnx.TrainingMode.EVAL,
                          dynamic_axes=dynamic_axes
                          )

    def export_to_disk(self, filename):
        filename = os.path.join(self.get_model_dir(), filename)

        # Export model to disk
        torch.save(self.model, filename)

    @staticmethod
    def get_model_dir():
        # Get directory of this file
        dir_path = os.path.dirname(os.path.realpath(__file__))

        # We store in ./models/
        return os.path.join(dir_path, "../../../../lfs/models")

    @staticmethod
    def from_wandb(wandb_id, version):
        model = ModelUtil.load_model(wandb_id, version)
        
        return ModelExporter(model)

def export_ffn():
    model_exporter = ModelExporter.from_wandb("5msibfu3", "latest")
    model_exporter.export_to_disk("ffn.pt")
    model_exporter.export_to_onnx("ffn.onnx")

def export_rnn():
    model_exporter = ModelExporter.from_wandb("g2j8uxxd", "latest")
    model_exporter.export_to_disk("rnn.pt")
    model_exporter.export_to_onnx("rnn.onnx")

if __name__ == "__main__":
    export_ffn()
    export_rnn()
    