
import os
from rmp_dl.learning.model_io.exporter import ModelExporter
import torch


class ModelImporter:
    @staticmethod
    def from_disk(name):
        # Get path of this file
        filename = os.path.join(ModelExporter.get_model_dir(), name)
        return torch.load(filename)
    

    @staticmethod
    def from_disk_onnx(name):
        import onnx
        # Get path of this file
        filename = os.path.join(ModelExporter.get_model_dir(), name)
        onnx_model = onnx.load(filename)
        onnx.checker.check_model(onnx_model)
        return onnx_model


    @staticmethod
    def load_rnn():
        return ModelImporter.from_disk("rnn.pt")

    @staticmethod
    def load_ffn():
        return ModelImporter.from_disk("ffn.pt")

    @staticmethod
    def load_rnn_onnx():
        return ModelImporter.from_disk_onnx("rnn.onnx")


if __name__ == "__main__":
    ModelImporter.load_rnn_onnx()
