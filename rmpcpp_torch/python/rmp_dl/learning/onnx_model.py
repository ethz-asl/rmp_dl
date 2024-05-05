import os
import numpy as np
import onnxruntime as rt
from rmp_dl.learning.model_io.exporter import ModelExporter
from rmp_dl.learning.model_io.importer import ModelImporter
import torch


class OnnxModel:
    def __init__(self, onnx_filename):
        self.sess = rt.InferenceSession(onnx_filename, providers=['CUDAExecutionProvider'])

    def print_info(self):
        def print_inf(param):
            print(param.name)
            print(param.shape)
            print(param.type)

        for inp in self.sess.get_inputs():
            print_inf(inp)
        
        for out in self.sess.get_outputs():
            print_inf(out)

    
    def _resolve_hidden_size(self):
        return (np.zeros((1, 1, 256), dtype=np.float32), np.zeros((1, 1, 256), dtype=np.float32))

    def __call__(self, rays, rel_pos, vel, hiddens):
        if hiddens is None:
            hiddens = self._resolve_hidden_size()
        
        if isinstance(rays, torch.Tensor):
            rays = rays.cpu().numpy()
        if isinstance(rel_pos, torch.Tensor):
            rel_pos = rel_pos.cpu().numpy()
        if isinstance(vel, torch.Tensor):
            vel = vel.cpu().numpy()


        result = self.sess.run(None,
        {
            "rays": rays,
            "rel_pos": rel_pos, 
            "input_hidden_h0": hiddens[0], 
            "input_hidden_c0": hiddens[1], 
        })

        return torch.from_numpy(result[0]), (result[1], result[2])

if __name__ == "__main__":
    filename = os.path.join(ModelExporter.get_model_dir(), "rnn.onnx")
    onnx_model = OnnxModel(filename)
    onnx_model.print_info()

    rays = torch.randn(1, 1, 1024)
    rel_pos = torch.randn(1, 1, 3)
    vel = torch.randn(1, 1, 3)
    hiddens = (torch.randn(1, 1, 256), torch.randn(1, 1, 256))

    print(onnx_model(rays, rel_pos, vel, hiddens))