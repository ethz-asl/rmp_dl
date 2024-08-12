

from bs4 import BeautifulSoup
import graphviz
from rmp_dl.learning.data.pipeline.data_pipeline import DataPipeline
from rmp_dl.learning.data.pipeline.graphviz.node_formatter import NodeFormatter

import rmp_dl.util.io as rmp_io



class DataPipelineGraphviz:
    def __init__(self, pipeline: DataPipeline):
        self.pipeline = pipeline

        self.dot = graphviz.Digraph("Pipeline", comment='Data Pipeline')
        
        self.dot.attr('node', shape='box', style='filled', fillcolor='white')
        self.dot.attr('node', labelloc='l') # left-aligned text

        # First pass, set up the nodes
        for name, node in self.pipeline.blocks.items():
            self.dot.node(name, label=NodeFormatter.format(node), fillcolor=NodeFormatter.get_color(node))

        # Second pass, set up the edges
        for name, node in self.pipeline.blocks.items():
            for input_node in node._get_inputs():
                self.dot.edge(input_node.name, name)

        self.dot = self.dot

    def render(self, filename: str, directory=".", view: bool=False):
        self.dot.render(directory=directory, filename=filename, view=view)

    def get_svg_string(self) -> str:
        return self.dot.pipe(format="svg").decode("utf-8")
    
    def get_html(self) -> str:
        svg_string = self.get_svg_string()
        bs = BeautifulSoup(svg_string, features="xml")
        return str(bs.prettify())

