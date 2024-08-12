
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import logging
from operator import itemgetter
import os
import tarfile
from typing import Any, List
from rmp_dl.learning.data.pipeline.multiprocess_util import MpUtil
from rmp_dl.learning.data.pipeline.pipeline_object_base import PipelineObjectBase
import torch

from typing import Protocol, Sized
#TODO: Fix this with multiprocessing
logger = logging.getLogger("rmpcpp_torch")

LRU_SIZE = 3

# For type hinting the data parameter in initialize_with_data 
class GetItemLenProtocol(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, index) -> object: ... 

class ChunkDataLoader:
    def __init__(self, path: str):
        self.path = path

        self.load_metadata_and_set_length()

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.worker_id = 0
            self.num_workers = 1
        else:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers

    def load_metadata_and_set_length(self):
        self.metadata = torch.load(f'{self.path}/metadata.pt')
        self.chunk_size = self.metadata['chunk_size']
        self.total_length = self.metadata['total_length']
        # Total length is the length across the whole dataset. 
        # Note that self.__len___() returns the length per process


    @staticmethod
    def is_cache(path: str, param_chunk_size) -> bool:
        if not os.path.isfile(path):
            return False
        
        try:
            with tarfile.open(path) as f:
                # f.getnames()[0] gets the first directory name inside the tarball
                # There should only be 1 nested directory, inside of which is the metadata file 
                names = f.getnames()
                dir_name = names[0]
                metadata_file = f.extractfile(dir_name + "/metadata.pt")
                metadata = torch.load(metadata_file)
                metadata_file.close()
                chunk_size, total_length = itemgetter('chunk_size', 'total_length')(metadata)
                if chunk_size != param_chunk_size:
                    logger.debug("Chunk size does not match when loading cache")
                    return False
                
                for i in range(0, total_length, chunk_size):
                    filename = dir_name + f"/chunk_{i // chunk_size}.pt"
                    if not filename in names:
                        logger.debug(f'Chunk {i // chunk_size} not found in {path}')
                        return False
        except:
            return False


        return True


    @staticmethod 
    def initialize_with_data(data: GetItemLenProtocol, path: str, chunk_size: int=1024) -> None:
        # We use an atomic counter to increment the file names
        MpUtil.reset_atomic_counter_with_barrier()
        ChunkDataLoader.write_data(data, path, chunk_size)

    @staticmethod
    def write_data(data: GetItemLenProtocol, path: str, chunk_size: int=1024) -> None:
        # We save all full chunks
        # If we have leftover data, we send it to the main process, which will gather it from all subprocess
        # and save into more full chunks, with a possible leftover chunk that is also saved. 
        # This way all chunks are the same size (except the last one)
        # and we can directly calculate which one to load in the dataloader
        for i in range(0, len(data) // chunk_size):
            min_index =  i * chunk_size
            max_index = (i + 1) * chunk_size
            chunk = data[min_index:max_index]
            
            chunk_id = MpUtil.get_and_increment_atomic_counter()
            filename = f'{path}/chunk_{chunk_id}.pt'
            PipelineObjectBase.log_debug(f"Saving chunk {chunk_id} to {filename}")
            torch.save(chunk, filename)
        
        
        leftover = list(data[len(data) // chunk_size * chunk_size:])
        PipelineObjectBase.log_debug(f"Leftover data: {len(leftover)}")

        if MpUtil.is_main_process():
            length_before_gather = MpUtil.get_atomic_counter() * chunk_size

        # We gather all leftover data on the main process 
        gathered_data = MpUtil.gather_on_process(leftover)

        # We save the leftover data in chunks
        if MpUtil.is_main_process():
            for i in range(0, len(gathered_data) // chunk_size):
                min_index =  i * chunk_size
                max_index = (i + 1) * chunk_size
                chunk = gathered_data[min_index:max_index]
                
                chunk_id = MpUtil.get_and_increment_atomic_counter()
                PipelineObjectBase.log_debug(f"Saving chunk {chunk_id} to {filename}")
                filename = f'{path}/chunk_{chunk_id}.pt'
                torch.save(chunk, filename)
            
            # We save the leftover data in a chunk
            if len(gathered_data) % chunk_size != 0:
                chunk_id = MpUtil.get_and_increment_atomic_counter()
                filename = f'{path}/chunk_{chunk_id}.pt'
                PipelineObjectBase.log_debug(f"Saving chunk {chunk_id} to {filename}")
                torch.save(gathered_data[len(gathered_data) // chunk_size * chunk_size:], filename)
        
        # We save the metadata
        if MpUtil.is_main_process():
            metadata_info = {
                'chunk_size': chunk_size,
                'total_length': length_before_gather + len(gathered_data)
            }

            torch.save(metadata_info, f'{path}/metadata.pt')
        
    @lru_cache(maxsize=LRU_SIZE)  
    def get_chunk(self, chunk_index):
        filename = f'{self.path}/chunk_{chunk_index}.pt'
        return torch.load(filename)

    def __len__(self):
        # Returns the length that this process sees of the data. 
        return (self.total_length // self.num_workers + (1 if self.worker_id < self.total_length % self.num_workers else 0))

    def __getitem__(self, index):
        # Indexes what this process sees of the data. 
        if index >= len(self):
            raise IndexError("Index out of range")
        chunk_index, within_chunk_index = divmod(index * self.num_workers + self.worker_id, self.chunk_size)
        chunk = self.get_chunk(chunk_index)

        return chunk[within_chunk_index]


    def add_data(self, data: GetItemLenProtocol):
        """Add data to an already existing cache
        """
        MpUtil.barrier()

        PipelineObjectBase.log_debug(f"Adding data to cache {self.path}")


        all_data: List[Any] = [d for d in data]

        if MpUtil.is_main_process():
            # We set the counter to the next chunk after the last chunk
            if self.total_length == 0:
                MpUtil.set_atomic_counter(0)
            else:
                last_chunk = (self.total_length - 1) // self.chunk_size
                MpUtil.set_atomic_counter(last_chunk + 1)

                # It can happen that the last block in the cache is not full in case we are chunking. 
                # So if we are chunking, we add the last block to the data of process 0 and proceed with adding data to the cache. 
                if self.chunk_size > 1:
                    chunk = self.get_chunk(last_chunk)
                    all_data.extend(chunk)
                    
                    # Decrease the counter, so we will overwrite the previous last chunk
                    MpUtil.set_atomic_counter(last_chunk - 1)

        MpUtil.barrier()

        ChunkDataLoader.write_data(data, self.path, self.chunk_size)

        MpUtil.barrier()

        self.load_metadata_and_set_length()

