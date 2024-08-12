import fcntl
import os
import shutil
import subprocess
from typing import Optional
from rmp_dl.learning.data.pipeline.multiprocess_util import MpUtil
from rmp_dl.learning.data.pipeline.nodes.chunk_dataloader import ChunkDataLoader

import shutil
from rmp_dl.learning.data.pipeline.pipeline_object_base import PipelineObjectBase

import wandb

class CacheOnDisk(PipelineObjectBase):
    # Get dir of this file
    lock_dir = os.path.dirname(os.path.abspath(__file__)) + "/cache_locks"
    os.makedirs(lock_dir, exist_ok=True)

    def __init__(self, 
                 # User arguments 
                 cache_name: str, 
                 store_permanently: bool, 
                 chunk_size: int, 
                 keep_in_memory: bool, 

                 # Set by the pipeline
                 dataset_short_term_caching_path: str, # Mandatory
                 dataset_long_term_storage_path: Optional[str], # Only used if store_permanently is True
                 **kwargs):
        """The cache on disk module can be used for short term storage to decrease RAM usage during a run, and for long term storage
        such that a dataset does not have to be regenerated. 
        Both use cases are non-exclusive, so you can do either one or both. 

        This module is also used to aggregate data across epochs. It is integrated within this node, as saving to disk is a
        kind of natural aggregation operation. 

        The module has a 2 stage setup, first resolving the long term storage, and then resolving the short term storage.

        For long term storage: 
        Set store_permanently to True and give a cache name. The cache will be stored in $cache_directory/$cache_name.tar.gz
        $cache_directory is set by passing a command line argument to the main.py script, as this is detmermined on which type of
        computing system you are running, e.g. when running on Euler, long term storage is on the scratch disk. Upon initialization, 
        the module will check if the tarball with the cache name exists, if it does, 
        it will copy it to $dataset_short_term_caching_path/$cache_name/...
        and extract it there. If it does not exist, it will create the cache in the short term caching path (by doing the setup of upstream nodes
        which generate their data and writing this data to the short term caching path), compress it to a tarball and 
        copy it to long term storage. In both cases, we should have a cache of the necessary data in the short term caching path.
        The long term storage then clears the memory of any upstream nodes, after which we run  short term storage setup. 

        For short term storage:
        If $keep_in_memory is set to True, the module will load the cache into memory. For smaller datasets this is preferred. 
        For larger datasets, we can set $keep_in_memory to False, in which case the module will load the cache from disk when needed to decrease RAM uage. 
        Important things here: Not keeping the dataset in memory is mainly meant for when we use sequenced training, 
        as the datasets we need for this can become very large. 
        In this case it works fine, as a single sequence is a couple MBs, 
        so with multiple processes loading we can keep the GPU occupied during training. 
        If you have a non-sequenced training, we will need to extract a single observation from the data which will be very small, 
        so during training we will have to fetch millions of tiny observations. 
        During training, this is a problem, as we need randomized access patterns. 
        This will result in millions of tiny observations being randomly loaded, which is very slow.
        So do not use this for non-sequenced datasets during training!!!!!
        During validation it is okay, as we don't need to randomize the data, 
        so data can be sequentially accessed and this module will make sure that chunks are cached such that
        sequential accesses are optimized. 
        Also note that if you put this node before the flatten node (which is preferred), 
        it will always save the sequenced data together as they are in 1 numpy array. 
        So it will already be kind of chunked (you can chunk more if you want using the chunk_size parameter, though not really recommended). 
        If you put it after the flatten node it will save the non-sequenced data, which is not chunked.
        So in the latter case you should definitely chunk it some more by specifiying a chunk size, 
        but I don't see any use case for putting it after a flaten node anyway, 
        as we much prefer to cache upstream, as we can then decide how to flatten the data, and do any other downstream operations on it. 
        And like I said, using this module for non-sequenced data during training is a bad idea anyway.
        

        Args:
            cache_name (str): _description_
            cache_directory (str): _description_
            store_permanently (bool): _description_
            chunk_size (int): _description_
            keep_in_memory (bool, optional): _description_. Defaults to False.
        """
        super().__init__(**kwargs)

        # We create a lockfile, as we may have multiple runs starting at the same time. In that case it can happen 
        # that they start writing to the same cache directory, which is not what we want. 
        self.lockfile_path = CacheOnDisk.lock_dir + "/" + cache_name + ".lock"
        with open(self.lockfile_path, "a+") as _: pass  # Create lockfile if it doesnt exist already
        self.cache_name = cache_name
        self.short_term_caching_path = dataset_short_term_caching_path
        if wandb.run:
            self.short_term_caching_path += "/" + wandb.run.id
        else:
            self.short_term_caching_path += "/test"
        
        self.short_term_caching_path += "/" + cache_name 
        os.makedirs(self.short_term_caching_path)
        self.long_term_storage_path: str = dataset_long_term_storage_path # type: ignore -> We throw an exception later on if this is not set and we need it
        self.store_permanently = store_permanently
        self.chunk_size = chunk_size    
        self.chunk_dataloader: ChunkDataLoader = None # type: ignore ->This will be set in setup
        self.keep_in_memory = keep_in_memory

        if self.store_permanently and self.long_term_storage_path is None:
            raise ValueError("If store_permanently is True, you must specify a long term storage path")

        if self.keep_in_memory:
            self.observations = []

    def setup(self):
        if len(self.inputs) == 0:
            raise ValueError(f"Input not set for CacheOnDisk {self.name}")
        self.step1_long_term_setup()
        self.step2_short_term_setup()
    
    def step1_long_term_setup(self):
        if not self.store_permanently:
            self.create_cache()
        else:
            self._create_cache_from_long_term_storage_if_exists()
        # Clear memory of upstream nodes
        self._get_input().clear_data()

        self.chunk_dataloader = ChunkDataLoader(self.short_term_caching_path)

    def step2_short_term_setup(self):
        if self.keep_in_memory:
            self._load_cache_into_memory()

    def on_epoch_start(self) -> None:
        super().on_epoch_start()
        # If we have an aggregator node upstream, we need to add the data to the cache

        # To make sure all processes follow the same code path in case some nodes have no new upstream data, 
        # we use an atomic counter to check if new data has been created for any process. 
        MpUtil.reset_atomic_counter_with_barrier()
        
        if len(self._get_input()) > 0:
            MpUtil.get_and_increment_atomic_counter()
        
        MpUtil.barrier()
        if MpUtil.get_atomic_counter() > 0: # This means that new data has been created. We need to add it to the cache
            MpUtil.reset_atomic_counter_with_barrier()
            # You are not allowed to store permanently. 
            # TODO: Fix this, even though this probably never happens, it is nice to be able to do this if we want to. 
            # I've disabled it now, because it requires repacking all the data every epoch with the current setup, which is 
            # too expensive. We would have to switch to saving long term setup at the end. However, in normal cases
            # we don't want that, as usually at the start all the data is there and we want to save it ASAP, as other runs
            # on other nodes on the cluster may be waiting for the long term storage data. 
            # Like I said, we probably never want to store permanently if we are aggregating, so low priority to fix this. 
            if self.store_permanently:
                raise RuntimeError("You can't store data permanently downstream of an aggregator")

            self.log_debug(f"Adding data to cache: {len(self._get_input())}")
            
            self.chunk_dataloader.add_data(self._get_input())
            self._get_input().clear_data()
            MpUtil.barrier()

            self.step2_short_term_setup()

    def _load_cache_into_memory(self):
        self.log_debug("Loading cache into memory")
        self.observations = [obs for obs in self.chunk_dataloader]
        self.log_debug(f"Cache loaded into memory. Size: {len(self.observations)}")

    def _create_cache_from_long_term_storage_if_exists(self):
        # Creates cache from long term storage if it exists, otherwise it will create the cache from scratch
        # by running the upstream nodes (self.create_cache())

        # Check if long term cache file exists
        # We are using lockfiles here. Note that these lockfiles are there to synchronize 
        # ~across different runs~, and not across different processes within the same run. 
        # I.e. if we start 2 runs at the same time, we want to make sure that only one of them creates the cache tarball
        # but the cache tarball is created by all of the processes in that run.
        # Note that we are assuming that local storage paths are unique across runs (which should be true when running on a cluster
        # as local scratch is unique for a run. )
        # 2023-10-02: The code below has been somewhat tested with a previous version (i.e. the runs in parallel thing). 
        # The new version is roughly the same, but introduced the separation of long term and short term storage, 
        # which makes it operate slightly differently. The more complicated multiple runs logic should still work out the same though. 
        # Single run logic runs fine and has been tested. So just be careful when starting multiple runs with a new cache. 
        with open(self.lockfile_path, "r+") as lockfile:
            exists = False
            fcntl.flock(lockfile, fcntl.LOCK_SH)
            try:
                if self._exists_valid_cache_tarball():
                    exists = True
            finally:
                fcntl.flock(lockfile, fcntl.LOCK_UN)

            # We want all processes to make it to here, as below the main process will lock the 
            # lockfile exclusively, and we want all processes to make it past the lock above before that.
            MpUtil.barrier()

            if not exists:
                # It can happen that 2 runs traverse the above code at the exact same time and end up here. 
                # In that case, we want to make sure that only one of them generates the cache and creates the tarball
                # So now we lock the lockfile exclusively and check again if the cache directory exists.
                # We don't want to lock the file exclusively above, as in most cases the cache directory will exist,
                # and we don't want to wait for all shared locks to be released

                if MpUtil.is_main_process():
                    # As explained above, we only want to lock the file exclusively if we are in the main process
                    # as we want the subprocesses to also help with creating the cache
                    fcntl.flock(lockfile, fcntl.LOCK_EX)
                # To make sure the subprocesses can only pass if the main process has passed we use a barrier
                # So if we have 2 runs ag the same time, only one main process will get past the lock above, and the other will wait. 
                # The subprocesses corresponding to the main process that is waiting, will get stuck here. 
                # Once the lock is removed, which means that the cache is created, the main process will get unstuck, 
                # get to the barrier, and all of the processes get past. The 2nd check for the cache below will make sure
                # that this run doesn't create the cache again, and just loads it from the other run. 
                MpUtil.barrier()

                if not self._exists_valid_cache_tarball():
                    try:
                        self.create_cache()
                        if MpUtil.is_main_process():
                            self._copy_short_term_cache_to_long_term_storage()
                    finally:
                        if MpUtil.is_main_process():
                            fcntl.flock(lockfile, fcntl.LOCK_UN)
                else: 
                    exists = True

        if exists and MpUtil.is_main_process():
            self._copy_long_term_storage_to_short_term_cache()

        MpUtil.barrier()

    def _copy_short_term_cache_to_long_term_storage(self):
        # We use process Popen to first compress the cache, and then copy it to the long term storage
        
        self.log("Compressing short term cache into tarball")

        parent_dir, dir_to_compress = os.path.split(self.short_term_caching_path)
        command = ["tar", "-zcf", f"{self.short_term_caching_path}.tar.gz", "-C", parent_dir, dir_to_compress]
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process.check_returncode()
                
        self.log("Compressing finished, copying tarball to long term storage.")
        # Copy command
        command = ["cp", f"{self.short_term_caching_path}.tar.gz", self.long_term_storage_path]
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process.check_returncode()

        self.log("Finished copying tarball to long term storage")

    def _copy_long_term_storage_to_short_term_cache(self):
        # First we copy the tarball from long term storage to short term cache
        self.log("Copying tarball from long term storage to short term cache")
        command = ["cp", f"{self.long_term_storage_path}/{self.cache_name}.tar.gz", self.short_term_caching_path]
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process.check_returncode()

        # Then we extract the tarball
        self.log("Extracting tarball")
        command = ["tar", "-zxf", f"{self.short_term_caching_path}/{self.cache_name}.tar.gz", "-C", self.short_term_caching_path + "/../"]
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process.check_returncode()

        self.log("Finished extracting tarball")

    def _exists_valid_cache_tarball(self):
        tarball = self.long_term_storage_path + "/" + self.cache_name + ".tar.gz"
        return ChunkDataLoader.is_cache(tarball, self.chunk_size)

    def create_cache(self):
        MpUtil.barrier() # We wait for all processes to be ready

        self._get_input().setup()
        
        MpUtil.barrier()

        if MpUtil.is_main_process():
            self.log_info(f"Creating short term cache in {self.short_term_caching_path}")
            
        ChunkDataLoader.initialize_with_data(self._get_input(), self.short_term_caching_path, self.chunk_size)

        if MpUtil.is_main_process():
            self.log_info("Cache created")
        self._get_input().clear_data()
        MpUtil.barrier()

    def load_cache(self, cache_directory):
        self.full_cache_directory = cache_directory
        self.chunk_dataloader = ChunkDataLoader(cache_directory)

        # TODO: Move this inside the chunkdataloader
        if self.keep_in_memory:
            self.observations = [obs for obs in self.chunk_dataloader]

    def __len__(self): 
        if self.keep_in_memory:
            return len(self.observations)
        return len(self.chunk_dataloader)

    def __getitem__(self, index):
        if self.keep_in_memory:
            return self.observations[index]
        return self.chunk_dataloader[index]
    
    def clear_data(self): # This class doesn't consume memory, so we don't need to clear anything
        # We clear input classes after creating cache, so we don't need to clear them here
        # TODO: It's a bit vague when to propagate clear upwards. Figure this out
        pass
