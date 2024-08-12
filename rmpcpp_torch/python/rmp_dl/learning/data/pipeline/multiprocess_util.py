
import copy
from typing import Iterable, List, TypeVar

import torch


class MpUtil:
    non_multiprocessed_counter = 0
    T = TypeVar("T")

    @staticmethod
    def id():
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return 0
        return worker_info.id

    @staticmethod
    def num_workers():
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return 1
        return worker_info.num_workers

    @staticmethod
    def get_seed():
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return 0
        return worker_info.seed

    @staticmethod
    def gather_on_process(data: Iterable[T], process_id=0) -> List[T]:
        """Basic function to gather all data to a single process. 
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # If we are not multiprocessing we just return the data
            return list(data)

        if process_id >= worker_info.num_workers:
            raise ValueError(f"Process id {process_id} is larger than the number of workers {worker_info.num_workers}")
        
        queue = worker_info.dataset.queue
        barrier = worker_info.dataset.barrier_object

        # We block until all processes are ready, as we reuse the same queue for any type of collective operation, so we make sure that
        # no process is still using the queue
        barrier.wait() 

        # The queue should now for sure be empty
        if not queue.empty():
            raise ValueError("Queue is not empty when gathering data. This should not happen")
        
        barrier.wait() # All queue checks have to finish before continuing

        gathered_data = []

        if process_id != worker_info.id:
            # Not the gathering process, so we start sending data
            for d in data:
                queue.put(d)
            queue.put(None) # Sentinel signalling end of data
        else:
            # Gathering process, so we start receiving data
            gathered_data.extend(data) # First add our own data
            sentinel_count = 0
            while sentinel_count < worker_info.num_workers - 1: # We have to receive num_workers - 1 sentinels, as we don't receive our own sentinel
                d = queue.get()
                if d is None:
                    sentinel_count += 1
                else:
                    gathered_data.append(copy.deepcopy(d))
                    del d # So we don't get any file descriptor leaks in case of large amounts of data

        # We block until all processes are ready and the queue is emptied
        # Not really too important this as we (should) also block at the start of any other collective operation
        # But I like the consistency and extra safety 
        barrier.wait()

        return gathered_data


    @staticmethod
    def barrier():
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return

        barrier = worker_info.dataset.barrier_object
        barrier.wait()

    @staticmethod
    def is_main_process():
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return True
        return worker_info.id == 0
    
    @staticmethod
    def reset_atomic_counter_with_barrier():
        MpUtil.barrier()
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            MpUtil.non_multiprocessed_counter = 0
            return
        # Only the main process needs to reset the counter
        if worker_info.id == 0:
            worker_info.dataset.counter.value = 0
        MpUtil.barrier()

    @staticmethod
    def get_and_increment_atomic_counter():
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            value = MpUtil.non_multiprocessed_counter
            MpUtil.non_multiprocessed_counter += 1
            return value
        with worker_info.dataset.counter_lock:
            value = worker_info.dataset.counter.value
            worker_info.dataset.counter.value += 1

        return value
    
    @staticmethod
    def set_atomic_counter(value):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            MpUtil.non_multiprocessed_counter = value
            return
        # Don't need to lock the lock, as we are not incrementing, so this is safe
        # as the counter is an atomic value. 
        worker_info.dataset.counter.value = value


    @staticmethod
    def get_atomic_counter():
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return MpUtil.non_multiprocessed_counter
        return worker_info.dataset.counter.value


    @staticmethod
    def is_multiprocessed():
        worker_info = torch.utils.data.get_worker_info()
        return worker_info is not None
    
    @staticmethod
    def get_rollout_to_dataset_queue():
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            raise RuntimeError("Trying to get queues while not multiprocessing")
        return worker_info.dataset.rollout_to_dataset_queue[worker_info.id]

    @staticmethod
    def get_dataset_to_rollout_queue():
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            raise RuntimeError("Trying to get queues while not multiprocessing")
        return worker_info.dataset.dataset_to_rollout_queue[worker_info.id]