import logging
from threading import BrokenBarrierError, Thread
import time
import numpy as np
from rmp_dl.learning.data.pipeline.data_pipeline import DataPipeline
from rmp_dl.learning.data.pipeline.multiprocess_communication import DatasetToMainCommunicationTypes, DatasetToRolloutCommunicationTypes, MainToDatasetCommunicationTypes, RolloutToDatasetCommunicationTypes, RolloutWorker, WandbLoggerCommunicationTypes
from rmp_dl.learning.data.pipeline.multiprocess_util import MpUtil
import torch

import wandb

logger = logging.getLogger("rmpcpp_torch")

class DataPipelineMpWrapper(torch.utils.data.IterableDataset, ):
    # We have to do iterable dataset, as our dataset changes size during training. Pytorch lightning checks size only once, 
    # at the start of training, and before making the workers, so we can't use a normal map-style dataset. 
    # One way around this would be to tell the trainer to reload the dataloader every epoch, but this results in state
    # not being persistent in the dataset workers, which we kind need in some of our cases (e.g. when we don't cache on disk, 
    # which we do for some validation sets, we need to persist the dataset to keep the data in RAM) 
    def __init__(self, name,
                  num_workers: int, 
                  manager, 
                  max_workers=None,
                  requires_rollout_workers_after_setup=True, 
                  *args, **kwargs):
        """Wrapper that makes data pipeline multiprocessing compatible.

        Args:
            name (str): Name of the dataset
            num_workers (int): Number of workers 
            manager: Manager object used to create the queues
            max_workers (int, optional): Maximum number of workers of this dataset. If none the global maximum (num_workers) is used, 
                otherwise it is min(max_workers, num_workers). Defaults to None.   

            requires_rollout_workers_after_setup (bool, optional): Whether the pipeline does rollouts after initial setup. 
                CAREFUL: Setting this to false while there are nodes that actually do require rollouts will result in a deadlock. 
                TODO: Fix this and throw an exception instead. Defaults to True.
            randomize_order (bool, optional): Whether to randomize the order every epoch. Defaults to False.
        """
        # Because we pass some queues as arguments to the worker_processes, we need to create them using a manager from the main process
        # (which has to live outside this class, as this class is multiprocessed by pytorch spawning multiple workers from it)
        # Otherwise we get this error:
        # Queue objects should only be shared between processes through inheritance
        # 
        # Furthermore, there are a bunch of other weird issues that could arise when not using a manager to create the queues, 
        # such as this warning (scroll sligthly up)
        # https://docs.python.org/3/library/multiprocessing.html?highlight=queue#multiprocessing.Queue
        self.name = name
        self.requires_rollout_workers_after_setup = requires_rollout_workers_after_setup
        num_workers = num_workers if not max_workers else min(max_workers, num_workers)
        self.num_workers = num_workers
        # Queue used to pass data between workers
        self.queue = manager.Queue(num_workers * 10)
        # Barrier used to synchronize workers
        self.barrier_object = manager.Barrier(self.num_workers)
        # Queue used to pass data from workers to main process for logging
        self.logging_queue = manager.Queue(num_workers * 100)
        # Queue used to log wandb data
        self.wandb_logging_queue = manager.Queue(num_workers * 100)

        # For some reason if you instantiate a Value via a manager, it does not have a 
        # get_lock() method, as the documentation suggests here:
        # https://docs.python.org/3.6/library/multiprocessing.html#multiprocessing.Value
        # and here
        # https://docs.python.org/3/library/multiprocessing.html#sharing-state-between-processes
        # 
        # This seems to be a bug in python, or just a mistake or something, see issue here:
        # https://github.com/python/cpython/issues/79967
        # Interestingly enough that has been open since 2019, and it is still not fixed. 
        # Anyway, the solution is to just explicitly also create a lock object to control
        # incrementing the value
        self.counter = manager.Value('i', 0)
        self.counter_lock = manager.Lock()

        self.pipeline = DataPipeline(*args, **kwargs)

        # We can't do any rollouts in any of the pipeline workers, as the moment we do any CUDA related things, 
        # a CUDA context is created, which takes anywhere from 700MB to 2GB of GPU memory. If we have 10 workers, 
        # this will eat 20GB. This is okay as the GPU can handle this, but once the context is created,
        # there is no way to destroy it other than exiting the process, e.g. :
        # 
        # https://github.com/pytorch/pytorch/issues/17157
        # 
        # And after rollouts are done, we need the GPU memory for training. However, we can't destroy the worker processes,
        # as they need to feed the data to the main process. 
        # We are also not allowed to spawn another process in the workers as we then get:
        # AssertionError: daemonic processes are not allowed to have children
        # Trying to make the workers (which are instantiated by pytorch) non-daemonic is either 
        # impossible or very hard and strongly discouraged, haven't looked too much into it but seems like a bad idea. 
        #

        # So to get around all of this, we spawn a new process for each dataset_worker at the start of the epoch which can do the rollouts. 
        # We call the pytorch dataset loader workers `dataset_workers`. 
        # And the workers that do the rollouts `rollout_workers`.
        # Every rollout_worker is initialized with an input and output queue, which is used to communicate with the dataset_worker process.
        # We initialize these queues here: an array of input queues, and an array of output queues, one for each dataset_worker rollout_worker pair.
        # As the dataset_workers get initialized as copies of this object, they all have access to these queues. 
        # The rollout_workers are then started by this class in the main process in the corresponding setup and on_epoch_end functions. 
        # The dataset_worker starts doing it's setup before iterating, and if it needs to do rollouts, it can use the queues to communicate them. 
        # Once the dataset workers have finished their setup/pre-iteration tasks, 
        # the main process sends a message to the rollout_worker to terminate, freeing up the CUDA memory. 

        # As a sidenote, we also use the queues below to communicate the latest model to the dataset_worker at the end of the epoch, 
        # as the dataset_worker can then use this for rollouts (it can send it along to the rollout_worker) if necessary. 
        # We don't send it immediately to the rollout worker, because we can do different rollouts with different configurations, 
        # so the dataset_worker is in charge of figuring out when and how to send it to the rollout_worker. 

        # Also, it is possible that a dataset does not require the rollout workers after setup, in which they are not spawned 
        # by the main process (saving a fair amount of time, as spawning these workers takes ~10-20s)

        # Queue used to send data from the rollout worker to the dataset worker
        self.rollout_to_dataset_queue = [manager.Queue(10) for _ in range(num_workers)] 

        # These are used to send data from the dataset worker to the rollout_worker.
        # At the start of sampling, we first send the sampling function to the rollout_worker, 
        # which contains the model, or instructions on how to get the model from disk (not too sure yet what I'll do)
        # Then we send the rollout_worker the world constructors on which it will do rollouts. 
        self.dataset_to_rollout_queue = [manager.Queue(10) for _ in range(num_workers)] 

        # Queue used to send data from the main process to the dataset worker
        self.main_to_dataset_queue = [manager.Queue(10) for _ in range(num_workers)]

        # Queue used to send data from the dataset_worker to the main process
        self.dataset_to_main_queue = [manager.Queue(10) for _ in range(num_workers)]
 
        # As there are multiple things we might send over these queues to communicate, we need to make sure we 
        # meticulously define what means what. See comments in multiprocess_communication for more info

        self.rollout_workers_started = False

        # Overwritten by workers when they do their setup
        self.np_rng = np.random.default_rng(torch.initial_seed())

    def _info_log(self, msg):
        if not MpUtil.is_main_process():
            return  # Should maybe throw exception, this function is only to be used in functions only used by the main process
        logger.info(f"{self.name}: {msg}")

    def _debug_log(self, msg):
        if not MpUtil.is_main_process():
            return  # Should maybe throw exception, this function is only to be used in functions only used by the main process
        logger.debug(f"{self.name}: {msg}")

    def _logging_handler(self):
        while True:
            try: 
                if not self.logging_queue.empty():
                    result = self.logging_queue.get()
                    if result is None: # Sentinel
                        break
                    # Bit ugly, but this covers the case where we log a message tuple without level (if we want that for any reason)
                    try: 
                        level, msg = result
                        logger.log(level, f"{self.name}: {msg}")
                    except:
                        self._debug_log(result)
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(e)
    
    def _wandb_logging_handler(self):
        while True:
            try: 
                if not self.wandb_logging_queue.empty():
                    data = self.wandb_logging_queue.get()
                    if data is None: # Sentinel
                        break
                    data_type, data = data
                    if wandb.run:
                        if data_type == WandbLoggerCommunicationTypes.NORMAL:
                            wandb.log(**data)
                        elif data_type == WandbLoggerCommunicationTypes.ARTIFACT:
                            wandb.run.log_artifact(**data)
                        logger.debug(f"wandb logging handler logged data to wandb")
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(e)


    def __iter__(self):
        yield from self.pipeline
            
    def post_setup(self, rollout_workers):
        if not self.num_workers > 0:
            return
        # This is only called if we are multiprocessing
        self.gracefully_kill_rollout_workers(rollout_workers)

        if not self.requires_rollout_workers_after_setup:
            for i in range(self.num_workers):
                # Signal such that dataset workers throw an exception if they do try to do rollouts
                # instead of deadlocking. 
                self.rollout_to_dataset_queue[i].put((RolloutToDatasetCommunicationTypes.NO_ROLLOUT_WORKER, None))


    def start_rollout_workers(self):
        if self.rollout_workers_started:
            raise RuntimeError("Trying to start rollout workers, but they are already started")

        self._debug_log("Starting rollout workers")
        workers = []
        for i in range(self.num_workers):
            # The rollout workers are spawned, and then no object actually maintains a reference to it. I
            # I can't store them in this class, as this class will be multiprocessed by pytorch, and we are not allowed
            # to store any objects that are not picklable in this class. So we can't store process objects. 
            # The rollout workers are thus not stored anywhere, and we can only communicate via the queues. 
            # The queues are used to shutdown the rollout worker. I should probably refactor this, such that the rollout workers 
            # are created outside this class, and that the queues are passed to this class, and that the outer class can keep track of the workers. 
            # TODO
            rollout_worker = RolloutWorker(self.rollout_to_dataset_queue[i], self.dataset_to_rollout_queue[i], self.logging_queue, i)
            process = rollout_worker.start()
            workers.append(process)
        self.rollout_workers_started = True
        return workers
    
    def force_kill_rollout_workers(self, workers):
        """We use this in case we have caught an exception somewhere. 
        Gracefully killing the rollout workers will most likely not be possible, as they might deadlock if an 
        excpetion is raised. So we force kill them.
        """
        self._debug_log("Force killing rollout workers")
        for i, worker in enumerate(workers):
            self._debug_log(f"Force killing rollout worker {i}")
            worker.terminate()
        self._debug_log("Force killed rollout workers")
        self.rollout_workers_started = False

    def gracefully_kill_rollout_workers(self, workers):
        self._debug_log("Killing rollout workers")
        if not self.rollout_workers_started:
            raise RuntimeError("Trying to kill rollout workers, but they are not started")
        for i in range(self.num_workers):
            # We kill the rollout_worker here
            # TODO: SLightly ugly to reuse the dataset_to_rollout queue. Maybe rename the queue
            self.dataset_to_rollout_queue[i].put((DatasetToRolloutCommunicationTypes.KILL, None))                
        for worker in workers:
            worker.join()
        self._debug_log("Killed rollout workers")
        self.rollout_workers_started = False

    def pre_iteration_task(self, model, current_epoch):
        self._debug_log("Starting pre-iteration tasks")
        model_args, state_dict = model.get_model_args_and_state_dict()
        # Empty the cache of the main process, which may have just done some RNN training and be occupying 10s of GBs of GPU memory
        torch.cuda.empty_cache()
        # This function is only called in the main process.
        if self.num_workers == 0:
            # We are not doing multiprocessing, so just setting the model and doing pre iteration tasks
            self.pipeline.set_model_args_and_state_dict(model_args, state_dict)
            self.pipeline.set_current_epoch(current_epoch)
            self.pipeline.on_epoch_start()
            return

        # We are multiprocessing
        # In this case, this function sends the model to the workers, and signals them to start doing their tasks such as rollouts
        if self.requires_rollout_workers_after_setup:
            # First we check if the communication queues are empty
            for queue in self.rollout_to_dataset_queue:
                if not queue.empty():
                    components = []
                    while not queue.empty():
                        components.append(queue.get())
                        print(queue.get())
                    raise RuntimeError("Queue should be empty")
            for queue in self.dataset_to_rollout_queue:
                if not queue.empty():
                    components = []
                    while not queue.empty():
                        components.append(queue.get())
                        print(queue.get())
                    raise RuntimeError("Queue should be empty")
                
            # We start the rollout workers in a separate process, such that we can kill them and their expensive cuda context before we do any training
            rollout_workers = self.start_rollout_workers()

        # We now have to communicate the model to the dataset workers, such that they can do rollouts if necessary
        # We send the model state dict, and the parameters used to create it
        # Model is of type RayLigthningModule, which contains the RayModel as attribute, and the model_args as attribute which contains
        # the arguments used to create the RayModel
        # I.e. RayModel can be recreated from these parameters as model = RayModel(**model_args); model.load_state_dict(state_dict)
        
        # We don't want the dataset workers to initialize a cuda context, so we copy to CPU before passing on
        # The dataset workers are in charge of moving it to the rollout workers, which are then allowed to put it on to the GPU, as
        # they will be killed before the training epoch starts so the cuda context will be freed. 
        state_dict = {k: v.cpu() for k, v in state_dict.items()} 
        for queue in self.main_to_dataset_queue:
            queue.put((MainToDatasetCommunicationTypes.MODEL_AND_CURRENT_EPOCH, ((model_args, state_dict), current_epoch)))

            # We also pass the signal to start working
            queue.put((MainToDatasetCommunicationTypes.START_PRE_ITERATION_TASKS, None))
        
        done = [False for _ in range(self.num_workers)]

        # Wait until all dataset workers give the ready signal
        # We do a checking loop, so we can catch any exceptions put into the queue
        try:
            while not all(done):
                time.sleep(0.1)
                for i, d in enumerate(done):
                    if d: continue
                    if not self.dataset_to_main_queue[i].empty():
                        data_type, data = self.dataset_to_main_queue[i].get()
                        if data_type == DatasetToMainCommunicationTypes.FINISHED_PRE_ITERATION_TASKS:
                            done[i] = True
                        elif data_type == DatasetToMainCommunicationTypes.EXCEPTION:
                            print(f"Caught error in dataset worker process {i}")
                            raise data
                        else:
                            raise RuntimeError("Unknown data type")

                    # We also check if the rollout worker is still alive
                    if self.requires_rollout_workers_after_setup:
                        exit_code = rollout_workers[i].exitcode
                        if exit_code is not None and exit_code > 1:
                            # If the exit code is 1, it means the worker threw an exception, which will be communicated via the queues
                            # above, so we don't have to do anything. If the exit code is anything else, it means the system killed the worker.
                            # They can get killed by e.g. slurm if we are consuming too much memory. Detect this, and exit
                            raise RuntimeError("Rollout workers died unexpectedly")
                    
        except Exception as e:
            if self.requires_rollout_workers_after_setup: # I.e. they are initialized and we need to kill them
                self.force_kill_rollout_workers(rollout_workers)
            self.barrier_object.abort()
            self.teardown()
            raise e
        
        if self.requires_rollout_workers_after_setup: # I.e. they are initialized and we need to kill them
            # Kill all the rollout_workers gracefully. 
            self.gracefully_kill_rollout_workers(rollout_workers)
        
        self._debug_log("Finished pre-iteration tasks")

    def check_rollout_workers_alive(self, rollout_workers):
        for rollout_worker in rollout_workers:
            if rollout_worker.exitcode is not None:
                return False
            
        return True
            
    def setup(self):
        # This is only called in the main thread
        # We are not multiprocessing
        if self.num_workers == 0 and torch.utils.data.get_worker_info() is None:
            self.pipeline.setup()

        # We are multiprocessing and we are the main thread
        if self.num_workers > 0 and torch.utils.data.get_worker_info() is None:
            # We have to start the logging handler thread
            logging_handler = Thread(target=self._logging_handler)
            logging_handler.start()

            # And the wandb logger thread
            wandb_logging_handler = Thread(target=self._wandb_logging_handler)
            wandb_logging_handler.start()


    def teardown(self):
        if self.num_workers > 0 and torch.utils.data.get_worker_info() is None:
            self.logging_queue.put(None) # Sentinel signalling exit
            self.wandb_logging_queue.put(None) # Sentinel signalling exit
        self.pipeline.teardown()

    def _worker__setup(self):
        try:
            self.np_rng = np.random.default_rng(torch.utils.data.get_worker_info().seed)
            # We setup the pipeline
            self.pipeline.setup()

            # The worker loop will be killed by the main process once it is no longer necessary
            worker_loop = Thread(target=self._worker__loop)
            worker_loop.start()
        except BrokenBarrierError:
            # In this case some another process has exited. So this process also just quits
            return
        except Exception as e:
            self.dataset_to_main_queue[MpUtil.id()].put((DatasetToMainCommunicationTypes.EXCEPTION, e))
            self.barrier_object.abort()
            raise e


    def _worker__loop(self):
        # We start a worker threading loop. 
        # This loop is responsible for doing the correct rollouts during training, which is timed by the main thread using 
        # signals through the queue. 
        queue = self.main_to_dataset_queue[MpUtil.id()]
        try:
            while True:
                # I do a loop here instead of blocking, so it is easier to debug when a thread is non responsive
                if not queue.empty():
                    data_type, data = queue.get()
                    if data_type == MainToDatasetCommunicationTypes.MODEL_AND_CURRENT_EPOCH:
                        ((model_args, state_dict), current_epoch) = data
                    
                        self.pipeline.set_model_args_and_state_dict(model_args, state_dict)
                        self.pipeline.set_current_epoch(current_epoch)
                    elif data_type == MainToDatasetCommunicationTypes.START_PRE_ITERATION_TASKS:
                        self._worker__do_pre_iteration_tasks()
                        # Signal that we are done with the pre iteration tasks
                        self.dataset_to_main_queue[MpUtil.id()].put((DatasetToMainCommunicationTypes.FINISHED_PRE_ITERATION_TASKS, None))
                    elif data_type == MainToDatasetCommunicationTypes.KILL:
                        return
                    else: 
                        raise ValueError("Unexpected communication type")
                else:
                    time.sleep(0.1)
        except BrokenBarrierError:
            # In this case some another process has exited. So this process also just quits
            return
        except Exception as e:
            self.dataset_to_main_queue[MpUtil.id()].put((DatasetToMainCommunicationTypes.EXCEPTION, e))
            self.barrier_object.abort()
            raise e

    def _worker__do_pre_iteration_tasks(self):
        # This is where we do any pre-epoch tasks such as doing rollouts. 
        # This function is called in the worker processes only
        self.pipeline.on_epoch_start()
        MpUtil.barrier()