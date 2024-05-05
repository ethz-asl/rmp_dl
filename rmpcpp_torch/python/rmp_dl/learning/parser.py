import argparse
import rmp_dl.util.io as rmp_io


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_title", type=str, help="Title of the run")
    parser.add_argument("--config_path", type=str, help="Location of the config file", 
                        default=rmp_io.resolve_config_directory("mini_config"))
    parser.add_argument("--train_dataset_config_name", type=str, help="Name of the training config file.", default="dagger")
    parser.add_argument("--validation_dataset_config_name", type=str, help="Name of the validation config file.", default="default")
    parser.add_argument("--model_config_name", type=str, help="Name of the model config file.", default="rnn")
    parser.add_argument("--num_workers", type=int, help="Number of workers for the rollout", default=1)
    parser.add_argument("--temporary_storage_path", type=str, help="Location to store temporary data.")
    parser.add_argument("--dataset_long_term_storage_path", type=str, help="Location where the data is stored for long term if a cache node with long term storage is used")
    parser.add_argument("--dataset_short_term_caching_path", type=str, help="Location where the data is stored for short term if a cache node is used.")
    parser.add_argument("--wandb_metadata_path", type=str, help="Location where wandb saves its metadata during a run", default=None)
    parser.add_argument("--logging_path", type=str, help="Location where the logging files are stored", default=None)
    parser.add_argument("--open3d_renderer_container_path_or_name", type=str, default="open3d-v16-image-renderer-final",
                        help="Name or path of the open3d renderer container. The default is open3d-v16-image-renderer-final, "
                        "which is the name of a docker container. "
                        "See the docker folder for instructions on how to build the container."
                        "If using singularity, pass the path to the container image file. The script docker/run_container_script.sh will automatically"
                        "determine whether to use docker or singularity based on what is INSTALLED; it will first try docker and then singularity,"
                        "so if you have both docker and singularity installed" 
                        "it will always try to use docker and fail. I'm assuming that in most cases the 2 are mutually exclusive, and if you do have both,"
                        "that you use docker")

    return parser

