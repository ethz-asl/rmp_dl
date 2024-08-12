from typing import List


class ValidationDatasetNames:
    """
    Class containing the names of the validation datasets. Lifted out of validation_datasets.py to avoid circular imports.
    """
    dataset_names: List[str] = []

    @staticmethod
    def resolve_validation_dataset_name(idx: int) -> str:
        return ValidationDatasetNames.dataset_names[idx]
    
    @staticmethod
    def append(name: str):
        ValidationDatasetNames.dataset_names.append(name)