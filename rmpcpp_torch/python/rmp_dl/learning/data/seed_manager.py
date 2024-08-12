class WorldgenSeedManager:
    FULL_TEST_START = 1e8
    TEST_START = 2E8
    VALIDATION_START = 3E8
    TRAIN_START = 4E8
    

    @staticmethod
    def get_seed(experiment_type: str, obstacles: int, i: int) -> int:
        if experiment_type == "full_test":
            start= WorldgenSeedManager.FULL_TEST_START
        elif experiment_type == "test":
            start = WorldgenSeedManager.TEST_START
        elif experiment_type == "validation":
            start = WorldgenSeedManager.VALIDATION_START
        elif experiment_type == "train":
            start = WorldgenSeedManager.TRAIN_START
        else:
            raise Exception("type must be one of full_test, test, validation, train")

        if i > 1e5:
            raise Exception("i must be less than 1e5")
        
        if obstacles > 5e2:
            raise Exception("obstacles must be less than 5e2")

        return int(start + obstacles * 1e5 + i)