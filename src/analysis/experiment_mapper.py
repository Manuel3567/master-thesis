from datetime import datetime


class ExperimentMapper:

    @staticmethod
    def map_id_to_config(experiment_id: int):

        config = []

        if experiment_id  == 1:
            config = [
                {
                "selected_features": ["ws_10m_loc_mean", "ws_100m_loc_mean", "power_t-96"],
                "train_start": "2022-01-01",
                "train_end": "2022-03-31",
                "val_start": "2023-01-01",
                "val_end": "2023-03-31",
                "random_state": 42
                }
            ]

        elif experiment_id  == 2:
            config = [
                {
                "selected_features": ["ws_10m_loc_mean", "ws_100m_loc_mean", "power_t-96"],
                "train_start": "2022-04-01",
                "train_end": "2022-06-30",
                "val_start": "2023-04-01",
                "val_end": "2023-06-30",
                "random_state": 42
                }
            ]

        elif experiment_id  == 3:
            config = [
                {
                "selected_features": ["ws_10m_loc_mean", "ws_100m_loc_mean", "power_t-96"],
                "train_start": "2022-07-01",
                "train_end": "2022-09-30",
                "val_start": "2023-07-01",
                "val_end": "2023-09-30",
                "random_state": 42
                }
            ]
        elif experiment_id  == 4:
            config = [
                {
                "selected_features": ["ws_10m_loc_mean", "ws_100m_loc_mean", "power_t-96"],
                "train_start": "2022-10-01",
                "train_end": "2022-12-31",
                "val_start": "2023-10-01",
                "val_end": "2023-12-31",
                "random_state": 42
                }
            ]
        elif experiment_id  == 5:
            config = [
                {
                "selected_features": ["ws_10m_loc_mean", "ws_100m_loc_mean", "power_t-96"],
                "train_start": "2022-10-01",
                "train_end": "2022-12-31",
                "val_start": "2023-01-01",
                "val_end": "2023-03-31",
                "random_state": 42
                }
            ]
        elif experiment_id  == 6:
            config = [
                {
                "selected_features": ["ws_10m_loc_mean", "ws_100m_loc_mean", "power_t-96"],
                "train_start": "2022-10-01",
                "train_end": "2022-12-31",
                "val_start": "2023-04-01",
                "val_end": "2023-06-30",
                "random_state": 42
                }
            ]
        elif experiment_id  == 7:
            config = [
                {
                "selected_features": ["ws_10m_loc_mean", "ws_100m_loc_mean", "power_t-96"],
                "train_start": "2022-10-01",
                "train_end": "2022-12-31",
                "val_start": "2023-07-01",
                "val_end": "2023-09-30",
                "random_state": 42
                }
            ]
        elif experiment_id  == 8:
            config = [
                {
                "selected_features": ["ws_10m_loc_1", "ws_10m_loc_2", "ws_10m_loc_3", "ws_10m_loc_4", "ws_10m_loc_5", "ws_10m_loc_6", "ws_10m_loc_7", "ws_10m_loc_8", "ws_10m_loc_9", "ws_10m_loc_10", 
                                        "ws_100m_loc_1", "ws_100m_loc_2", "ws_100m_loc_3", "ws_100m_loc_4", "ws_100m_loc_5", "ws_100m_loc_6", "ws_100m_loc_7", "ws_100m_loc_8", "ws_100m_loc_9", "ws_100m_loc_10", "power_t-96"],
                "train_start": "2022-10-01",
                "train_end": "2022-12-31",
                "val_start": "2023-01-01",
                "val_end": "2023-06-30",
                "random_state": 42
                }
            ]
        elif experiment_id  == 9:
            config = [
                {
                "selected_features": ["ws_10m_loc_1", "ws_10m_loc_2", "ws_10m_loc_3", "ws_10m_loc_4", "ws_10m_loc_5", "ws_10m_loc_6", "ws_10m_loc_7", "ws_10m_loc_8", "ws_10m_loc_9", "ws_10m_loc_10", 
                                        "ws_100m_loc_1", "ws_100m_loc_2", "ws_100m_loc_3", "ws_100m_loc_4", "ws_100m_loc_5", "ws_100m_loc_6", "ws_100m_loc_7", "ws_100m_loc_8", "ws_100m_loc_9", "ws_100m_loc_10", "power_t-96"],
                "train_start": "2022-10-01",
                "train_end": "2022-12-31",
                "val_start": "2023-07-01",
                "val_end": "2023-12-31",
                "random_state": 42
                }
            ]
        elif experiment_id  == 10:
            config = [
                {
                "selected_features": ["ws_10m_loc_mean", "ws_100m_loc_mean", "ws_10m_loc_1", "ws_10m_loc_2", "ws_10m_loc_3", "ws_10m_loc_4", "ws_10m_loc_5", "ws_10m_loc_6", "ws_10m_loc_7", "ws_10m_loc_8", "ws_10m_loc_9", "ws_10m_loc_10", 
                                        "ws_100m_loc_1", "ws_100m_loc_2", "ws_100m_loc_3", "ws_100m_loc_4", "ws_100m_loc_5", "ws_100m_loc_6", "ws_100m_loc_7", "ws_100m_loc_8", "ws_100m_loc_9", "ws_100m_loc_10", "power_t-96"],
                "train_start": "2022-10-01",
                "train_end": "2022-12-31",
                "val_start": "2023-01-01",
                "val_end": "2023-06-30",
                "random_state": 42
                }
            ]
        elif experiment_id  == 11:
            config = [
                {
                "selected_features": ["ws_10m_loc_mean", "ws_100m_loc_mean", "ws_10m_loc_1", "ws_10m_loc_2", "ws_10m_loc_3", "ws_10m_loc_4", "ws_10m_loc_5", "ws_10m_loc_6", "ws_10m_loc_7", "ws_10m_loc_8", "ws_10m_loc_9", "ws_10m_loc_10", 
                                        "ws_100m_loc_1", "ws_100m_loc_2", "ws_100m_loc_3", "ws_100m_loc_4", "ws_100m_loc_5", "ws_100m_loc_6", "ws_100m_loc_7", "ws_100m_loc_8", "ws_100m_loc_9", "ws_100m_loc_10", "power_t-96"],
                "train_start": "2022-10-01",
                "train_end": "2022-12-31",
                "val_start": "2023-07-01",
                "val_end": "2023-12-31",
                "random_state": 42
                }
            ]
        elif experiment_id  == 12:
            config = [
                {
                "selected_features": ["ws_10m_loc_mean", "ws_100m_loc_mean", "ws_10m_loc_1", "ws_10m_loc_2", "ws_10m_loc_3", "ws_10m_loc_4", "ws_10m_loc_5", "ws_10m_loc_6", "ws_10m_loc_7", "ws_10m_loc_8", "ws_10m_loc_9", "ws_10m_loc_10", 
                                        "ws_100m_loc_1", "ws_100m_loc_2", "ws_100m_loc_3", "ws_100m_loc_4", "ws_100m_loc_5", "ws_100m_loc_6", "ws_100m_loc_7", "ws_100m_loc_8", "ws_100m_loc_9", "ws_100m_loc_10", "power_t-96", "interval_index"],
                "train_start": "2022-10-01",
                "train_end": "2022-12-31",
                "val_start": "2023-01-01",
                "val_end": "2023-06-30",
                "random_state": 42
                }
            ]
        elif experiment_id == 13:
            config = [
                {
                "selected_features": ["ws_10m_loc_mean", "ws_100m_loc_mean", "ws_10m_loc_1", "ws_10m_loc_2", "ws_10m_loc_3", "ws_10m_loc_4", "ws_10m_loc_5", "ws_10m_loc_6", "ws_10m_loc_7", "ws_10m_loc_8", "ws_10m_loc_9", "ws_10m_loc_10", 
                                        "ws_100m_loc_1", "ws_100m_loc_2", "ws_100m_loc_3", "ws_100m_loc_4", "ws_100m_loc_5", "ws_100m_loc_6", "ws_100m_loc_7", "ws_100m_loc_8", "ws_100m_loc_9", "ws_100m_loc_10", "power_t-96", "interval_index"],
                "train_start": "2022-10-01",
                "train_end": "2022-12-31",
                "val_start": "2023-07-01",
                "val_end": "2023-12-31",
                "random_state": 42
                }
            ]
        elif experiment_id == 14:
            config = [
                {
                "selected_features": ["ws_10m_loc_mean", "ws_100m_loc_mean","power_t-96"],
                "train_start": "2022-01-01",
                "train_end": "2022-03-31",
                "val_start": "2023-04-01",
                "val_end": "2023-06-30",
                "random_state": 42
                }
            ]
        elif experiment_id == 15:
            config = [
                {
                "selected_features": ["ws_10m_loc_mean", "ws_100m_loc_mean","power_t-96"],
                "train_start": "2022-07-01",
                "train_end": "2022-09-30",
                "val_start": "2023-04-01",
                "val_end": "2023-06-30",
                "random_state": 42
                }
            ]
        elif experiment_id == 16:
            config = [
                {
                "selected_features": ["ws_10m_loc_mean", "ws_100m_loc_mean","power_t-96"],
                "train_start": "2022-01-01",
                "train_end": "2022-03-31",
                "val_start": "2023-10-01",
                "val_end": "2023-12-31",
                "random_state": 42
                }
            ]
        elif experiment_id == 17:
            config = [
                {
                "selected_features": ["ws_10m_loc_mean", "ws_100m_loc_mean","power_t-96"],
                "train_start": "2022-01-01",
                "train_end": "2022-03-31",
                "val_start": "2023-07-01",
                "val_end": "2023-09-30",
                "random_state": 42
                }
            ]
        elif experiment_id == 18:
            config = [
                {
                "selected_features": ["ws_10m_loc_mean", "ws_100m_loc_mean","power_t-96"],
                "train_start": "2022-04-01",
                "train_end": "2022-06-30",
                "val_start": "2023-01-01",
                "val_end": "2023-03-31",
                "random_state": 42
                }
            ]
        elif experiment_id == 19:
            config = [
                {
                "selected_features": ["ws_10m_loc_mean", "ws_100m_loc_mean","power_t-96"],
                "train_start": "2022-04-01",
                "train_end": "2022-06-30",
                "val_start": "2023-07-01",
                "val_end": "2023-09-30",
                "random_state": 42
                }
            ]
        elif experiment_id == 20:
            config = [
                {
                "selected_features": ["ws_10m_loc_mean", "ws_100m_loc_mean","power_t-96"],
                "train_start": "2022-04-01",
                "train_end": "2022-06-30",
                "val_start": "2023-10-01",
                "val_end": "2023-12-31",
                "random_state": 42
                }
            ]
        elif experiment_id == 21:
            config = [
                {
                "selected_features": ["ws_10m_loc_mean", "ws_100m_loc_mean","power_t-96"],
                "train_start": "2022-07-01",
                "train_end": "2022-09-30",
                "val_start": "2023-01-01",
                "val_end": "2023-03-31",
                "random_state": 42
                }
            ]
        elif experiment_id == 22:
            config = [
                {
                "selected_features": ["ws_10m_loc_mean", "ws_100m_loc_mean","power_t-96"],
                "train_start": "2022-07-01",
                "train_end": "2022-09-30",
                "val_start": "2023-10-01",
                "val_end": "2023-12-31",
                "random_state": 42
                }
            ]
        elif experiment_id == 23:
            config = [
                {
                "selected_features": ["power_t-96"],
                "train_start": "2022-01-01",
                "train_end": "2022-03-31",
                "val_start": "2023-01-01",
                "val_end": "2023-03-31",
                "random_state": 42
                }
            ]
        
        elif experiment_id == 24:
            config = [
                {
                "selected_features": ["power_t-96"],
                "train_start": "2022-10-01",
                "train_end": "2022-12-31",
                "val_start": "2023-01-01",
                "val_end": "2023-12-31",
                "random_state": 42
                }
            ]

        elif experiment_id == 25:
            config = [
                {
                "selected_features": ["ws_10m_loc_mean", "ws_100m_loc_mean","power_t-96"],
                "train_start": "2022-10-01",
                "train_end": "2022-12-31",
                "val_start": "2023-01-01",
                "val_end": "2023-12-31",
                "random_state": 42
                }
            ]

        elif experiment_id == 26:
            config = [
                {
                "selected_features": ["ws_10m_loc_1", "ws_10m_loc_2", "ws_10m_loc_3", "ws_10m_loc_4", "ws_10m_loc_5", "ws_10m_loc_6", "ws_10m_loc_7", "ws_10m_loc_8", "ws_10m_loc_9", "ws_10m_loc_10", 
                                        "ws_100m_loc_1", "ws_100m_loc_2", "ws_100m_loc_3", "ws_100m_loc_4", "ws_100m_loc_5", "ws_100m_loc_6", "ws_100m_loc_7", "ws_100m_loc_8", "ws_100m_loc_9", "ws_100m_loc_10", "power_t-96"],
                "train_start": "2022-10-01",
                "train_end": "2022-12-31",
                "val_start": "2023-01-01",
                "val_end": "2023-12-31",
                "random_state": 42
                }
            ]
        elif experiment_id == 27:
            config = [
                {
                "selected_features": ["ws_10m_loc_mean", "ws_100m_loc_mean", "ws_10m_loc_1", "ws_10m_loc_2", "ws_10m_loc_3", "ws_10m_loc_4", "ws_10m_loc_5", "ws_10m_loc_6", "ws_10m_loc_7", "ws_10m_loc_8", "ws_10m_loc_9", "ws_10m_loc_10", 
                                        "ws_100m_loc_1", "ws_100m_loc_2", "ws_100m_loc_3", "ws_100m_loc_4", "ws_100m_loc_5", "ws_100m_loc_6", "ws_100m_loc_7", "ws_100m_loc_8", "ws_100m_loc_9", "ws_100m_loc_10", "power_t-96"],
                "train_start": "2022-10-01",
                "train_end": "2022-12-31",
                "val_start": "2023-01-01",
                "val_end": "2023-12-31",
                "random_state": 42
                }
            ]
        elif experiment_id == 28:
            config = [
                {
                "selected_features": ["ws_10m_loc_mean", "ws_100m_loc_mean", "ws_10m_loc_1", "ws_10m_loc_2", "ws_10m_loc_3", "ws_10m_loc_4", "ws_10m_loc_5", "ws_10m_loc_6", "ws_10m_loc_7", "ws_10m_loc_8", "ws_10m_loc_9", "ws_10m_loc_10", 
                                        "ws_100m_loc_1", "ws_100m_loc_2", "ws_100m_loc_3", "ws_100m_loc_4", "ws_100m_loc_5", "ws_100m_loc_6", "ws_100m_loc_7", "ws_100m_loc_8", "ws_100m_loc_9", "ws_100m_loc_10", "power_t-96", "interval_index"],
                "train_start": "2022-10-01",
                "train_end": "2022-12-31",
                "val_start": "2023-01-01",
                "val_end": "2023-12-31",
                "random_state": 42
                }
            ]
        elif experiment_id == 29:
            config = [
                {
                "selected_features": ["ws_10m_loc_mean", "ws_100m_loc_mean", "ws_10m_loc_1", "ws_10m_loc_2", "ws_10m_loc_3", "ws_10m_loc_4", "ws_10m_loc_5", "ws_10m_loc_6", "ws_10m_loc_7", "ws_10m_loc_8", "ws_10m_loc_9", "ws_10m_loc_10", 
                                        "ws_100m_loc_1", "ws_100m_loc_2", "ws_100m_loc_3", "ws_100m_loc_4", "ws_100m_loc_5", "ws_100m_loc_6", "ws_100m_loc_7", "ws_100m_loc_8", "ws_100m_loc_9", "ws_100m_loc_10", "power_t-96", "interval_index"],
                "train_start": "2016-01-01",
                "train_end": "2022-12-31",
                "val_start": "2023-01-01",
                "val_end": "2023-12-31",
                "random_state": 42
                }
            ]
        elif experiment_id == 30:
            config = [
                {
                "selected_features": ["ws_10m_loc_mean", "ws_100m_loc_mean", "ws_10m_loc_1", "ws_10m_loc_2", "ws_10m_loc_3", "ws_10m_loc_4", "ws_10m_loc_5", "ws_10m_loc_6", "ws_10m_loc_7", "ws_10m_loc_8", "ws_10m_loc_9", "ws_10m_loc_10", 
                                        "ws_100m_loc_1", "ws_100m_loc_2", "ws_100m_loc_3", "ws_100m_loc_4", "ws_100m_loc_5", "ws_100m_loc_6", "ws_100m_loc_7", "ws_100m_loc_8", "ws_100m_loc_9", "ws_100m_loc_10", "power_t-96"],
                "train_start": "2016-01-01",
                "train_end": "2022-12-31",
                "val_start": "2023-01-01",
                "val_end": "2023-12-31",
                "random_state": 42
                }
            ]
        elif experiment_id == 31:
            config = [
                {
                "selected_features": ["ws_10m_loc_1", "ws_10m_loc_2", "ws_10m_loc_3", "ws_10m_loc_4", "ws_10m_loc_5", "ws_10m_loc_6", "ws_10m_loc_7", "ws_10m_loc_8", "ws_10m_loc_9", "ws_10m_loc_10", 
                                        "ws_100m_loc_1", "ws_100m_loc_2", "ws_100m_loc_3", "ws_100m_loc_4", "ws_100m_loc_5", "ws_100m_loc_6", "ws_100m_loc_7", "ws_100m_loc_8", "ws_100m_loc_9", "ws_100m_loc_10", "power_t-96"],
                "train_start": "2016-01-01",
                "train_end": "2022-12-31",
                "val_start": "2023-01-01",
                "val_end": "2023-12-31",
                "random_state": 42
                }
            ]
        elif experiment_id == 32:
            config = [
                {
                "selected_features": ["ws_10m_loc_mean", "ws_100m_loc_mean","power_t-96"],
                "train_start": "2016-01-01",
                "train_end": "2022-12-31",
                "val_start": "2023-01-01",
                "val_end": "2023-12-31",
                "random_state": 42
                }
            ]
        elif experiment_id == 33:
            config = [
                {
                "selected_features": ["power_t-96"],
                "train_start": "2016-01-01",
                "train_end": "2022-12-31",
                "val_start": "2023-01-01",
                "val_end": "2023-12-31",
                "random_state": 42
                }
            ]
        elif experiment_id == 34:
            config = [
                {
                "selected_features": ["ws_10m_loc_mean", "ws_100m_loc_mean","power_t-96"],
                "train_start": "2022-09-01",
                "train_end": "2022-12-31",
                "val_start": "2023-01-01",
                "val_end": "2023-03-31",
                "random_state": 42
                }
            ]
        elif experiment_id == 35:
            config = [
                {
                "selected_features": ["ws_10m_loc_mean", "ws_100m_loc_mean","power_t-96"],
                "train_start": "2022-08-01",
                "train_end": "2022-12-31",
                "val_start": "2023-01-01",
                "val_end": "2023-03-31",
                "random_state": 42
                }
            ]
        elif experiment_id == 36:
            config = [
                {
                "selected_features": ["ws_10m_loc_mean", "ws_100m_loc_mean","power_t-96"],
                "train_start": "2022-07-01",
                "train_end": "2022-12-31",
                "val_start": "2023-01-01",
                "val_end": "2023-03-31",
                "random_state": 42
                }
            ]
        elif experiment_id == 37:
            config = [
                {
                "selected_features": ["ws_10m_loc_mean", "ws_100m_loc_mean","power_t-96"],
                "train_start": "2022-01-01",
                "train_end": "2022-12-31",
                "val_start": "2023-01-01",
                "val_end": "2023-03-31",
                "random_state": 42
                }
            ]
        elif experiment_id == 38:
            config = [
                {
                "selected_features": ["ws_10m_loc_mean", "ws_100m_loc_mean", "ws_10m_loc_1", "ws_10m_loc_2", "ws_10m_loc_3", "ws_10m_loc_4", "ws_10m_loc_5", "ws_10m_loc_6", "ws_10m_loc_7", "ws_10m_loc_8", "ws_10m_loc_9", "ws_10m_loc_10", 
                                        "ws_100m_loc_1", "ws_100m_loc_2", "ws_100m_loc_3", "ws_100m_loc_4", "ws_100m_loc_5", "ws_100m_loc_6", "ws_100m_loc_7", "ws_100m_loc_8", "ws_100m_loc_9", "ws_100m_loc_10", "power_t-96", "interval_index"],
                "train_start": "2022-01-01",
                "train_end": "2022-12-31",
                "val_start": "2023-01-01",
                "val_end": "2023-03-31",
                "random_state": 42
                }
            ]
        elif experiment_id == 39:
            config = [
                {
                "selected_features": ["ws_10m_loc_mean", "ws_100m_loc_mean", "ws_10m_loc_1", "ws_10m_loc_2", "ws_10m_loc_3", "ws_10m_loc_4", "ws_10m_loc_5", "ws_10m_loc_6", "ws_10m_loc_7", "ws_10m_loc_8", "ws_10m_loc_9", "ws_10m_loc_10", 
                                        "ws_100m_loc_1", "ws_100m_loc_2", "ws_100m_loc_3", "ws_100m_loc_4", "ws_100m_loc_5", "ws_100m_loc_6", "ws_100m_loc_7", "ws_100m_loc_8", "ws_100m_loc_9", "ws_100m_loc_10", "power_t-96", "interval_index"],
                "train_start": "2022-01-01",
                "train_end": "2022-12-31",
                "val_start": "2023-04-01",
                "val_end": "2023-06-30",
                "random_state": 42
                }
            ]
        elif experiment_id == 40:
            config = [
                {
                "selected_features": ["ws_10m_loc_mean", "ws_100m_loc_mean", "ws_10m_loc_1", "ws_10m_loc_2", "ws_10m_loc_3", "ws_10m_loc_4", "ws_10m_loc_5", "ws_10m_loc_6", "ws_10m_loc_7", "ws_10m_loc_8", "ws_10m_loc_9", "ws_10m_loc_10", 
                                        "ws_100m_loc_1", "ws_100m_loc_2", "ws_100m_loc_3", "ws_100m_loc_4", "ws_100m_loc_5", "ws_100m_loc_6", "ws_100m_loc_7", "ws_100m_loc_8", "ws_100m_loc_9", "ws_100m_loc_10", "power_t-96", "interval_index"],
                "train_start": "2022-01-01",
                "train_end": "2022-12-31",
                "val_start": "2023-07-01",
                "val_end": "2023-09-30",
                "random_state": 42
                }
            ]
        elif experiment_id == 41:
            config = [
                {
                "selected_features": ["ws_10m_loc_mean", "ws_100m_loc_mean", "ws_10m_loc_1", "ws_10m_loc_2", "ws_10m_loc_3", "ws_10m_loc_4", "ws_10m_loc_5", "ws_10m_loc_6", "ws_10m_loc_7", "ws_10m_loc_8", "ws_10m_loc_9", "ws_10m_loc_10", 
                                        "ws_100m_loc_1", "ws_100m_loc_2", "ws_100m_loc_3", "ws_100m_loc_4", "ws_100m_loc_5", "ws_100m_loc_6", "ws_100m_loc_7", "ws_100m_loc_8", "ws_100m_loc_9", "ws_100m_loc_10", "power_t-96", "interval_index"],
                "train_start": "2022-01-01",
                "train_end": "2022-12-31",
                "val_start": "2023-10-01",
                "val_end": "2023-12-31",
                "random_state": 42
                }
            ]
        elif experiment_id == 42:
            config = [
                {
                "selected_features": ["ws_10m_loc_mean", "ws_100m_loc_mean","power_t-96"],
                "train_start": "2022-01-01",
                "train_end": "2022-12-31",
                "val_start": "2023-01-01",
                "val_end": "2023-03-31",
                "random_state": 42
                }
            ]
        elif experiment_id == 43:
            config = [
                {
                "selected_features": ["ws_10m_loc_mean", "ws_100m_loc_mean","power_t-96"],
                "train_start": "2022-01-01",
                "train_end": "2022-12-31",
                "val_start": "2023-04-01",
                "val_end": "2023-06-30",
                "random_state": 42
                }
            ]
        elif experiment_id == 44:
            config = [
                {
                "selected_features": ["ws_10m_loc_mean", "ws_100m_loc_mean","power_t-96"],
                "train_start": "2022-01-01",
                "train_end": "2022-12-31",
                "val_start": "2023-07-01",
                "val_end": "2023-09-30",
                "random_state": 42
                }
            ]
        elif experiment_id == 45:
            config = [
                {
                "selected_features": ["ws_10m_loc_mean", "ws_100m_loc_mean","power_t-96"],
                "train_start": "2022-01-01",
                "train_end": "2022-12-31",
                "val_start": "2023-10-01",
                "val_end": "2023-12-31",
                "random_state": 42
                }
            ]

        elif experiment_id == 46:
            config = [
                {
                "selected_features": ["ws_10m_loc_1", "ws_10m_loc_2", "ws_10m_loc_3", "ws_10m_loc_4", "ws_10m_loc_5", "ws_10m_loc_6", "ws_10m_loc_7", "ws_10m_loc_8", "ws_10m_loc_9", "ws_10m_loc_10", 
                                        "ws_100m_loc_1", "ws_100m_loc_2", "ws_100m_loc_3", "ws_100m_loc_4", "ws_100m_loc_5", "ws_100m_loc_6", "ws_100m_loc_7", "ws_100m_loc_8", "ws_100m_loc_9", "ws_100m_loc_10", "power_t-96"],
                "train_start": "2022-01-01",
                "train_end": "2022-12-31",
                "val_start": "2023-01-01",
                "val_end": "2023-03-31",
                "random_state": 42
                }
            ]
        elif experiment_id == 47:
            config = [
                {
                "selected_features": ["ws_10m_loc_1", "ws_10m_loc_2", "ws_10m_loc_3", "ws_10m_loc_4", "ws_10m_loc_5", "ws_10m_loc_6", "ws_10m_loc_7", "ws_10m_loc_8", "ws_10m_loc_9", "ws_10m_loc_10", 
                                        "ws_100m_loc_1", "ws_100m_loc_2", "ws_100m_loc_3", "ws_100m_loc_4", "ws_100m_loc_5", "ws_100m_loc_6", "ws_100m_loc_7", "ws_100m_loc_8", "ws_100m_loc_9", "ws_100m_loc_10", "power_t-96"],
                "train_start": "2022-01-01",
                "train_end": "2022-12-31",
                "val_start": "2023-04-01",
                "val_end": "2023-06-30",
                "random_state": 42
                }
            ]
        elif experiment_id == 48:
            config = [
                {
                "selected_features": ["ws_10m_loc_1", "ws_10m_loc_2", "ws_10m_loc_3", "ws_10m_loc_4", "ws_10m_loc_5", "ws_10m_loc_6", "ws_10m_loc_7", "ws_10m_loc_8", "ws_10m_loc_9", "ws_10m_loc_10", 
                                        "ws_100m_loc_1", "ws_100m_loc_2", "ws_100m_loc_3", "ws_100m_loc_4", "ws_100m_loc_5", "ws_100m_loc_6", "ws_100m_loc_7", "ws_100m_loc_8", "ws_100m_loc_9", "ws_100m_loc_10", "power_t-96"],
                "train_start": "2022-01-01",
                "train_end": "2022-12-31",
                "val_start": "2023-07-01",
                "val_end": "2023-09-30",
                "random_state": 42
                }
            ]
        elif experiment_id == 49:
            config = [
                {
                "selected_features": ["ws_10m_loc_1", "ws_10m_loc_2", "ws_10m_loc_3", "ws_10m_loc_4", "ws_10m_loc_5", "ws_10m_loc_6", "ws_10m_loc_7", "ws_10m_loc_8", "ws_10m_loc_9", "ws_10m_loc_10", 
                                        "ws_100m_loc_1", "ws_100m_loc_2", "ws_100m_loc_3", "ws_100m_loc_4", "ws_100m_loc_5", "ws_100m_loc_6", "ws_100m_loc_7", "ws_100m_loc_8", "ws_100m_loc_9", "ws_100m_loc_10", "power_t-96"],
                "train_start": "2022-01-01",
                "train_end": "2022-12-31",
                "val_start": "2023-10-01",
                "val_end": "2023-12-31",
                "random_state": 42
                }
            ]
        elif experiment_id == 50:
            config = [
                {
                "selected_features": ["ws_10m_loc_mean", "ws_100m_loc_mean", "ws_10m_loc_1", "ws_10m_loc_2", "ws_10m_loc_3", "ws_10m_loc_4", "ws_10m_loc_5", "ws_10m_loc_6", "ws_10m_loc_7", "ws_10m_loc_8", "ws_10m_loc_9", "ws_10m_loc_10", 
                                        "ws_100m_loc_1", "ws_100m_loc_2", "ws_100m_loc_3", "ws_100m_loc_4", "ws_100m_loc_5", "ws_100m_loc_6", "ws_100m_loc_7", "ws_100m_loc_8", "ws_100m_loc_9", "ws_100m_loc_10", "power_t-96"],
                "train_start": "2022-01-01",
                "train_end": "2022-12-31",
                "val_start": "2023-01-01",
                "val_end": "2023-03-31",
                "random_state": 42
                }
            ]
        elif experiment_id == 51:
            config = [
                {
                "selected_features": ["ws_10m_loc_mean", "ws_100m_loc_mean", "ws_10m_loc_1", "ws_10m_loc_2", "ws_10m_loc_3", "ws_10m_loc_4", "ws_10m_loc_5", "ws_10m_loc_6", "ws_10m_loc_7", "ws_10m_loc_8", "ws_10m_loc_9", "ws_10m_loc_10", 
                                        "ws_100m_loc_1", "ws_100m_loc_2", "ws_100m_loc_3", "ws_100m_loc_4", "ws_100m_loc_5", "ws_100m_loc_6", "ws_100m_loc_7", "ws_100m_loc_8", "ws_100m_loc_9", "ws_100m_loc_10", "power_t-96"],
                "train_start": "2022-01-01",
                "train_end": "2022-12-31",
                "val_start": "2023-04-01",
                "val_end": "2023-06-30",
                "random_state": 42
                }
            ]
        elif experiment_id == 52:
            config = [
                {
                "selected_features": ["ws_10m_loc_mean", "ws_100m_loc_mean", "ws_10m_loc_1", "ws_10m_loc_2", "ws_10m_loc_3", "ws_10m_loc_4", "ws_10m_loc_5", "ws_10m_loc_6", "ws_10m_loc_7", "ws_10m_loc_8", "ws_10m_loc_9", "ws_10m_loc_10", 
                                        "ws_100m_loc_1", "ws_100m_loc_2", "ws_100m_loc_3", "ws_100m_loc_4", "ws_100m_loc_5", "ws_100m_loc_6", "ws_100m_loc_7", "ws_100m_loc_8", "ws_100m_loc_9", "ws_100m_loc_10", "power_t-96"],
                "train_start": "2022-01-01",
                "train_end": "2022-12-31",
                "val_start": "2023-07-01",
                "val_end": "2023-09-30",
                "random_state": 42
                }
            ]
        elif experiment_id == 53:
            config = [
                {
                "selected_features": ["ws_10m_loc_mean", "ws_100m_loc_mean", "ws_10m_loc_1", "ws_10m_loc_2", "ws_10m_loc_3", "ws_10m_loc_4", "ws_10m_loc_5", "ws_10m_loc_6", "ws_10m_loc_7", "ws_10m_loc_8", "ws_10m_loc_9", "ws_10m_loc_10", 
                                        "ws_100m_loc_1", "ws_100m_loc_2", "ws_100m_loc_3", "ws_100m_loc_4", "ws_100m_loc_5", "ws_100m_loc_6", "ws_100m_loc_7", "ws_100m_loc_8", "ws_100m_loc_9", "ws_100m_loc_10", "power_t-96"],
                "train_start": "2022-01-01",
                "train_end": "2022-12-31",
                "val_start": "2023-10-01",
                "val_end": "2023-12-31",
                "random_state": 42
                }
            ]
        elif experiment_id == 54:
            config = [
                {
                "selected_features": ["power_t-96"],
                "train_start": "2022-01-01",
                "train_end": "2022-12-31",
                "val_start": "2023-01-01",
                "val_end": "2023-03-31",
                "random_state": 42
                }
            ]
        elif experiment_id == 55:
            config = [
                {
                "selected_features": ["power_t-96"],
                "train_start": "2022-01-01",
                "train_end": "2022-12-31",
                "val_start": "2023-04-01",
                "val_end": "2023-06-30",
                "random_state": 42
                }
            ]
        elif experiment_id == 56:
            config = [
                {
                "selected_features": ["power_t-96"],
                "train_start": "2022-01-01",
                "train_end": "2022-12-31",
                "val_start": "2023-07-01",
                "val_end": "2023-09-30",
                "random_state": 42
                }
            ]
        elif experiment_id == 57:
            config = [
                {
                "selected_features": ["power_t-96"],
                "train_start": "2022-01-01",
                "train_end": "2022-12-31",
                "val_start": "2023-10-01",
                "val_end": "2023-12-31",
                "random_state": 42
                }
            ]
        elif experiment_id == 58:
            config = [
                {
                "selected_features": ["power_t-96"],
                "train_start": "2022-10-01",
                "train_end": "2022-12-31",
                "val_start": "2023-01-01",
                "val_end": "2023-06-30",
                "random_state": 42
                }
            ]
        elif experiment_id == 59:
            config = [
                {
                "selected_features": ["power_t-96"],
                "train_start": "2022-10-01",
                "train_end": "2022-12-31",
                "val_start": "2023-07-01",
                "val_end": "2023-12-31",
                "random_state": 42
                }
            ]
        else:
            raise ValueError(f"Experiment ID {experiment_id} is not valid.")
        
        return config
    from datetime import datetime

    @staticmethod
    def get_experiment_ids_for_time_range(time_range: str):
        # A method to filter experiment ids based on the time range.
        # We're only interested in experiments that belong to Q4 2022

        valid_ids = []
        for experiment_id in range(1, 30):  # Assuming there are 29 experiments
            config = ExperimentMapper.map_id_to_config(experiment_id)
            for item in config:
                if time_range in item["train_start"] or time_range in item["val_start"]:
                    valid_ids.append(experiment_id)
                    break
        return valid_ids
    

    @staticmethod
    def extract_date_abbreviations_from_config(config):
        """
        Processes training and validation date ranges into readable formats:
        - Qx YEAR for quarters
        - H1/H2 YEAR for half-years
        - FY YEAR for full year
        - YEAR-YEAR if the period spans multiple years
        """

        def months_between(start_date_str, end_date_str):
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
            months_diff = (end_date.year - start_date.year) * 12 + end_date.month - start_date.month
            if end_date.day >= start_date.day:
                months_diff += 1
            return months_diff, start_date.year, end_date.year

        def get_period(months_diff, start_date_str, start_year, end_year):
            if start_year != end_year:
                return f"{start_year}-{end_year}"

            start_month = datetime.strptime(start_date_str, "%Y-%m-%d").month

            if months_diff == 3:
                quarter = (start_month - 1) // 3 + 1
                return f"Q{quarter} {start_year}"
            elif months_diff == 6:
                return f"H1 {start_year}" if start_month <= 6 else f"H2 {start_year}"
            elif months_diff == 12:
                return f"FY {start_year}"
            else:
                return f"{start_year}-{end_year}"

        # Extract dates
        train_start = config[0]["train_start"]
        train_end = config[0]["train_end"]
        val_start = config[0]["val_start"]
        val_end = config[0]["val_end"]

        # Compute month diffs and periods
        train_month_diff, train_start_year, train_end_year = months_between(train_start, train_end)
        val_month_diff, val_start_year, val_end_year = months_between(val_start, val_end)

        train_period = get_period(train_month_diff, train_start, train_start_year, train_end_year)
        val_period = get_period(val_month_diff, val_start, val_start_year, val_end_year)

        return f"{train_period} / {val_period}"

    @staticmethod
    def get_feature_string_from_selected_features(config):

        selected_features = config[0]["selected_features"]
        
        if isinstance(selected_features, str):
            selected_features = [selected_features]  # Convert string to list

        # Define the conditions and return the corresponding string
        if set(selected_features) == {"power_t-96", "ws_10m_loc_mean", "ws_100m_loc_mean"}:
            return "power, mean ws"
        elif set(selected_features) == {"power_t-96"}:
            return "power"
        elif set(selected_features) == {"power_t-96", "ws_10m_loc_1", "ws_10m_loc_2", "ws_10m_loc_3", "ws_10m_loc_4", "ws_10m_loc_5", "ws_10m_loc_6", "ws_10m_loc_7", "ws_10m_loc_8", "ws_10m_loc_9", "ws_10m_loc_10", 
                                        "ws_100m_loc_1", "ws_100m_loc_2", "ws_100m_loc_3", "ws_100m_loc_4", "ws_100m_loc_5", "ws_100m_loc_6", "ws_100m_loc_7", "ws_100m_loc_8", "ws_100m_loc_9", "ws_100m_loc_10"}:
            return "power, ws at 10 loc"
        elif set(selected_features) == {"power_t-96", "ws_10m_loc_mean", "ws_100m_loc_mean", "ws_10m_loc_1", "ws_10m_loc_2", "ws_10m_loc_3", "ws_10m_loc_4", "ws_10m_loc_5", "ws_10m_loc_6", "ws_10m_loc_7", "ws_10m_loc_8", "ws_10m_loc_9", "ws_10m_loc_10", 
                                        "ws_100m_loc_1", "ws_100m_loc_2", "ws_100m_loc_3", "ws_100m_loc_4", "ws_100m_loc_5", "ws_100m_loc_6", "ws_100m_loc_7", "ws_100m_loc_8", "ws_100m_loc_9", "ws_100m_loc_10"}:
            return "power, all ws"
        elif set(selected_features) == {"power_t-96", "ws_10m_loc_mean", "ws_100m_loc_mean", "ws_10m_loc_1", "ws_10m_loc_2", "ws_10m_loc_3", "ws_10m_loc_4", "ws_10m_loc_5", "ws_10m_loc_6", "ws_10m_loc_7", "ws_10m_loc_8", "ws_10m_loc_9", "ws_10m_loc_10", 
                                        "ws_100m_loc_1", "ws_100m_loc_2", "ws_100m_loc_3", "ws_100m_loc_4", "ws_100m_loc_5", "ws_100m_loc_6", "ws_100m_loc_7", "ws_100m_loc_8", "ws_100m_loc_9", "ws_100m_loc_10", "interval_index"}:
            return "power, all ws, time bin"
        else:
            return "Unknown features"