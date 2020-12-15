"""
Name: Octopus Module
Description: contains classes which interacts directly with the file taken by IVM (acceleration files).
Author: Mario Ambrosino
Date: 15/12/2020
TODO: decouple from shared_parameters - develop a parameters data structure.
"""


# System Libraries
# Third-Party Libraries
import numpy as np
# Project Libraries
import octolib.shared as shared
from octolib.metaframe import MetaFrame


class Trip:
    def extract_item(self, value):
        """
        Extract value from dataframe with fixed uuid
        :param value: column name of the value to extract
        :return: string file containing the value
        """
        return str(self.meta(uid = self.uuid, value = value))

    def extract_list(self, value, sep = ","):
        """
        Extract list from dataframe with fixed uuid
        :param value: column name of the value to extract
        :param sep: list separator
        :return: list extracted from metaframe
        """
        return [int(s) for s in (self.meta(uid = self.uuid, value = value))[1:-1].split(sep)]

    def __init__(self, uid):
        """
        Trip Class, loads mostly metadata of a given dataset file
        :param uid: the unique identifier to select a specific dataset
        """
        # unique universal identifier for the given dataset
        self.uuid = str(uid)
        self.meta = MetaFrame(path_meta = shared.META_PATH, path_data = shared.DATASET_PATH)
        # Trip metadata
        self.dataset = self.extract_item("Dataset")
        self.train = self.extract_item("Train")
        self.direction = self.extract_item("Direction")
        self.sides = ["N", "S"]
        self.sensors = range(4)
        self.avg_speed = self.extract_item("Avg_Speed")
        self.engine_conf = self.extract_item("Engine_Status")
        self.component = self.extract_item("Component")
        self.num_trip = self.extract_item("Num_Trip")
        self.pos_zero = shared.STARTING_POINT
        # Paths
        self.accel_path = self.extract_item("Accel_Path")
        self.scores_path = self.extract_item("Scores_Path")
        self.vel_path = self.extract_item("Vel_Path")
        self.num_samples = int(self.extract_item("N_acc"))

        if self.dataset == "Estratto_Rettilineo_AR":
            # Ground_Truth values by IVM
            self.weldings = np.array(self.extract_list(value = "Weldings", sep = ", "))
            self.bearing_columns = self.extract_list(value = "Bearing_Columns", sep = ", ")
            self.reference_bearing = shared.reference_dict[self.train][self.direction]
            self.is_wheel_faulty = self.extract_item("Wheel_Fault")
