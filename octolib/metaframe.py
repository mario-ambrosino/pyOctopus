# System Libraries
import json
import os
import re
import time
import uuid

# Third-Party Libraries
import pandas as pd

# Third-Party Libraries TODO: exclude param import.
import octolib.shared as shared
from octolib.utils import get_directory_structure


class MetaFrame:
    """
    Class containing MetaFrame methods - it enables to access in a viable way all the elements in the dataset folder
    without knowing which is the structure, i.e. acts on the metadata of IVM files. The general rule for this
    class it define for each expected function of pyOctopus an insulated function which acts literaly as a main()
    function.

    TODO: transform this class in a Singleton Class
    """

    @staticmethod
    def get_metadata(data_folder_path):
        """
        Given a data_path, it extracts frame-data contained into it and construct a pandas DataFrame which contains
        every information about it. It should follow the convention chosen in the first release of datasets from IVM

        Parameters
        ----------
        data_folder_path: string
            Folder path for data.

        Returns
        -------
        metaframe: Pandas.DataFrame
            The dataframe which holds frame-data about datasets contained in data_path folder.

        """
        directory_structure = get_directory_structure(data_folder_path)

        file_list = []
        # Access recursively the directory tree
        for dataset in directory_structure.keys():
            for direction in directory_structure[dataset].keys():
                for train in directory_structure[dataset][direction].keys():
                    for speed_category in directory_structure[dataset][direction][train].keys():
                        for item in list(directory_structure[dataset][direction][train][speed_category].keys()):
                            meta_data = item.rsplit("_")
                            if meta_data[0] == "Rettilineo":
                                # Load dataset feature from filename (meta_data object)
                                # Directional Component
                                component = meta_data[1]
                                # Number of trip with the same features (e.g. speed class, engine status etc.)
                                num_trip = meta_data[2][-1]
                                # Engine Status Feature
                                engine_specs = meta_data[2][:-1]
                                engine_status = "N/D"
                                if engine_specs == speed_category:
                                    engine_status = None
                                elif engine_specs == "F" + speed_category:
                                    engine_status = "F"
                                elif engine_specs == "FF" + speed_category:
                                    engine_status = "FF"

                                # Acceleration Matrix Path
                                accel_path = "{}/{}/{}/{}/Rettilineo_{}_{}{}_{}.txt".format(
                                    data_folder_path, direction, train, speed_category, component, engine_specs,
                                    num_trip,
                                    train)
                                # Count rows in Acceleration file
                                num_accel = sum(1 for _ in open(accel_path))
                                # Speed Matrix Path
                                vel_path = "{}/{}/{}/Profili delle velocità/Vel_{}_{}{}.txt".format(
                                    data_folder_path, direction, train, train, engine_specs, num_trip)
                                # Count rows in Speed file
                                num_vel = sum(1 for _ in open(vel_path))
                                # Check if there is discrepancy in the acceleration and speed files.
                                vel_acc_discrepancy = num_accel - num_vel
                                # TODO IMPUTATION STRATEGY for speed when discrepancy occurs
                                # Scores for each bearings_labels (algorithm computation results)
                                score_path = "{}/{}/{}/{}/Scores_{}_{}{}_{}.txt".format(
                                    data_folder_path, direction, train, speed_category, component, engine_specs,
                                    num_trip,
                                    train)
                                # Unified Score (algorithm computation results)
                                uscore_path = "{}/{}/{}/{}/UnifiedScore_{}_{}{}_{}.txt".format(
                                    data_folder_path, direction, train, speed_category, component, engine_specs,
                                    num_trip,
                                    train)
                                # Ground Truth JSON file from IVM. It has the following structure:
                                #
                                # {
                                #     "Bearing_Columns" : [5,6,7,8,9,10,11,12],
                                #     "Reference_Bearing" : 12,
                                #     "Weldings" : [15217, 32697, 50208, 67956, 85588, 103145, 120809, 138461],
                                #     "Wheel_Fault" : True
                                # }
                                #
                                gt_path = "{}/{}/{}/{}/GT_{}{}_{}.txt".format(
                                    data_folder_path, direction, train, speed_category, engine_specs, num_trip, train)

                                # Init column
                                accel_columns = None
                                reference_Bearing = None
                                weldings = None
                                is_wheel_defective = None

                                try:
                                    json_file = open(gt_path)
                                    gt_raw = json_file.read()
                                    # Clean file from human errors (i introduced too much special characters,
                                    # the parser was angry with me)
                                    gt_raw = re.sub(
                                        '\n', '', re.sub(
                                            '\t', '', re.sub(
                                                ' ', '', gt_raw
                                                ).strip()
                                            ).strip()
                                        ).strip()
                                    gt_raw = re.sub('False', '"False"', gt_raw)
                                    gt_raw = re.sub('True', '"True"', gt_raw)
                                    ground_truth = json.loads(gt_raw)
                                    reference_Bearing = ground_truth["Reference_Bearing"]
                                    accel_columns = str(ground_truth["Bearing_Columns"])
                                    weldings = str(ground_truth["Weldings"])
                                    is_wheel_defective = ground_truth["Wheel_Fault"]
                                    json_file.close()
                                except IOError:
                                    print("[{}] -".format(time.ctime()) + "File {} not accessible".format(gt_path))

                                # SCORE INCLUSION
                                N_shifts_path = "{}/{}/{}/{}/N_Shifts_{}_{}{}_{}.txt".format(
                                    data_folder_path, direction, train, speed_category, component, engine_specs,
                                    num_trip,
                                    train)
                                S_shifts_path = "{}/{}/{}/{}/S_Shifts_{}_{}{}_{}.txt".format(
                                    data_folder_path, direction, train, speed_category, component, engine_specs,
                                    num_trip,
                                    train)
                                N_shifts = None
                                S_shifts = None
                                try:
                                    N_shifts_file = open(N_shifts_path)
                                    S_shifts_file = open(S_shifts_path)

                                    N_shifts = N_shifts_file.read()
                                    N_shifts = re.sub(
                                        '\n', '', re.sub(
                                            '\t', '', re.sub(
                                                ' ', '', N_shifts
                                                ).strip()
                                            ).strip()
                                        ).strip()
                                    S_shifts = S_shifts_file.read()
                                    S_shifts = re.sub(
                                        '\n', '', re.sub(
                                            '\t', '', re.sub(
                                                ' ', '', S_shifts
                                                ).strip()
                                            ).strip()
                                        ).strip()
                                except IOError:
                                    # print("[{}] -".format(time.ctime()) + "Shift File not accessible")
                                    pass

                                processed_item = [uuid.uuid4(), dataset, direction, train, speed_category, component,
                                                  num_trip,
                                                  engine_status, accel_path, vel_path, score_path, uscore_path,
                                                  (engine_status == "0") or (engine_status == "2"),
                                                  (engine_status == "0") or (engine_status == "1"), num_accel, num_vel,
                                                  vel_acc_discrepancy, reference_Bearing, accel_columns, weldings,
                                                  is_wheel_defective, N_shifts, S_shifts]
                                file_list.append(processed_item)

        column_names = ["ID", "Dataset", "Direction", "Train", "Avg_Speed", "Component", "Num_Trip", "Engine_Status",
                        "Accel_Path", "Vel_Path", "Scores_Path", "UnifiedScore_Path", "is_ABU-B_enabled",
                        "is_ABU-A_enabled", "N_acc", "N_vel", "N_discrepance", "Reference_Bearing", "Bearing_Columns",
                        "Weldings", "Wheel_Fault", "N_Shifts", "S_Shifts"]

        return pd.DataFrame(file_list, columns = column_names)

    def export_metaframe(self, name):
        """
        Access recursively to all the content of the dataset folder and extracts DataFrame from each of the subfolders,
        merging after all the frame-frame in a unique one.

        Parameters
        ----------
        name: str
            Name of the csv file to be exported.
        """
        list_meta = []
        for index, data_folder in enumerate(os.listdir(self.data)):
            print(index, data_folder)
            list_meta.append(self.get_metadata(data_folder_path = "{}/{}".format(self.data, data_folder)))
        meta_frame = pd.concat(list_meta)
        meta_frame.to_csv("public/export/{}.csv".format(name), index = False)

    def __init__(self, path_meta, path_data):
        """
        At execution of given modules, check if all works and prints credits, then loads the meta-frame.
        TODO: merge with setup.py and environment check.

        Parameters
        ----------
        path_meta: str
            Path string to meta file
        path_data: str
            Path string to data file

        """
        if shared.verbose_mode:
            print("[{}] # ".format(time.ctime()) + "# Loading PyOctopus Trip Module")
            print("# OCTOLIB_VERSION = {} - REFACTORED VERSION".format(shared.module_version))
            print("# LAST_MODIFIED = {}".format(shared.module_last_modified_data))
            print("# pyOctopus @ Octopus Project - MARIO AMBROSINO")
        self.data = path_data  #: Dataset Path
        self.path = path_meta  #: Metaframe Path
        # Metadata Dataframe Uploaddata
        try:
            self.frame = pd.read_csv(self.path)  #: pandas.DataFrame holding MetaFrame information
            self.columns = self.frame.columns   #: list of the meta-features
            self.UUID = self.frame["ID"].values  #: UUIDs in the metaframe
            self.num_datasets = len(self.frame.index)  #: Number of available datasets in the metaframe
        except FileNotFoundError:
            print("[{}] # ".format(time.ctime()) + "Warning - Meta-frame not found.")
            print("[{}] # ".format(time.ctime()) +
                  "Alternative Route -> Generate Meta-frame from {} folder".format(shared.DATASET_PATH))
            self.export_metaframe(name = "meta")
            self.frame = pd.read_csv(str(self.path))
        print("[{}] # ".format(time.ctime()) + "Init Completed.")

    def __call__(self, uid, value):
        """
        On call of the metaframe, give a selected value with a given UUID equal to "uid" and with a column equal to
        "value". Note: polimorphism of uid input: if one uses a UUID string it will return the corresponding object
        into the metaframe, while if you pass an integer as uid it will return the corresponding UUID indexed by the
        integer in the metaframe used.

        Parameters
        ----------

        uid: (str, int)
             a UUID identifier from the meta-frame;
        value: str
             the column identier belonging to self.columns;

        Returns
        -------

        meta-data:
             the value stored into the meta-frame.

        """
        if isinstance(uid, str) and isinstance(value, str):
            if uid in self.UUID and value in self.columns:
                # return column with the given "item" feature
                return self.frame.loc[self.UUID == uid][value].values[0]
        if isinstance(uid, int) and isinstance(value, str):
            if uid in range(self.num_datasets) and value in self.columns:
                return self.frame[value].iloc[uid]
        else:
            print(
                "[{}] # ".format(time.ctime()) +
                "# ERROR: Wrong access to meta-frame ID:{}. VALUE:{}".format(uid, value)
                )

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in self.columns:
                # return column with the given "item" feature
                return self.frame[item]
            else:
                print("[{}] # ".format(time.ctime()) + "# ERROR: Wrong attribute chosen: {}".format(item))
                pass  # TODO raise exception for wrong choice in column and adopt a proper behaviour
        if isinstance(item, int):
            if item in range(self.num_datasets):
                return self.frame.iloc[item]
            else:
                print("[{}] # ".format(time.ctime()) + "# ERROR: Out of Index".format(item))
                pass  # TODO raise exception for wrong choice in column and adopt a proper behaviour
