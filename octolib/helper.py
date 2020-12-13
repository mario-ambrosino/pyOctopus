"""
Helper Routine: High-Level API to use pyOctopus functionality.
"""
import os

# Versioning
module_version = "0.3"
module_last_modified_data = "2020/11/27"
verbose_mode = True

# Path
ROOT_PATH = "/home/mario/WS/OCTOPUS/PyOctopus_v.{}".format(module_version)
DATASET_PATH = not os.path.join(ROOT_PATH, "datasets")
META_PATH = "export/meta.csv"

# Parameters
SAMPLING_FREQUENCY = 2000  # Hz

# Wavelet Parameters
MIN_SCALE = 1  # Minimum Scale
MAX_SCALE = 51  # Maximum Scale. WARNING! Choose it w.r.t. RAM footprint TODO method to choose MAX_SCALE properly
WINDOW = 1500  # Rolling Window Width for pandas moving average methods
SCALE_JUMP = 1  # Gap between scales
SCALES = list(range(MIN_SCALE, MAX_SCALE, SCALE_JUMP))  # Scales List
WAVELET = "cmor1-1"  # Complex Morlet with Central Frequency and Central Width equal to 1
WINDOW_TYPE = "hamming"  # Window-type for rolling and padding pandas algorithm.
DISCRETE_WAVELET = "db4"
SIGMA_DENOISING_FILTER = 0.2
FILTER_THRESHOLD = len(SCALES)  # actually uses full spectrum to ease calculation
SIGMA_TH = 2
Z_SCORE_THRESHOLD = 2
STARTING_POINT = 864  # [m] - Reference Position
MANUAL_SHIFTS = True
SHIFTED_ACCEL = True

# CLUSTERING
CLUSTERING_EPS = 10
CLUSTERING_MS = 10
CLUSTERING_METRICS = "cityblock"

relative_ns_shift = {
    "N": {
        "Andata": {
            "A": 3840,
            "B": 1920,
            "C": 1280,
            },
        "Ritorno": {
            "A": 3840,
            "B": 1920,
            "C": 1280,
            }
        },
    "S": {
        "Andata": {
            "A": 0,
            "B": 0,
            "C": 0,
            },
        "Ritorno": {
            "A": 0,
            "B": 0,
            "C": 0,
            }
        },
    }

shifts_dict = {
    "Ritorno": {
        "A": [4883, 3594, 1290, 0],
        "B": [2396, 1763, 633, 0],
        "C": [1642, 1208, 434, 0],
        },
    "Andata": {
        "A": [4883, 3594, 1290, 0],
        "B": [2396, 1763, 633, 0],
        "C": [1642, 1208, 434, 0],
        }
    }

reference_dict = {
    "5": {
        "Andata": 12,
        "Ritorno": 12,
        },
    "7": {
        "Andata": 5,
        "Ritorno": 5,
        }
    }

speed_dict = {
    "A": 15 / 3.6,  # 4.16 m/s
    "B": 30 / 3.6,  # 8.32 m/s
    "C": 45 / 3.6,  # 12.5 m/s
    }

samples_per_length = {
    "A": int(SAMPLING_FREQUENCY / speed_dict["A"]),  # 480 samples/m
    "B": int(SAMPLING_FREQUENCY / speed_dict["B"]),  # 240 samples/m
    "C": int(SAMPLING_FREQUENCY / speed_dict["C"]),  # 120 samples/m
    }

engine_dict = {
    "0": "",
    "1": "F",
    "2": "FF"
    }

import json
# System Methods
import re
import time
import uuid
# Warning Clean-up
import warnings
import zipfile
from functools import reduce

import numpy as np
# Data Manipulation Libraries
import pandas as pd
# Fourier and Wavelet Libraries
from numpy.fft import *

# Graphic Import
warnings.filterwarnings('ignore')

from octolib.shared_parameters import *
import octolib.trip as tp


def cross_correlation_using_fft(x, y):
    f1 = fft(x)
    f2 = fft(np.flipud(y))
    cc = np.real(ifft(f1 * f2))
    return fftshift(cc)


def get_directory_structure(input_path):
    """
    Creates a nested dictionary that represents the folder structure of input_path. Needed for the execution of the
    dataframe generator.

    Parameters
    ----------
    input_path: (string) The path to analyze

    Returns
    -------
    directory: (dict) The dictionary which maps the information about the folder

    """
    directory = {}
    input_path = input_path.rstrip(os.sep)
    start = input_path.rfind(os.sep) + 1
    for path, dirs, files in os.walk(input_path):
        folders = path[start:].split(os.sep)
        subdir = dict.fromkeys(files)
        parent = reduce(dict.get, folders[:-1], directory)
        parent[folders[-1]] = subdir
    return directory


def extract_data(input_path, output_path):
    """
    Given a zipped dataset, extract it in output_path.
    Parameters
    ----------
    input_path: string
        The path of the zip file to be extracted
    output_path: string
        The path of the target folder for the extraction

    Returns
    -------

    """
    with zipfile.ZipFile(input_path, 'r') as zip_ref:
        zip_ref.extractall(output_path)
        zip_ref.close()


def generate_metadata(data_path="datasets/Estratto_Rettilineo_AR"):
    """
    Given a data_path, it extracts meta-data contained into it and construct a pandas DataFrame which contains
    every information about it. It should follow the convention chosen in the first release of datasets from IVM
    Parameters
    ----------
    data_path: string
        The path of the root folder for the data.

    Returns
    -------
    metaframe: Pandas.DataFrame
        The dataframe which holds meta-data about datasets contained in data_path folder.

    """
    directory_structure = get_directory_structure(input_path=data_path)

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
                            # Load Path related to "Rettilineo" file, either if the corresponding files doesn't exist
                            # Acceleration Matrix Path
                            accel_path = "{}/{}/{}/{}/Rettilineo_{}_{}{}_{}.txt".format(
                                data_path, direction, train, speed_category, component, engine_specs, num_trip, train)
                            # Count rows in Acceleration file
                            num_accel = sum(1 for _ in open(accel_path))
                            # Speed Matrix Path
                            vel_path = "{}/{}/{}/Profili delle velocità/Vel_{}_{}{}.txt".format(
                                data_path, direction, train, train, engine_specs, num_trip)
                            # Count rows in Speed file
                            num_vel = sum(1 for _ in open(vel_path))
                            # Check if there is discrepancy in the acceleration and speed files.
                            vel_acc_discrepancy = num_accel - num_vel
                            # TODO IMPUTATION STRATEGY for speed when discrepancy occurs
                            # Scores for each bearings_labels (algorithm computation results)
                            score_path = "{}/{}/{}/{}/Scores_{}_{}{}_{}.txt".format(
                                data_path, direction, train, speed_category, component, engine_specs, num_trip, train)
                            # Unified Score (algorithm computation results)
                            uscore_path = "{}/{}/{}/{}/UnifiedScore_{}_{}{}_{}.txt".format(
                                data_path, direction, train, speed_category, component, engine_specs, num_trip, train)
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
                                data_path, direction, train, speed_category, engine_specs, num_trip, train)

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
                                data_path, direction, train, speed_category, component, engine_specs, num_trip, train)
                            S_shifts_path = "{}/{}/{}/{}/S_Shifts_{}_{}{}_{}.txt".format(
                                data_path, direction, train, speed_category, component, engine_specs, num_trip, train)
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
                                print("[{}] -".format(time.ctime()) + "Shift File not accessible")

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

    return pd.DataFrame(file_list, columns=column_names)


def generate_metaframe(folder="datasets", name="meta"):
    """
    Access recursively to all the content of the folder and extracts meta-frame from each of the subfolders,
    merging after all the meta-frame in a unique one.

    Parameters
    ----------
    folder: the root director
    name: string
        Name of the csv file to be exported.

    Returns
    -------

    """
    list_meta = []
    for index, data_folder in enumerate(os.listdir(folder)):
        print(index, data_folder)
        list_meta.append(generate_metadata(data_path="{}/{}".format(folder, data_folder)))
    meta_frame = pd.concat(list_meta)
    meta_frame.to_csv("export/{}.csv".format(name), index=False)


def test_init(ID: int = 23):
    """
    Simple Loader for test purpose. It helps to load quickly a dataset indexed by a simple integer ID in a meta-frame
    already loaded
    Parameters
    ----------
    ID: int
        Index in the meta-frame file into the tp.meta_frame attribute

    Returns
    -------
    a Octopus.Track object containing the data in the dataset labelled with ID

    """
    identifier = ID
    UUID = tp.meta_frame["ID"]
    return tp.Octopus.Track(UUID[identifier])


def generate_vibration_images():
    """
    Generate Vibration Images with aligned track for all the item in the meta-frame, with weldings in black.
    Returns
    -------

    """
    UUID = tp.meta_frame["ID"]
    for identifier, uid in enumerate(UUID):
        X = tp.Octopus.Track(UUID[identifier])
        print("[{}] # ".format(time.ctime()) + "Acceleration Preprocessing Completed.")
        print("-Data Load Completed.")
        X.plot_accelerations()
        # X.plot_unsorted_accelerations()


def test_alignment_score(threshold: float = 0.7,
                         detection_range: float = 0.5,
                         plot: bool = False,
                         export: bool = True,
                         epsilon: float = CLUSTERING_EPS,
                         min_samples_cluster: int = CLUSTERING_MS,
                         cluster_metrics: str = CLUSTERING_METRICS,
                         ):
    """
    Score Generator Routine API: It evaluates the correct alignment for the ground truth index for weldings and the
    shifting of the acceleration sensor. It follows this data pipeline:

    For each dataset:
     - take the acceleration from file,
     - shift them to align with the weldings,
     - perform SAWP score on each sensor track,
     - extract anomalous-SAWP index slices
     - cluster them with DBSCAN
     - check if the nearest cluster to a given weldings has a space distance which is lesser then detection_range
     - export the meta-data and score in a score file.

    Parameters
    ----------
    threshold : float
        multiplication factor for the decision boundary for anomaly score
    detection_range : float
        range (in meters) to assess whether a cluster correctly verifies the hypothesis to be a welding.
    plot : bool
        if True, it exports an image for each dataset
    export : bool
        if True, it exports the score file
    epsilon : float
        The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        This is not a maximum bound on the distances of points within a cluster.
        This is the most important DBSCAN parameter to choose appropriately for your data set and distance function.
    min_samples_cluster : int
        The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
        This includes the point itself.
    cluster_metrics : str
        The metric chosen for clustering in the set [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’]

    Returns
    -------

    """
    score_list = []
    UUID = tp.meta_frame["ID"]
    print("[{}] # ".format(time.ctime()) + "Score Generator Helper started.")
    for identifier, uid in enumerate(UUID):
        X = tp.Octopus.Track(UUID[identifier])
        for side in X.sides:
            for sensor in range(4):
                X.anomaly_cluster[side][sensor]["Error_X"] = detection_range
                # North&South Side anomaly detection & z-score
                print("[{}] # ".format(time.ctime()) + "Sensor:{}/4 - Side:{}".format(sensor + 1, side))
                X.get_anomalies(side=side, sensor=sensor, threshold=threshold)
        print("[{}] # ".format(time.ctime()) + "Anomaly Detection completed.")
        X.get_anomaly_clusters(eps=epsilon,
                               min_samples=min_samples_cluster,
                               metric=cluster_metrics)
        X.evaluate_prediction(error_length=detection_range)
        for side in X.sides:
            for sensor in range(4):
                score_list.append(
                    (uid, X.train, X.direction, X.avg_speed, X.num_trip, X.component, side, sensor,
                     X.anomaly_cluster[side][sensor]["Avg_Index"],
                     X.anomaly_cluster[side][sensor]["performance"][0],
                     X.anomaly_cluster[side][sensor]["performance"][1],
                     X.anomaly_cluster[side][sensor]["performance"][2],
                     X.anomaly_cluster[side][sensor]["Error_X"],
                     )
                    )
        X.plot_clusters()
    if plot:
        X.plot_scores()
        print("[{}] # ".format(time.ctime()) + "Scores Plot completed.")

    if export:
        # Generate Lists
        columns = ("UUID", "Train", "Direction", "Speed", "Num_Trip", "Component", "Side", "Sensor",
                   "Cluster Centroids", "PD", "ED", "PFA", "Error_X")
        export_df = pd.DataFrame(score_list, columns=columns)
        export_df.to_csv("export/scores_{}.csv".format(threshold))
    print("[{}] # ".format(time.ctime()) + "Score Generator Helper completed.")


def generate_shifts(export=True):
    dict_shifts = {}
    UUID = tp.meta_frame["ID"]
    for identifier, uid in enumerate(UUID):
        try:
            X = tp.Octopus.Track(UUID[identifier])
            print("-Data Load Completed.")
            N_shifts, S_shifts = X.shift_sync_signals()
            print("Shift Evaluated:\nN:{}\nS:{}".format(N_shifts, S_shifts))
            dict_shifts[uid] = [N_shifts, S_shifts]
        except IndexError:
            print("[{}] # ".format(time.ctime()) + "Index out of range - Error for track {}".format(UUID[identifier]))
    if export:
        out_file = open("shifts.txt", "w")
        out_file.write(str(dict_shifts))
        out_file.close()
    return dict_shifts
