"""
PyOctopus.trip Module: Class container for OCTOPUS project.
"""

import time
# Warning Clean-up
import warnings

import numpy as np
# Data Manipulation Libraries
import pandas as pd
# Graphic Import
import plotly.graph_objects as go
# Fourier and Wavelet Libraries
import pywt
from plotly.subplots import make_subplots
from scipy.signal import correlate as corr

# System Methods
warnings.filterwarnings('ignore')

from . import shared_parameters as param
from sklearn.cluster import DBSCAN

meta_frame = None


def init():
    """
    At execution of given modules, check if all works and prints credits.
    # TODO: Load error handling for lack of requirements and automate setup init suggestion.
    :return: meta_df: a dataframe which contains all reference about our project.
    """
    if param.verbose_mode:
        print("[{}] # ".format(time.ctime()) + "# Loading PyOctopus Trip Module")
        print("# OCTOLIB_VERSION = {} - REFACTORED VERSION".format(param.module_version))
        print("# LAST_MODIFIED = {}".format(param.module_last_modified_data))
        print("# pyOctopus @ Octopus Project - MARIO AMBROSINO")

    # Metadata Dataframe Upload
    try:
        meta_df = pd.read_csv(str(param.META_PATH))
    except FileNotFoundError:
        print("[{}] # ".format(time.ctime()) + "Warning - Meta-frame not found.")
        print("[{}] # ".format(time.ctime()) +
              "Alternative Route -> Generate Meta-frame from {} folder".format(param.DATASET_PATH))
        from octolib.helper import generate_metaframe
        generate_metaframe()
        meta_df = pd.read_csv(str(param.META_PATH))
    print("[{}] # ".format(time.ctime()) + "Init Completed.")
    return meta_df


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def signal_sync(reference, signal):
    """
    Returns the optimal shift to maximize correlation
    between reference signal and the analyzed signal
    to be shifted.

    :param reference: Leading (static) signal
    :param signal: Signal to be shifted
    :return: the shift for signal
    """
    corr_data = corr(reference, signal)
    shift = np.argmax(corr_data)
    return shift - int(len(corr_data) / 2)


def naive_integration(data):
    """
    It simply make the cumulative sum of the data, performing a naive integration.
    In order to get better numerical result one could use three-point or five-point
    interpolation of the data or using spline approximation but at this point isn't necessary.
    TODO: find better integration strategy, i.e. from from Numerical Methods course.
    """
    return pd.DataFrame(np.cumsum(data)).to_numpy()


def uuid_to_value(uid=None, value=None, meta=None):
    """ Given UUID, gives the relative accel_path """
    return meta[meta["ID"] == uid][value].values[0]


def skiprow_logic(index, start, end, step=1):
    # please note: implicit skipping of header in dataset
    if index in range(start, end, step):
        return False
    else:
        return True


# UGLY WORKAROUND TO GET AWAY WITH DIFFERENT SEPARATORS
def separator_parser(dataset):
    if dataset == "Santa_Maria_a_Vico":
        return "\t"
    elif dataset == "Estratto_Rettilineo_AR":
        return " "


# Trip Class
class Octopus:
    class Trip:

        def extract_item(self, value):
            """
            Extract value from dataframe with fixed uuid
            :param value: column name of the value to extract
            :return: string file containing the value
            """
            return str(uuid_to_value(uid=self.uuid, value=value))

        def extract_list(self, value, sep=","):
            """
            Extract list from dataframe with fixed uuid
            :param value: column name of the value to extract
            :param sep: list separator
            :return: list extracted from metaframe
            """
            return [int(s) for s in (uuid_to_value(uid=self.uuid, value=value))[1:-1].split(sep)]

        def __init__(self, uid):
            """
            Trip Class, loads mostly metadata of a given dataset file
            :param uid: the unique identifier to select a specific dataset
            """
            # unique universal identifier for the given dataset
            self.uuid = str(uid)

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
            self.pos_zero = param.STARTING_POINT
            # Paths
            self.accel_path = self.extract_item("Accel_Path")
            self.scores_path = self.extract_item("Scores_Path")
            self.vel_path = self.extract_item("Vel_Path")
            self.num_samples = int(self.extract_item("N_acc"))

            if self.dataset == "Estratto_Rettilineo_AR":
                # Ground_Truth values by IVM
                self.weldings = np.array(self.extract_list(value="Weldings", sep=", "))
                self.bearing_columns = self.extract_list(value="Bearing_Columns", sep=", ")
                self.reference_bearing = param.reference_dict[self.train][self.direction]
                self.is_wheel_faulty = self.extract_item("Wheel_Fault")

    class Track(Trip):
        def __init__(self, uid, start=0, stop=-1, pos_zero=864):
            """
            Track Class, used to select a slice of a specific dataset.
            :param uid: the unique identifier to select a specific dataset
            :param start: start of the slice in the chosen dataset
            :param stop: ending of the slice in the chosen dataset
            """
            super().__init__(uid)

            # Track Delimiters
            self.start = int(start)
            if stop == -1:
                self.stop = self.num_samples
            else:
                self.stop = int(stop)
            # Kinematics property
            self.vel = self.get_speed()
            self.accel = self.get_acceleration()
            self.pos_zero = pos_zero
            self.position = self.get_avg_previous_pos() + naive_integration(self.vel / param.SAMPLING_FREQUENCY)
            self.time = np.linspace(start=start, stop=stop, num=int(self.num_samples / param.SAMPLING_FREQUENCY))

            # Permutation Map
            self.permutation_sensors = self.get_dict_direction()
            # Holds bearing labels after rearranging
            self.bearings_labels = {
                side: {key: value for (key, value) in self.permutation_sensors["I{}".format(side)].items()}
                for side in self.sides}

            # Shifts
            if param.MANUAL_SHIFTS:
                # Loads shifts obtained via numerical estimation.
                self.shifts = {side: param.shifts_dict[self.direction][self.avg_speed] for side in self.sides}
            else:
                # Loads shifts file contained into datasets folder.
                self.shifts = {side: self.extract_list("{}_Shifts".format(side), ',') for side in self.sides}

            # Total padding for the acceleration array to be rolled
            self.total_pad = {side: sum([np.abs(value) for value in self.shifts[side]]) for side in self.sides}

            # Position here is relative to the start of the track. pos_zero takes account of the reference system.

            self.positions = {side: {sensor: None for sensor in self.sensors} for side in self.sides}
            self.reference_sensor = self.permutation_sensors["S"][self.reference_bearing]
            self.weldings_holder = np.array(
                self.weldings + (self.shifts["S"][self.reference_sensor]) * np.ones(len(self.weldings))).astype(
                np.int32)
            self.shifted_weldings = {
                side: {
                    sensor: (self.weldings_holder + param.relative_ns_shift[side][self.direction][self.avg_speed])
                    for sensor in self.sensors
                    } for side in self.sides
                }
            # Echo of welding on the opposite rail
            self.weldings_echoes = {
                "N": self.shifted_weldings["S"],
                "S": self.shifted_weldings["N"]
                }
            self.scores = {side: {sensor: None for sensor in self.sensors} for side in self.sides}
            self.anomalies = {side: {sensor: None for sensor in self.sensors} for side in self.sides}
            self.wsd = {side: {sensor: None for sensor in self.sensors} for side in self.sides}
            self.thresholds = {side: {sensor: None for sensor in self.sensors} for side in self.sides}

            self.anomaly_cluster = {
                side: {
                    sensor: {
                        "N_labels": None, "N_noise": None, "Anomalies": [], "Avg_Index": [], "std": [],
                        "performance": [], "mask": {"echoes": None, "weldings": None},
                        "DB_OBJECT": None, "Error_X": None,
                        }
                    for sensor in self.sensors
                    } for side in self.sides
                }
            print("[{}][T:{}][D:{}][S:{}][N:{}][X:{}] # TRIP LOADED".format(
                time.ctime(), self.train, self.direction, self.avg_speed, self.num_trip, self.component)
                )
            # EXECUTE PREPROCESSING
            self.preprocess_accelerations()
            print("[{}] # ".format(time.ctime()) + "Acceleration Preprocessing Completed.")

        def get_acceleration(self):
            """
            Returns all the accelerometric signals read by dataset indexed by the metaframe
            :return: a NumPy array containing a slice of all sensor signals and without mean value
            """
            # print("Reading Acceleration Data from {}".format(self.accel_path))
            temp = pd.read_table(
                str(self.accel_path),
                header=None, sep="{}".format(separator_parser(self.dataset)),
                engine='python',
                skiprows=lambda x: skiprow_logic(x, self.start, self.stop))
            return (temp - temp.mean()).to_numpy()

        def preprocess_accelerations(self):
            # Acquire Acceleration from dataset
            accelerations = pd.read_table(
                str(self.accel_path),
                header=None, sep="{}".format(separator_parser(self.dataset)),
                engine='python',
                skiprows=lambda x: skiprow_logic(x, self.start, self.stop))
            # Zero-mean acceleration
            accelerations = (accelerations - accelerations.mean()).to_numpy()
            # Define Acceleration Data Structure via lists
            north_list = [0, 0, 0, 0]
            south_list = [0, 0, 0, 0]
            # Rearrange Accelerations:
            for key in self.bearing_columns:
                if key in [*self.permutation_sensors["N"]]:
                    target = self.permutation_sensors["N"][key]
                    # print("key: {} -> target: N{} for TRAIN {} DIRECTION {}".format(key, target, self.train,
                    #                                                                 self.direction))
                    north_list[target] = accelerations[:, key - 5]
                elif key in [*self.permutation_sensors["S"]]:
                    target = self.permutation_sensors["S"][key]
                    # print("key: {} -> target: S{}".format(key, target))
                    south_list[target] = accelerations[:, key - 5]
            # Clear Space
            del accelerations
            # Rewrite accelerations with the shifts
            accelerations = {
                "N": np.transpose(np.array(north_list)),
                "S": np.transpose(np.array(south_list))
                }

            # Allocate structures for the abs
            signals = {
                side:
                    {sensor: np.empty((self.num_samples + self.total_pad[side]))
                     for sensor in self.sensors
                     }
                for side in self.sides
                }
            positions = {
                side:
                    {sensor: np.empty((self.num_samples + self.total_pad[side]))
                     for sensor in self.sensors
                     }
                for side in self.sides
                }

            # Padding and rolling loops
            for index_side, side in enumerate(self.sides):
                for index_sensor, sensor in enumerate(self.bearings_labels[side]):
                    # Init Empty values
                    signals[side][index_sensor][:] = np.NaN
                    positions[side][index_sensor][:] = np.NaN
                    # Assign Existent Accelerations and Roll
                    signals[side][index_sensor][0:self.num_samples] = accelerations[side][:, index_sensor]
                    signals[side][index_sensor][:] = np.roll(signals[side][index_sensor],
                                                             self.shifts[side][index_sensor])
                    # Assign Existent Positions and Roll
                    positions[side][index_sensor][0:self.num_samples] = self.position[:, 0]
                    positions[side][index_sensor][:] = np.roll(positions[side][index_sensor],
                                                               self.shifts[side][index_sensor])

            self.positions = positions
            self.accel = signals

        def get_speed(self):
            """
            Returns all the signals
            :return: a numpy array containing a slice of all sensor signals
            """
            # print("Reading Speed Data from {}".format(self.vel_path))
            return pd.read_table(
                str(self.vel_path),
                header=None, sep=separator_parser(self.dataset),
                engine='python',
                skiprows=lambda x: skiprow_logic(x, self.start, self.stop)).to_numpy()

        def get_avg_previous_pos(self):
            """
            Returns the train position at the starting point
            # LAST MODIFIED: Added pos_zero
            :return: a numpy array containing a slice of all sensor signals
            """
            ret_value = 0
            if self.start != 0:
                mean_speed = float(
                    pd.read_table(
                        str(self.vel_path),
                        header=None,
                        engine='python',
                        skiprows=lambda x: skiprow_logic(x, 0, self.start)).mean()
                    )
                ret_value = self.pos_zero + mean_speed * self.start / param.SAMPLING_FREQUENCY
            return ret_value

        # noinspection DuplicatedCode
        def get_dict_direction(self):
            """
            Generates a permutation mapping for sensor displacement in order to define north and south sensors in an
            automatic way. At the moment the bearing columns are fixed e.g. self.bearing_columns = [5,6,7,8,9,10,11,12]
            :return: Dictionary with North, South and Inverse North, South mapping.
            """
            north = None
            south = None

            if self.train == "5":
                if self.direction == "Andata":
                    north = {(index if index % 2 == 1 else None): int(3 - (index - 5) / 2) for index in
                             self.bearing_columns}
                    south = {(index if index % 2 == 0 else None): int(3 - (index - 6) / 2) for index in
                             self.bearing_columns}
                elif self.direction == "Ritorno":
                    north = {(index if index % 2 == 1 else None): int(3 - (index - 5) / 2) for index in
                             self.bearing_columns}
                    south = {(index if index % 2 == 0 else None): int(3 - (index - 6) / 2) for index in
                             self.bearing_columns}
            elif self.train == "7":
                if self.direction == "Andata":
                    north = {(index if index % 2 == 0 else None): int(0 + (index - 6) / 2) for index in
                             self.bearing_columns}
                    south = {(index if index % 2 == 1 else None): int(0 + (index - 5) / 2) for index in
                             self.bearing_columns}
                elif self.direction == "Ritorno":
                    north = {(index if index % 2 == 1 else None): int(3 - (index - 5) / 2) for index in
                             self.bearing_columns}
                    south = {(index if index % 2 == 0 else None): int(3 - (index - 6) / 2) for index in
                             self.bearing_columns}

            # inverse dictionary
            i_south = {value: key for key, value in south.items()}
            i_north = {value: key for key, value in north.items()}

            return {"N": north, "IN": i_north, "S": south, "IS": i_south}

        # WAVELET MODULES
        def get_wsd(self, side="S", sensor=0):
            """
            Perform Wavelet Spectrum Density for a given column in a north_south decomposition of accelerations
            and then weights it with the defintion seen in Molodova et al. 2013.
            WARNING: High Memory usage. Use caution in storing it in memory
            :param sensor: integer from 0 to 3
            :param side: "N" or "S" - North or South side
            :return: Weighted WSD
            """
            print("[{}] # ".format(time.ctime()) + "Starting WSD generation module.")
            # Wavelet Coefficients Calculation

            signal = np.nan_to_num(self.accel[side][sensor], nan=0)
            wavelet_coefficients, _ = pywt.cwt(
                np.nan_to_num(signal),
                param.SCALES,
                param.WAVELET,
                sampling_period=1 / param.SAMPLING_FREQUENCY)

            weights = np.array(param.SCALES).reshape((len(param.SCALES), 1))
            WSD = np.sum(
                param.SCALE_JUMP * np.abs(wavelet_coefficients) ** 2 / (param.SAMPLING_FREQUENCY * weights),
                axis=0
                )
            self.wsd[side][sensor] = pd.Series(WSD).fillna(0)
            print("[{}] # ".format(time.ctime()) + "WSD generation module completed")

        def get_anomalies(self, side="S", sensor=0, threshold=param.Z_SCORE_THRESHOLD, in_place=True):
            """
            Perform Simple Anomaly detection with adaptive thresholding technique developed in November 2020.
            METHODS:
                - "Adan": Adaptive Threshold Bollinger's Band Anomaly Detector
                - "FiZ": Fixed-Thresholded Z-score Anomaly Detector
            :param in_place: if in_place = True -> write in object attributes
            :param side: train side
            :param sensor: index of the sensor
            :param threshold: multiplicative factor to control severity in threshold
            :return: if not in place anomalous_index and acceptance_band
            """
            # Performs partial spectrum analysis
            self.get_wsd(side=side, sensor=sensor)
            print("[{}] # ".format(time.ctime()) + "POI Discovery module started.")
            self.get_z_score(side=side, sensor=sensor, threshold=threshold, in_place=True)
            # Anomalous points are indexed with the condition of threshold on acceptance_band.
            acceptance_band = float(threshold * self.scores[side][sensor].std()) * np.ones_like(
                self.scores[side][sensor])
            anomalous_index = np.where(
                np.array(self.scores[side][sensor]) > acceptance_band
                )
            print("[{}] # ".format(time.ctime()) +
                  "POI Discovery module completed. POI found = {}".format(len(anomalous_index[0])))

            if in_place:
                self.anomalies[side][sensor] = anomalous_index[0]
                self.thresholds[side][sensor] = acceptance_band
            else:
                return anomalous_index

        def get_z_score(self, side="S", sensor: int = 0, threshold=param.Z_SCORE_THRESHOLD, in_place=True):
            """
            As alternative to Bollinger Band on SAWP score, try to use z-score here.
            :param in_place: if in_place -> modifies object
            :param side: train side
            :param sensor: index of the sensor
            :param threshold: multiplicative factor to control severity in threshold
            :return: z_score
            """
            print("[{}] # ".format(time.ctime()) + "Z-score generation module started.")
            mean = self.wsd[side][sensor].fillna(value=0).rolling(param.WINDOW,
                                                                  win_type=param.WINDOW_TYPE).mean().fillna(value=0)
            sigma = threshold * self.wsd[side][sensor].fillna(value=1).rolling(param.WINDOW,
                                                                               win_type=param.WINDOW_TYPE).std().fillna(
                value=1)
            z_score = ((self.wsd[side][sensor].fillna(value=0) - mean) / sigma).replace([np.inf, -np.inf],
                                                                                        np.nan).fillna(value=0)
            print("[{}] # ".format(time.ctime()) + "Z-score generation module completed.")
            if in_place:
                self.scores[side][sensor] = z_score
            else:
                return z_score

        def plot_accelerations(self):
            """
            Plots the unsorted acceleration signals for a given trip.
            :return: Nothing
            TODO REFACTOR AFTER DICT REFACTORING
            """
            fig = make_subplots(rows=8, cols=1,
                                shared_xaxes=True,
                                vertical_spacing=0
                                )
            for index_side, side in enumerate(self.sides):
                for index_sensor, sensor in enumerate(self.bearings_labels[side]):
                    # Odd Part

                    # Complete Shifted Signal
                    fig.add_trace(
                        go.Scattergl(
                            x=self.positions["S"][0],
                            y=self.accel[side][index_sensor],
                            mode='lines',
                            name='Signal_{}{}'.format(side, sensor),
                            line=dict(color='blue', width=1),
                            ),
                        row=index_sensor + 1 + 4 * index_side, col=1,
                        )
                    # Shifted Weldings
                    fig.add_trace(
                        go.Scattergl(
                            x=self.positions["S"][0][self.shifted_weldings[side][index_sensor]],
                            y=self.accel[side][index_sensor][self.shifted_weldings[side][index_sensor]],
                            mode='markers',
                            name='North Signal {}'.format(sensor),
                            line=dict(color='black', width=20),
                            ),
                        row=index_sensor + 1 + 4 * index_side, col=1,
                        )
            fig.update_layout(
                height=1024,
                width=2048,
                title_text="Vibration Collection: T:{} D:{} S:{} C:{} UUID:{}".format(
                    self.train, self.direction, self.avg_speed, self.component, self.uuid
                    ), showlegend=False)

            fig.write_image("images/Vibration_T_{}_D_{}_S_{}_C_{}_U{}.png".format(
                self.train, self.direction, self.avg_speed, self.component, self.uuid
                )
                )

        def plot_scores(self):

            fig = make_subplots(rows=4, cols=4,
                                shared_xaxes=True,
                                vertical_spacing=0
                                )
            for index_side, side in enumerate(self.sides):
                for index_sensor, sensor in enumerate(self.bearings_labels[side]):
                    # Odd Part

                    # Complete Shifted Signal
                    fig.add_trace(
                        go.Scattergl(
                            x=self.position,
                            y=self.accel[side][index_sensor],
                            mode='lines',
                            name='Signal_{}{}'.format(side, sensor),
                            line=dict(color='blue', width=1),
                            ),
                        row=index_sensor + 1, col=2 * index_side + 1,
                        )
                    # Shifted Weldings
                    fig.add_trace(
                        go.Scattergl(
                            x=self.position[self.shifted_weldings[side][index_sensor]],
                            y=self.accel[side][index_sensor][self.shifted_weldings[side][index_sensor]],
                            mode='markers',
                            name='North Signal {}'.format(sensor),
                            line=dict(color='black', width=20),
                            ),
                        row=index_sensor + 1, col=2 * index_side + 1,
                        )
                    # Anomalous Signals
                    fig.add_trace(
                        go.Scattergl(
                            x=self.position[self.anomalies[side][index_sensor]],
                            y=self.accel[side][index_sensor][self.anomalies[side][index_sensor]],
                            mode='markers',
                            name='North Signal {}'.format(sensor),
                            line=dict(color='red', width=2),
                            ),
                        row=index_sensor + 1, col=2 * index_side + 1,
                        )

                    # Even Part

                    # Scores
                    fig.add_trace(
                        go.Scattergl(
                            x=self.position,
                            y=self.scores[side][index_sensor],
                            mode='lines',
                            name='Scores_{}{}'.format(side, sensor),
                            line=dict(color='blue', width=1),
                            ),
                        row=index_sensor + 1, col=2 * index_side + 2,
                        )
                    # Shifted Weldings
                    fig.add_trace(
                        go.Scattergl(
                            x=self.position[self.shifted_weldings[side][index_sensor]],
                            y=self.scores[side][index_sensor][self.shifted_weldings[side][index_sensor]],
                            mode='markers',
                            name='North Signal {}'.format(sensor),
                            line=dict(color='black', width=20),
                            ),
                        row=index_sensor + 1, col=2 * index_side + 2,
                        )
                    fig.add_trace(
                        go.Scattergl(
                            x=self.position,
                            y=self.thresholds[side][index_sensor],
                            mode='lines',
                            name='North Signal {}'.format(sensor),
                            line=dict(color='red', width=1),
                            ),
                        row=index_sensor + 1, col=2 * index_side + 2,
                        )

            fig.update_layout(
                height=1024 * 2,
                width=2048 * 2,
                title_text="Scores for : T:{} D:{} S:{} C:{} UUID:{}".format(
                    self.train, self.direction, self.avg_speed, self.component, self.uuid
                    ), showlegend=False)
            fig.update_xaxes(range=[0, 300])
            fig.write_image("images/Scores_T_{}_D_{}_S_{}_C_{}_U{}.png".format(
                self.train, self.direction, self.avg_speed, self.component, self.uuid
                ))
            print("[{}] # ".format(time.ctime()) + "# Image Exported")

        # ML Methods
        # noinspection PyTypeChecker
        def get_anomaly_clusters(self,
                                 eps=param.CLUSTERING_EPS,
                                 min_samples=param.CLUSTERING_MS,
                                 metric=param.CLUSTERING_METRICS
                                 ):
            """
            Uses DBSCAN to cluster point in the anomaly dataset and stores them in an ad-hoc dictionary
            WARNING: To be performed after retrival of anomaly (self.get_anomalies)
            :param eps: Max Distance from the density cluster;
            :param min_samples: minimum number of points required to form a dense region;
            :param metric: DBSCAN distance metric;
            :return:
            """
            print("[{}] # ".format(time.ctime()) + "Starting DBSCAN clustering.")
            for side in self.sides:
                for sensor in self.sensors:
                    print("[{}] # ".format(time.ctime()) + "DBSCAN -> {}{} sensor".format(side, sensor))
                    # Performs DBSCAN
                    local_db = DBSCAN(
                        eps=eps,
                        min_samples=min_samples,
                        metric=metric
                        ).fit(
                        np.array(
                            self.anomalies[side][sensor]).reshape(-1, 1)
                        )
                    # Store DB_object
                    self.anomaly_cluster[side][sensor]["DB_OBJECT"] = local_db
                    # Count how many labels where found
                    # noinspection PyTypeChecker
                    self.anomaly_cluster[side][sensor]["N_labels"] = len(set(local_db.labels_)) - (
                        1 if -1 in local_db.labels_ else 0)
                    # Count outliers in anomaly detection
                    self.anomaly_cluster[side][sensor]["N_noise"] = list(local_db.labels_).count(-1)
                    print("[{}] # ".format(time.ctime()) + "DBSCAN -> Clusters: {}; Outliers: {};".format(
                        self.anomaly_cluster[side][sensor]["N_labels"],
                        self.anomaly_cluster[side][sensor]["N_noise"]))
                    # For each cluster, stores in the anomaly_cluster dict the corresponding anomaly index slice
                    for cluster in range(self.anomaly_cluster[side][sensor]["N_labels"]):
                        self.anomaly_cluster[side][sensor]["Anomalies"].append(
                            self.anomalies[side][sensor][
                                np.where(local_db.labels_ == cluster)
                            ]
                            )
                        # Cluster centroids
                        self.anomaly_cluster[side][sensor]["Avg_Index"].append(
                            int(
                                np.array(
                                    self.anomaly_cluster[side][sensor]["Anomalies"][cluster],
                                    dtype=np.int32
                                    ).mean()
                                )
                            )
                        # Standard Deviation calculation TODO: change to a proper metrics (example: half range width)
                        self.anomaly_cluster[side][sensor]["std"].append(
                            int(
                                np.array(
                                    self.anomaly_cluster[side][sensor]["Anomalies"][cluster],
                                    dtype=np.int32
                                    ).std()
                                )
                            )

            print("[{}] # ".format(time.ctime()) + "DBSCAN clustering Completed.")

        def evaluate_prediction(self, mode="Global", error_length=3):
            # OPTION ONE TODO use predict method for each of the dbscan predictor objects
            # OPTION TWO (actual) use a custom metric based on distance from weldings
            # WARNING: ASSUMING SHIFTING IS COMPLETED
            print("[{}] # ".format(time.ctime()) + "Starting performance evaluation. MODE: {}".format(mode))

            for side in self.sides:
                for sensor in self.sensors:
                    N_weldings = len(self.shifted_weldings[side][sensor])
                    N_prediction = len(self.anomaly_cluster[side][sensor]["Avg_Index"])
                    # Rail Weldings calculation:
                    x = np.array(self.shifted_weldings[side][sensor]).reshape((N_weldings, 1))
                    y = np.array(self.anomaly_cluster[side][sensor]["Avg_Index"])
                    distance_matrix = np.abs(x - y)
                    pred_distance = list(np.min(distance_matrix, axis=1))
                    mask_pred_weld = list(np.argmin(distance_matrix, axis=1).astype(np.int32))
                    pred_sigma = self.anomaly_cluster[side][sensor]["std"]
                    zip_pred = zip(pred_distance, pred_sigma)
                    # TODO less hard-thresholding function
                    # TODO define a proper criterion
                    pred_array = np.array(
                        [
                            1 if (distance / (error_length * param.samples_per_length[self.avg_speed]) < 1) else 0
                            for distance, sigma in zip_pred
                            ]
                        ).astype(np.int32)
                    prob_detection = np.sum(pred_array) / N_weldings

                    N_echoes = len(self.weldings_echoes[side][sensor])
                    # Rail Weldings calculation:
                    echo_x = np.array(self.weldings_echoes[side][sensor]).reshape((N_weldings, 1))
                    echo_matrix = np.abs(echo_x - y)
                    echo_distance = list(np.min(echo_matrix, axis=1))
                    mask_echo_weld = list(np.argmin(echo_matrix, axis=1).astype(np.int32))
                    echo_sigma = self.anomaly_cluster[side][sensor]["std"]
                    echo_pred = zip(echo_distance, echo_sigma)
                    # TODO less hard-thresholding function
                    # TODO define a proper criterion
                    echo_array = np.array(
                        [
                            1 if (distance / (error_length * param.samples_per_length[self.avg_speed]) < 1) else 0
                            for distance, sigma in echo_pred
                            ],
                        ).astype(np.int32)

                    echo_detection = np.sum(echo_array) / N_echoes
                    prob_false_alarm = (
                                               N_prediction - N_weldings * prob_detection - N_echoes * echo_detection) / N_prediction
                    self.anomaly_cluster[side][sensor]["performance"] = [prob_detection, echo_detection,
                                                                         prob_false_alarm]
                    print("[{}] # ".format(time.ctime()) + "Performance. PD: {:.2%}; ED: {:.2%}; PFA: {:.2%}".format(
                        prob_detection, echo_detection, prob_false_alarm))
                    self.anomaly_cluster[side][sensor]["mask"]["echoes"] = mask_echo_weld
                    self.anomaly_cluster[side][sensor]["mask"]["weldings"] = mask_pred_weld

        def plot_clusters(self, ):
            """
            Plots the cluster obtained
            :return: Nothing
            TODO REFACTOR AFTER DICT REFACTORING
            """
            fig = make_subplots(rows=8, cols=1,
                                shared_xaxes=True,
                                vertical_spacing=0
                                )
            # Plot Signals
            for index_side, side in enumerate(self.sides):
                for index_sensor, sensor in enumerate(self.bearings_labels[side]):
                    fig.add_trace(
                        go.Scattergl(
                            x=self.positions["S"][0],
                            y=self.accel[side][index_sensor],
                            mode='lines',
                            line=dict(color='black', width=1),
                            opacity=0.2
                            ),
                        row=index_sensor + 4 * index_side + 1, col=1,
                        )
                    fig.update_yaxes(title_text="Bearing {}{}".format(side, sensor),
                                     row=index_sensor + 4 * index_side + 1, col=1,
                                     showgrid=False, showticklabels=True)

            # Plot Clusters
            for index_side, side in enumerate(self.sides):
                for index_sensor, sensor in enumerate(self.bearings_labels[side]):
                    for index_cluster, cluster in enumerate(self.anomaly_cluster[side][index_sensor]["Anomalies"]):
                        # COLOR METHOD
                        if index_cluster in self.anomaly_cluster[side][index_sensor]["mask"]["weldings"]:
                            color = "blue"
                            opacity = 0.8
                        elif index_cluster in self.anomaly_cluster[side][index_sensor]["mask"]["echoes"]:
                            color = "green"
                            opacity = 0.6
                        else:
                            color = "red"
                            opacity = 0.5
                        fig.add_trace(
                            go.Scattergl(
                                x=self.positions["S"][0][cluster],
                                y=self.accel[side][index_sensor][cluster],
                                mode='lines',
                                opacity=opacity,
                                line=dict(width=3, color=color),
                                ),
                            row=index_sensor + 4 * index_side + 1, col=1,
                            )
                        fig.update_yaxes(title_text="{}{}".format(side, sensor),
                                         row=index_sensor + 4 * index_side + 1, col=1,
                                         showgrid=False, showticklabels=True)

            for index_side, side in enumerate(self.sides):
                for index_sensor, sensor in enumerate(self.bearings_labels[side]):
                    # Shifted Weldings
                    fig.add_trace(
                        go.Scattergl(
                            x=self.positions["S"][0][self.shifted_weldings[side][index_sensor]],
                            y=self.accel[side][index_sensor][self.shifted_weldings[side][index_sensor]],
                            mode='markers',
                            marker=dict(color='yellow', size=2),
                            error_x=dict(
                                type='data',  # value of error bar given in data coordinates
                                array=self.anomaly_cluster[side][index_sensor]["Error_X"] * np.ones_like(
                                    self.shifted_weldings[side][index_sensor]),
                                visible=True),
                            ),
                        row=index_sensor + 4 * index_side + 1, col=1,
                        )

                    # Shifted Echo Weldings
                    fig.add_trace(
                        go.Scattergl(
                            x=self.positions["S"][0][self.weldings_echoes[side][index_sensor]],
                            y=self.accel[side][index_sensor][self.weldings_echoes[side][index_sensor]],
                            mode='markers',
                            marker=dict(color='orange', size=2),
                            error_x=dict(
                                type='data',  # value of error bar given in data coordinates
                                array=self.anomaly_cluster[side][index_sensor]["Error_X"] * np.ones_like(
                                    self.shifted_weldings[side][index_sensor]),
                                visible=True),
                            ),
                        row=index_sensor + 4 * index_side + 1, col=1,
                        )

            fig.update_layout(
                height=1024,
                width=2048,
                title_text="Vibration Collection: T:{} D:{} S:{} C:{} UUID:{}".format(
                    self.train, self.direction, self.avg_speed, self.component, self.uuid
                    ), showlegend=False)

            fig.write_image("images/CVIB_T_{}_D_{}_S_{}_C_{}_U{}.png".format(
                self.train, self.direction, self.avg_speed, self.component, self.uuid
                )
                )
