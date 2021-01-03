# Sys
import os
import time
import json

# 3-rd part libraries
import numpy as np

# Project Libraries
import octolib.shared as shared
import octolib.track as track
from fastdtw import fastdtw
import pywt
from plotly.express import imshow
from plotly import graph_objects as go
from plotly.subplots import make_subplots


class Cluster(track.Track):
    """
    Creates, generates and handles cluster objects.
    """
    def load_clusterfile(self, side, sensor):
        """
        Loads cluster file, if available, generated from the last execution of
        :meth:`octolib.helper.export_clusters()` helper method

        Parameters
        ----------
        side: str = {"N","S"}
            North or South Side
        sensor: int = {0,1,2,3}
            Sensor ID

        """
        # import label_file:
        labels = None
        label_path = f"export/Clusters/{self.direction}/00{self.train}/{self.avg_speed}/Labels_" \
                     f"{self.uuid}_{self.train}_{self.direction}_{self.avg_speed}_{self.component}" \
                     f"_{self.num_trip}_{self.engine_conf}_{side}_{sensor}.json"
        if os.path.isfile(label_path):
            print(f"[{time.ctime()}] # Cluster Labels FILE FOUND in {label_path}")
            # if label_path is a file, import relative labels
            label_file = open(label_path, "r")
            label_json = json.load(label_file)
            label_file.close()
            labels = list(map(int, label_json["labels"][1:-1].split(",")))
        else:
            print(f"[{time.ctime()}] # Cluster FILE NOT FOUND in {label_path}")
        return labels

    def __init__(self, uid, radius: int = shared.CLUSTER_RADIUS):
        """
        Anomaly Cluster Init module. It loads all the anomalies for each side and sensor on super attributes,
        then locally generates a dictionary for slices of index centered in anomaly detected by the algorithm.
        TODO: include

        Parameters
        ----------

        uid: the unique identifier to select a specific dataset;
        radius: the radius (in samples unit) from the center of the point of interest.

        """
        super().__init__(uid)
        self.get_all_anomalies()
        self.clusters = {
            side:
                {
                    sensor:
                        {
                            index: {
                                "Center": center,
                                "Range":  radius,
                                "Slice":  (slice(
                                    max(0, center - radius),
                                    min(self.num_samples, center + radius)
                                    ) if abs(
                                    min(self.num_samples, center + radius) - max(0, center - radius)) == 2 * radius
                                           else None)} for (index, center) in
                            enumerate(self.anomaly_cluster[side][sensor][
                                          "Avg_Index"])
                            } for sensor in self.sensors
                    } for side in self.sides
            }
        self.clean_clusters()

    def __call__(self, side: str, sensor: int, cluster, variable: str = "Accelerations"):
        """
        On call, the class Cluster instance takes from self.accel the slice centered on the cluster for a given side
        and sensor index.

        Parameters
        ----------

        side: int
            The side ("N" or "S") of the train;
        sensor: int
            The sensor, indexed by advancement order of the train (first index is the first sensor which sees an
            anomaly);
        cluster: int or str
            If int, the cluster index. If str, the key of the dictionary.

        Returns
        -------

        accel_slice: np.array
            The array slice for acceleration centered around the point of interest.

        """
        if variable == "Accelerations":
            if isinstance(side, str) and isinstance(sensor, int) and isinstance(cluster, int):
                if (side in self.sides) and (sensor in self.sensors) and (
                        cluster in range(len(self.anomaly_cluster[side][sensor]["Avg_Index"]))):
                    return self.accel[side][sensor][self.clusters[side][sensor][cluster]["Slice"]]
        elif variable == "Positions":
            if isinstance(side, str) and isinstance(sensor, int) and isinstance(cluster, int):
                if (side in self.sides) and (sensor in self.sensors) and (
                        cluster in range(len(self.anomaly_cluster[side][sensor]["Avg_Index"]))):
                    return self.positions["S"][0][self.clusters[side][sensor][cluster]["Slice"]]
                else:
                    print("[{}] # ".format(time.ctime()) + "ERROR: Outside valid interval - DEBUG: {}{}-{} ".format(
                        side, sensor, cluster)
                          )

    def clean_clusters(self):
        """
        Clean the cluster from null slices.
        """
        for side in self.sides:
            for sensor in self.sensors:
                for (index, _) in enumerate(self.anomaly_cluster[side][sensor]["Avg_Index"]):
                    if self.clusters[side][sensor][index]["Slice"] is None:
                        self.clusters[side][sensor].pop(index)

    def get_all_anomalies(self,
                          threshold: float = shared.SIGMA_TH,
                          detection_range: float = shared.GROUND_WINDOW_DER,
                          epsilon: float = shared.CLUSTERING_EPS,
                          min_samples_cluster: int = shared.CLUSTERING_MS,
                          cluster_metrics: str = shared.CLUSTERING_METRICS,
                          ):
        """
        Takes all the anomaly clusters for a given dataset

        Parameters
        ----------

        threshold : float
            multiplication factor for the decision boundary for anomaly score
        detection_range : float
            range (in meters) to assess whether a cluster correctly verifies the hypothesis to be a welding.
        epsilon : float
            The maximum distance between two samples for one to be considered as in the neighborhood of the other.
            This is not a maximum bound on the distances of points within a cluster.
            This is the most important DBSCAN parameter to choose appropriately for your data set and distance function.
        min_samples_cluster : int
            The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
            This includes the point itself.
        cluster_metrics : str
            The metric chosen for clustering in the set [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’]

        """
        print("[{}] # ".format(time.ctime()) + "Init All Anomalies Calculation for clustering")
        for side in self.sides:
            for sensor in range(4):
                self.anomaly_cluster[side][sensor]["Error_X"] = detection_range
                # North&South Side anomaly detection & z-score
                print("[{}] # ".format(time.ctime()) + "Sensor:{}/4 - Side:{}".format(sensor + 1, side))
                self.get_anomalies(side = side, sensor = sensor, threshold = threshold)
        print("[{}] # ".format(time.ctime()) + "Anomaly Detection completed.")
        self.get_anomaly_clusters(eps = epsilon,
                                  min_samples = min_samples_cluster,
                                  metric = cluster_metrics)
        self.evaluate_prediction(error_length = detection_range)

    def dtw_dist(self, side, sensor, x_index, y_index, mode: str = "direct"):
        """
        Calculate distance between two clusters x and y  using DTW

        Parameters
        ----------
        side: str = {"N","S"}
            Side of the train
        sensor: int = {0,1,2,3}
            Index for the sensor
        x_index: int
            Index for cluster x
        y_index: int
            Index for cluster y
        mode: str
            Mode of execution of the distance.

        Returns
        -------

        dtw_distance: float
            the DTW distance.

        """
        dtw_distance = None
        # Raw Signal DTW distance
        if mode == "direct":
            dtw_distance, _ = fastdtw(
                x = self(side = side, sensor = sensor, cluster = x_index),
                y = self(side = side, sensor = sensor, cluster = y_index),
                )
        # Wavelet SAWP score DTW distance
        elif mode == "sawp":
            dtw_distance, _ = fastdtw(
                x = self.get_cwt(side = side, sensor = sensor, index = x_index, mode = "sawp"),
                y = self.get_cwt(side = side, sensor = sensor, index = y_index, mode = "sawp"),
                )
        return dtw_distance

    def plot_all_clusters(self):
        for side in self.sides:
            for sensor in self.sensors:
                for index in range(len(self.clusters[side][sensor])):
                    wave = self.get_cwt(side = side, sensor = sensor, index = index, mode = "wsd")
                    fig = imshow(wave)
                    fig.write_image("images/WSD_clusters/WSD_T{}_D{}_S{}_C{}_L{}_A{}_CLU{}.png".format(
                        self.train, self.direction, self.avg_speed, self.component, side, sensor, index)
                        )
                    print("Written image in 'images/WSD_clusters/WSD_T{}_D{}_S{}_C{}_L{}_A{}_CLU{}.png'".format(
                        self.train, self.direction, self.avg_speed, self.component, side, sensor, index)
                        )

    def export_all_clusters(self):
        export = {
            "Train":       self.train,
            "Direction":   self.direction,
            "Avg_Speed":   self.avg_speed,
            "Component":   self.component,
            "Engine_Conf": self.engine_conf,
            }
        for side in self.sides:
            export_side = {}
            for sensor in self.sensors:
                export_sensor = {}
                labels = self.load_clusterfile(side, sensor)
                for index in range(len(self.clusters[side][sensor])):
                    try:
                        export_index = {
                            "Cluster":       index,
                            "isWelding":     index in self.anomaly_cluster[side][sensor]["mask"]["weldings"],
                            "isEcho":        index in self.anomaly_cluster[side][sensor]["mask"]["echoes"],
                            "Mean_Position": float(
                                (self.positions["S"][0][self.clusters[side][sensor][index]["Slice"]]).mean()),
                            "Positions":     list(
                                self.positions["S"][0][self.clusters[side][sensor][index]["Slice"]]),
                            "Accelerations": list(
                                self.accel[side][sensor][self.clusters[side][sensor][index]["Slice"]]),
                            "Label":         labels[index] if labels is not None else None
                            }
                        export_index = {index: export_index}
                        export_sensor = {**export_sensor, **export_index}
                    except KeyError:
                        print(f"[{time.ctime()}] # Key Error: {index} - Manual Bypass")
                export_sensor = {sensor: export_sensor}
                export_side = {**export_side, **export_sensor}
            export_side = {side: export_side}
            export = {**export, **export_side}
            file_path = f"export/Clusters/{self.direction}/00{self.train}/{self.avg_speed}/Cluster_" \
                        f"{self.uuid}_{self.train}_{self.direction}_{self.avg_speed}_{self.component}" \
                        f"_{self.num_trip}_{self.engine_conf}.json"
            fp = open(file_path, "w")
            fp.write(json.dumps(export, indent = 3))
            fp.close()

    def dist_matrix(self, side, sensor, mode = "direct"):
        """
        Dynamic Time Warp Distance Matrix between elements of a clustering class.

        Parameters
        ----------

        side: str
            train side
        sensor: int
            sensor id
        mode: str
            distance calculation execution

        Returns
        -------

        dtw_dist_matric: np.array(shape(n_signals,n_signals))
            a matrix with the distances betwee

        """

        n_signals = len(self.clusters[side][sensor].keys())
        dtw_dist_matrix = np.empty((n_signals, n_signals))
        time_step = 0
        for x_index in range(0, n_signals):
            for y_index in range(0, x_index):
                dtw_dist_matrix[x_index, y_index] = self.dtw_dist(side, sensor, x_index, y_index, mode = mode)
                print(f"[{time.ctime()}] # " + f"DTW:{time_step}/{int(n_signals * (n_signals + 1) / 2)} - "
                                               f"Distance({x_index},{y_index}) = {dtw_dist_matrix[x_index, y_index]}")
                dtw_dist_matrix[y_index, x_index] = dtw_dist_matrix[x_index, y_index]
                time_step = time_step + 1
        return dtw_dist_matrix

    def get_cwt(self, side, sensor, index,
                scales = shared.SCALES,
                wavelet = shared.WAVELET,
                samp_period = 1 / shared.SAMPLING_FREQUENCY, mode = "coeff"):
        """
        Returns Continuous Wavelet Transform. Wrapper for pywt function

        Parameters
        ----------

        side: str = {"N","S"}
            Side of the train
        sensor: int = {0,1,2,3}
            Index for the sensor
        index: int
            Index for number of cluster
        scales:
            Number of scales for the pywt.cwt
        wavelet:
            Wavelet selected for the pywt.cwt
        samp_period:
            Sampling Period for the pywt.cwt
        mode: str = {"coeff", "wsd", "sawp"}
            Output format.

        Returns
        -------

        wavelet_coeff: np.array
            2-dim array containing the complex values of the wavelet function calculated for the signal.
        wave_freq: np.array
            1-dim array containing the frequency corresponding to scales.

        """
        wavelet_coeff, wave_freq = pywt.cwt(
            data = np.nan_to_num(self(side = side, sensor = sensor, cluster = index)),
            scales = scales,
            wavelet = wavelet,
            sampling_period = samp_period
            )
        if mode == "coeff":
            return wavelet_coeff
        if mode == "wsd":
            return np.abs(wavelet_coeff ** 2)
        if mode == "sawp":
            weights = np.array(shared.SCALES).reshape((len(shared.SCALES), 1))
            return np.sum(
                shared.SCALE_JUMP * np.abs(wavelet_coeff ** 2) / (shared.SAMPLING_FREQUENCY * weights),
                axis = 0
                )

    def plot_labeled_clusters(self):
        """
        Plots the cluster obtained
        """
        fig = make_subplots(rows = 8, cols = 1,
                            shared_xaxes = True,
                            vertical_spacing = 0
                            )
        # Plot Signals
        for index_side, side in enumerate(self.sides):
            for index_sensor, sensor in enumerate(self.bearings_labels[side]):
                fig.add_trace(
                    go.Scattergl(
                        x = self.positions["S"][0],
                        y = self.accel[side][index_sensor],
                        mode = 'lines',
                        line = dict(color = 'black', width = 1),
                        opacity = 0.2
                        ),
                    row = index_sensor + 4 * index_side + 1, col = 1,
                    )
                fig.update_yaxes(title_text = "Bearing {}{}".format(side, sensor),
                                 row = index_sensor + 4 * index_side + 1, col = 1,
                                 showgrid = False, showticklabels = True)

        # Plot Clusters
        for index_side, side in enumerate(self.sides):
            for index_sensor, sensor in enumerate(self.bearings_labels[side]):
                labels = self.load_clusterfile(side, index_sensor)

                for index_cluster, cluster in enumerate(
                        self.anomaly_cluster[side][index_sensor]["Anomalies"]):
                    if index_cluster < len(labels):
                        # DASH METHOD
                        if index_cluster in self.anomaly_cluster[side][index_sensor]["mask"]["weldings"]:
                            dash = "dot"
                            opacity = 0.8
                        elif index_cluster in self.anomaly_cluster[side][index_sensor]["mask"]["echoes"]:
                            dash = "dash"
                            opacity = 0.6
                        else:
                            dash = "solid"
                            opacity = 0.5
                        # COLOR METHOD
                        if labels[index_cluster] == 0:
                            color = "red"
                        elif labels[index_cluster] == 1:
                            color = "blue"
                        else:
                            color = "purple"

                        fig.add_trace(
                            go.Scattergl(
                                x = self.positions["S"][0][cluster],
                                y = self.accel[side][index_sensor][cluster],
                                mode = 'lines',
                                opacity = opacity,
                                line = dict(width = 3, color = color, dash = dash),
                                ),
                            row = index_sensor + 4 * index_side + 1, col = 1,
                            )
                        fig.update_yaxes(title_text = "{}{}".format(side, sensor),
                                         row = index_sensor + 4 * index_side + 1, col = 1,
                                         showgrid = False, showticklabels = True)

        for index_side, side in enumerate(self.sides):
            for index_sensor, sensor in enumerate(self.bearings_labels[side]):
                # Shifted Weldings
                fig.add_trace(
                    go.Scattergl(
                        x = self.positions["S"][0][self.shifted_weldings[side][index_sensor]],
                        y = self.accel[side][index_sensor][self.shifted_weldings[side][index_sensor]],
                        mode = 'markers',
                        marker = dict(color = 'yellow', size = 2),
                        error_x = dict(
                            type = 'data',  # value of error bar given in data coordinates
                            array = self.anomaly_cluster[side][index_sensor]["Error_X"] * np.ones_like(
                                self.shifted_weldings[side][index_sensor]),
                            visible = True),
                        ),
                    row = index_sensor + 4 * index_side + 1, col = 1,
                    )

                # Shifted Echo Weldings
                fig.add_trace(
                    go.Scattergl(
                        x = self.positions["S"][0][self.weldings_echoes[side][index_sensor]],
                        y = self.accel[side][index_sensor][self.weldings_echoes[side][index_sensor]],
                        mode = 'markers',
                        marker = dict(color = 'orange', size = 2),
                        error_x = dict(
                            type = 'data',  # value of error bar given in data coordinates
                            array = self.anomaly_cluster[side][index_sensor]["Error_X"] * np.ones_like(
                                self.shifted_weldings[side][index_sensor]),
                            visible = True),
                        ),
                    row = index_sensor + 4 * index_side + 1, col = 1,
                    )

        fig.update_layout(
            height = 1024,
            width = 2048,
            title_text = "Labeled Vibrations for T:{} D:{} S:{} C:{} UUID:{}".format(
                self.train, self.direction, self.avg_speed, self.component, self.uuid
                ), showlegend = False)

        fig.write_image("images/Labeled_Vibrations/Labeled_Vibrations_T_{}_D_{}_S_{}_C_{}_U{}.png".format(
            self.train, self.direction, self.avg_speed, self.component, self.uuid
            )
            )


class Wave(Cluster):
    """
    Prototype for wavelet algebraic manipulation.
    EARLY DEVELOPMENT STAGE - DO NOT USE!
    """

    def __init__(self, uid, side, sensor, index):
        """
        Init Wave object

        Parameters
        ----------

        uid: str
            UUID-like string
        side: str = {"N","S"}
            Side of the train
        sensor: int
            Sensor ID
        index:
            Index of the anomaly detected
        """
        super().__init__(uid, radius = shared.CLUSTER_RADIUS)
        self.wave = self.get_cwt(side = side, sensor = sensor, index = index)

    def __add__(self, other):
        if isinstance(other, Wave):
            return self.wave + other.wave
        else:
            raise TypeError

    def __mul__(self, other):
        if isinstance(other, Wave):
            # Frobenius Product
            return np.trace(np.product(self.wave, np.conjugate(other.wave)))
        if isinstance(other, float):
            return np.multiply(self.wave, other)
