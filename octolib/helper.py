"""
Helper Routine: High-Level API to use pyOctopus functionality.
"""

# System Methods
import time
import warnings
import pandas as pd

# Graphic Import
warnings.filterwarnings('ignore')
from octolib.shared_parameters import *
import octolib.trip as tp


def generate_vibration_images():
    """
    Generate Vibration Images with aligned track for all the item in the frame-frame, with weldings in black.
    Returns
    -------

    """
    meta = tp.MetaFrame(path_data = DATASET_PATH, path_meta = META_PATH)
    for identifier, uid in enumerate(meta.UUID):
        X = tp.Octopus.Track(uid)
        print("[{}] # ".format(time.ctime()) + "Acceleration Preprocessing Completed.")
        print("-Data Load Completed.")
        X.plot_accelerations()


def automate_score_generation(
        threshold_set,
        detection_range_set,
        e_set,
        ms_set,
        cm_set,
        mode = "grid_search"
        ):
    """
    Automates the generation of scores to evaluate dataset. A first mode proposed is a grid search algorithm. The
    scores are saved in csv files into the export folder.

    Parameters
    ----------
    threshold_set: list
        A list of floats containing thresholds (cfr. self.test_alignment_score method)
    detection_range_set: list
        A list of floats containing detection range (cfr. self.test_alignment_score method)
    e_set: list
        A list of ints containing epsilon for DBSCAN (cfr. self.test_alignment_score method)
    ms_set: list
        A list of ints containing minimum samles for DBSCAN (cfr. self.test_alignment_score method). To reduce
        complexity start setting ms_set elements equal to e_sets
    cm_set: list
        A list of strings containing metrics to choose in [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’]
    mode: string
        Execution mode: actually we're using grid_search but a random_search method is expected soon

    Returns
    -------

    """
    if mode == "grid_search":
        for threshold in threshold_set:
            for detection_range in detection_range_set:
                for e in e_set:
                    for ms in ms_set:
                        for cm in cm_set:
                            print("[{}] # ".format(time.ctime())
                                  + "Parameters:: M: {}.T: {}.D: {}.E: {}.C: {}.M: {}.".format(
                                    mode, threshold, detection_range, e, ms, cm)
                                  )
                            test_alignment_score(
                                threshold = threshold,
                                detection_range = detection_range,
                                plot = False,
                                export = True,
                                epsilon = e,
                                min_samples_cluster = ms,
                                cluster_metrics = cm,
                                )


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
     - export the frame-data and score in a score file.

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
    meta = tp.MetaFrame(path_data = DATASET_PATH, path_meta = META_PATH)
    X = None
    print("[{}] # ".format(time.ctime()) + "Score Generator Helper started.")
    for identifier, uid in enumerate(meta.UUID):
        X = tp.Octopus.Track(uid)
        for side in X.sides:
            for sensor in range(4):
                X.anomaly_cluster[side][sensor]["Error_X"] = detection_range
                # North&South Side anomaly detection & z-score
                print("[{}] # ".format(time.ctime()) + "Sensor:{}/4 - Side:{}".format(sensor + 1, side))
                X.get_anomalies(side = side, sensor = sensor, threshold = threshold)
        print("[{}] # ".format(time.ctime()) + "Anomaly Detection completed.")
        X.get_anomaly_clusters(eps = epsilon,
                               min_samples = min_samples_cluster,
                               metric = cluster_metrics)
        X.evaluate_prediction(error_length = detection_range)
        for side in X.sides:
            for sensor in range(4):
                score_list.append(
                    (uid, X.train, X.direction, X.avg_speed, X.num_trip, X.component, side, sensor,
                     X.anomaly_cluster[side][sensor]["Avg_Index"],
                     X.anomaly_cluster[side][sensor]["performance"][0],
                     X.anomaly_cluster[side][sensor]["performance"][1],
                     X.anomaly_cluster[side][sensor]["performance"][2],
                     threshold,
                     detection_range,
                     epsilon,
                     min_samples_cluster,
                     cluster_metrics,
                     )
                    )
    if plot:
        X.plot_clusters()
        X.plot_scores()
        print("[{}] # ".format(time.ctime()) + "Scores Plot completed.")

    if export:
        # Generate Lists
        columns = ("UUID", "Train", "Direction", "Speed", "Num_Trip", "Component", "Side", "Sensor",
                   "Cluster Centroids", "PD", "ED", "PFA", "Error_X", "Detection_Range", "Epsilon",
                   "Min_Samples_Cluster", "Cluster_Metrics")
        export_df = pd.DataFrame(score_list, columns = columns)
        export_df.to_csv(
            "export/scores_T{}_D{}_E{}_C{}_M{}.csv".format(threshold, detection_range, epsilon, min_samples_cluster,
                                                           cluster_metrics, )
            )
    print("[{}] # ".format(time.ctime()) + "Score Generator Helper completed.")


def generate_shifts(export = True):
    dict_shifts = {}
    meta = tp.MetaFrame(path_data = DATASET_PATH, path_meta = META_PATH)
    for identifier, uid in enumerate(meta.UUID):
        try:
            X = tp.Octopus.Track(uid)
            print("-Data Load Completed.")
            N_shifts, S_shifts = X.shift_sync_signals()
            print("Shift Evaluated:\nN:{}\nS:{}".format(N_shifts, S_shifts))
            dict_shifts[uid] = [N_shifts, S_shifts]
        except IndexError:
            print("[{}] # ".format(time.ctime()) + "Index out of range - Error for track {}".format(uid))
    if export:
        out_file = open("shifts.txt", "w")
        out_file.write(str(dict_shifts))
        out_file.close()
    return dict_shifts
