# TODO GitSecret Task

# Versioning
module_version = "0.5"
module_last_modified_data = "2020/12/15"
verbose_mode = False

# Path
DATASET_PATH = "datasets"
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
CLUSTERING_EPS = 15
CLUSTERING_MS = 15
CLUSTERING_METRICS = "cityblock"
CLUSTER_RADIUS = 256

# GROUND WINDOW
GROUND_WINDOW_DER = 0.5

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
