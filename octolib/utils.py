"""
contains functions which could be useful for several classes.
"""

# System Libraries
import os
import zipfile
import json

# Third-Party Libraries
import numpy as np
import pandas as pd
from numpy.fft import *
from scipy.signal import correlate as corr


def get_dataset(file_path = "private/data_packages"):
    """
    Method to retrive dataset for repository init

    Parameters
    ----------

    file_path: str
        Path for data packages root folder.
    """
    from mega import Mega
    KEYS_FOLDER = "private/keys"
    KEY_FILE = os.path.join(KEYS_FOLDER, "keys.json")
    key_pointer = open(KEY_FILE, "r")
    keyring = json.load(key_pointer)
    repo = keyring["1"]["DATASET_REPOSITORY"]
    data_key = keyring["1"]["KEY_REPOSITORY"]
    data_name = keyring["1"]["DATASET_NAME"]
    data_url = f"{repo}#{data_key}"
    # login using a temporary anonymous account
    mega = Mega()
    m = mega.login()
    m.download_url(url = data_url, dest_path = file_path, dest_filename = data_name)


def init_folder(folders):
    """
    Generate the folders to quickstart the workspace

    Parameters
    ----------
    folders: list(str)
        the folders list

    """
    import pathlib
    for folder in folders:
        pathlib.Path(folder).mkdir(parents = True, exist_ok = True)


def get_directory_structure(data_folder_path):
    """
    Creates a nested dictionary that represents the folder structure of input_path. Needed for the execution of the
    dataframe generator.

    Parameters
    ----------
    data_folder_path: (string)
        The path to analyze

    Returns
    -------
    directory: (dict)
        The dictionary which maps the information about the folder

    """
    from functools import reduce
    directory = {}
    input_path = data_folder_path.rstrip(os.sep)
    start = input_path.rfind(os.sep) + 1
    for path, dirs, files in os.walk(input_path):
        folders = path[start:].split(os.sep)
        subdir = dict.fromkeys(files)
        parent = reduce(dict.get, folders[:-1], directory)
        parent[folders[-1]] = subdir
    return directory


def cross_correlation_using_fft(x, y):
    """
    Performs Fourier Phase Shift between signals x and y

    Parameters
    ----------
    x: numpy.ndarray
        Reference Signal
    y: numpy.ndarray
        Comparison Signal

    Returns
    -------
    fourier_phase_shift: int
        Shift Index
    """
    f1 = fft(x)
    f2 = fft(np.flipud(y))
    cc = np.real(ifft(f1 * f2))
    return fftshift(cc)


def signal_sync(reference, signal):
    """
    Returns the optimal shift to maximize correlation between reference signal and the analyzed signal to be shifted.

    Parameters
    ----------

    reference:
        Leading (static) signal;
    signal:
        Signal to be shifted.

    Returns
    -------

    shift: int
        the shift for signal, corrected because of distorsion in correlation.
    """
    corr_data = corr(reference, signal)
    shift = np.argmax(corr_data)
    return shift - int(len(corr_data) / 2)


def extract_data(input_path, output_path):
    """
    Given a zipped dataset, extract it in output_path.

    Parameters
    ----------

    input_path: string
        The path of the zip file to be extracted
    output_path: string
        The path of the target folder for the extraction
    """
    with zipfile.ZipFile(input_path, 'r') as zip_ref:
        zip_ref.extractall(output_path)
        zip_ref.close()


def skiprow_logic(index, start, end, step = 1):
    """
    Function lambidified in ``pandas.read_csv()`` method to select certain slice of data

    Parameters
    ----------

    index: int
        the column chosen. It is the main argument of ``skiprow_logic`` lambda function
    start: int
        iloc starting position of the DataFrame
    end: int
        iloc end position of the DataFrame
    step: int
        iloc gap between iloc index of the DataFrame

    Returns
    -------

    isSkipped: bool
        if the item should be skipped or not.
    """
    # please note: implicit skipping of header in dataset
    if index in range(start, end, step):
        return False
    else:
        return True


def separator_parser(dataset):
    """
    Provides a way to parse dataset with different separators.

    Parameters
    ----------

    dataset: str
        the name of the root folder, related to the dataset

    Returns
    -------

    separator: str
        the separator character
    """
    if dataset == "Santa_Maria_a_Vico":
        return "\t"
    elif dataset == "Estratto_Rettilineo_AR":
        return " "


def naive_integration(data):
    """
    It simply make the cumulative sum of the data, performing a naive integration.
    In order to get better numerical result one could use three-point or five-point
    interpolation of the data or using spline approximation but at this point isn't necessary.

    Parameters
    ----------
    data: iterable
        the data generic array to be "integrated"

    Returns
    -------
    integrated_data: numpy.ndarray
        the data integrated.


    """
    return pd.DataFrame(np.cumsum(data)).to_numpy()
