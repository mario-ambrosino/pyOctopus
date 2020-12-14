import numpy as np
from numpy.fft import *
from scipy.signal import correlate as corr
import zipfile


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

    :return:
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

    Returns
    -------

    """
    with zipfile.ZipFile(input_path, 'r') as zip_ref:
        zip_ref.extractall(output_path)
        zip_ref.close()