"""code only for the fourier transform."""

# Import required packages
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.io import wavfile as wav


# useful defined functions
def create_vis(
    x_array: np.array = None,
    y_array: np.array = None,
    coords: list = None,
    title: str = None,
    x_title: str = None,
    y_title: str = None,
) -> plt.Figure:
    """Create a visualisation."""
    # set up figure to amend
    fig, ax = plt.subplots(figsize=(10, 4))

    # plot graph depending on input
    if coords:
        ax.plot(coords, color="#A020F0")
    elif x_array:
        ax.plot(x_array, y_array, color="#A020F0")
    else:
        ax.plot(y_array, color="#A020F0")

    # titles and axes constructing
    if x_title:
        ax.set_xlabel(x_title)
    if y_title:
        ax.set_ylabel(y_title)
    if title:
        fig.suptitle(t=title, fontsize="xx-large", color="#A020F0")

    return fig


def calculate_fourier_transform(
    data: np.array,
    frequency: int,
    n_arr: np.array,
    dp: int,
) -> float:
    """Compute fourier transform equation for a single frequency."""
    # complex number calculation
    equation_inside_trig_func = -2 * np.pi * frequency * n_arr / dp
    real_component = np.mean(data * np.cos(equation_inside_trig_func))
    imaginary_component = np.mean(data * -np.sin(equation_inside_trig_func))

    # combining real and imaginary parts
    complex_output = complex(real_component, imaginary_component)

    # to get a single output, we find the absolute of the complex number
    return np.abs(complex_output)


# Get sound information and the required data descriptors
sound_info = {}

# Initial look into the stag sound
stag_sound_filename = Path.cwd() / "Red_stag_roar-Juan_Carlos_-2004708707.wav"
sound_info["rate"], sound_info["data"] = wav.read(stag_sound_filename)
sound_info["duration"] = 11

# clean data
sound_info["data_1d"] = np.delete(sound_info["data"], 0, 1)
sound_info["nonzero_data"] = sound_info["data_1d"][sound_info["data_1d"] != 0]

# required variables
sound_info["num_of_nonzero_data_points"] = len(sound_info["nonzero_data"])
sound_info["frequencies"] = np.arange(20000)

time_based_vis = create_vis(
    y_array=sound_info["data_1d"],
    title="Sound of a stag",
    x_title="Time",
    y_title="Intensity",
)

# list of transformed data points for each frequency
frequencies_transformed = [
    calculate_fourier_transform(
        data=sound_info["nonzero_data"],
        frequency=freq,
        n_arr=np.arange(sound_info["num_of_nonzero_data_points"]),
        dp=sound_info["num_of_nonzero_data_points"],
    )
    for freq in sound_info["frequencies"]
]

# Analysis of the the frequency intensities
print(
    f"The max term in the frequency power is{np.max(frequencies_transformed)} and "
    f"the min is {np.min(frequencies_transformed)}"
)


# Visualise fourier transform transform output
freq_based_vis = create_vis(
    y_array=frequencies_transformed,
    title="Fourier Transform",
    x_title="Frequencies",
    y_title="Power",
)


# Number of samples in normalized_tone
N = sound_info["data_1d"].size

# use pre-built fasst fourier transform function from scipy
yf = fft(sound_info["data_1d"].flatten())
xf = fftfreq(N, 1 / sound_info["data_1d"].size)

# clean and plot results
coords = zip(xf, yf, strict=False)
coords_filtered = [c for c in coords if c[0] > 0 and c[0] < 20000]

# plot output
fft_output = create_vis(
    coords=coords_filtered,
    title="Fast Fourier Transform from Scipy",
    x_title="Frequency",
    y_title="Power",
)
