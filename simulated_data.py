# -*- coding: utf-8 -*-
"""
Original version : @author: Rémi Flamary / GitHub : https://github.com/rflamary/OST/
Modified version : Project Lucas Haubert / Course "Computational Optimal Transport"
"""

import numpy as np
import matplotlib.pyplot as plt

# 1st experiment : Reproduce the toy experiment of article "Optimal spectral transportation with application to music transcription"

def create_harmonic_template(fundamental_freq, num_harmonics, max_freq, sample_rate, std_dev_factor):

    frequencies = np.linspace(0, max_freq, sample_rate)
    spectrum = np.zeros_like(frequencies)

    damping_factor = 0.5

    for i in range(1, num_harmonics + 1):

        harmonic_freq = i * fundamental_freq
        amplitude = np.exp(-damping_factor * (i - 1))
        std_dev = 100 / std_dev_factor
        spectrum += amplitude * np.exp(-0.5 * ((frequencies - harmonic_freq) ** 2) / (std_dev ** 2))

    return frequencies, spectrum

def create_harmonic_dictionary(num_templates, start_freq, end_freq, num_harmonics, max_freq, sample_rate, std_dev_factor):
    harmonic_dict = {}
    frequency_step = (end_freq - start_freq) / (num_templates - 1)

    for i in range(num_templates):
        fundamental_freq = start_freq + i * frequency_step
        frequencies, spectrum = create_harmonic_template(fundamental_freq, num_harmonics, max_freq, sample_rate, std_dev_factor)
        harmonic_dict[fundamental_freq] = spectrum

    return frequencies, harmonic_dict

# Parameters
num_templates = 12
start_freq = 110  
end_freq = 220   
num_harmonics = 6
std_dev_factor = 20
max_freq = 800  
sample_rate = 44100 

# Harmonic dictionary
frequencies, harmonic_dict = create_harmonic_dictionary(num_templates, start_freq, end_freq, num_harmonics, max_freq, sample_rate, std_dev_factor)

# Plot
def plot_harmonic_dict(frequencies, harmonic_dict):
    plt.figure(figsize=(15, 6))
    for spectrum in harmonic_dict.values():
        plt.plot(frequencies, spectrum)
    plt.title('Synthetic dictionary of spectral templates W')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

# plot_harmonic_dict(frequencies, harmonic_dict)

# Build propagation of fundamental shifts

# Selection of second and sixth templates
second_template = list(harmonic_dict.values())[1]
sixth_template = list(harmonic_dict.values())[5]

# Add them
combined_sample = second_template + sixth_template

# Propagated shifting function
def apply_cumulative_frequency_shift(fundamental_freq, shift_factor, num_harmonics, max_freq, sample_rate, std_dev_factor):
    shifted_spectrum = np.zeros(sample_rate)
    damping_factor = 0.5

    for i in range(1, num_harmonics + 1):
        shifted_harmonic_freq = fundamental_freq * i * (shift_factor ** i)
        amplitude = np.exp(-damping_factor * (i - 1))
        std_dev = 100 / std_dev_factor
        shifted_spectrum += amplitude * np.exp(-0.5 * ((frequencies - shifted_harmonic_freq) ** 2) / (std_dev ** 2))

    return shifted_spectrum

# Starting from fundamental frequencies
fundamental_freq_2 = list(harmonic_dict.keys())[1]
fundamental_freq_6 = list(harmonic_dict.keys())[5]

shift_factor = 1.05  # 5% augmentation
shifted_template_2 = apply_cumulative_frequency_shift(fundamental_freq_2, shift_factor, num_harmonics, max_freq, sample_rate, std_dev_factor)
shifted_template_6 = apply_cumulative_frequency_shift(fundamental_freq_6, shift_factor, num_harmonics, max_freq, sample_rate, std_dev_factor)

# Generate a shift agregated sample
shifted_sample = shifted_template_2 + shifted_template_6

# Plot
def plot_toy_exp(frequencies, combined_sample, new_sample, transformation):
    plt.figure(figsize=(15, 6))
    plt.plot(frequencies, combined_sample, label='Combined Sample')
    if transformation == 'shift':
        plt.plot(frequencies, new_sample, label='Shifted Sample')
        plt.title('Comparison of Combined and Frequency-Shifted Samples')
    elif transformation == 'timbre_vars' :
        plt.plot(frequencies, new_sample, label='New Timbre Sample')
        plt.title('Comparison of Combined and New-Timbre Samples')       
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()

# plot_toy_exp(frequencies, combined_sample, shifted_sample, 'shift')
    


    


