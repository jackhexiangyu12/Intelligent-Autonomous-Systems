import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_value_function(values):
    """
    Plots the value function as a heatmap.
    """
    plt.figure()
    sns.heatmap(values, annot=True, cmap='Blues', fmt='.3f')
    plt.show()