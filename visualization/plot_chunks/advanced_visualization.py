#! /home/toni/.pyenv/shims/python3
"""
Created on Nov 25, 2020

@author:toni
"""
from __future__ import division, print_function

import shutil
import tempfile
import urllib
import zipfile

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


df = pd.read_csv("property_tax_report_2018.csv")


temp_dir = tempfile.mkdtemp()
data_source = ('http://archive.ics.uci.edu/ml/machine-learning-databases/'
               '00275/Bike-Sharing-Dataset.zip')
zipname = temp_dir + '/Bike-Sharing-Dataset.zip'
urllib.request.urlretrieve(data_source, zipname)

zip_ref = zipfile.ZipFile(zipname, 'r')
zip_ref.extractall(temp_dir)
zip_ref.close()

daily_path = temp_dir + '/day.csv'
daily_data = pd.read_csv(daily_path)
daily_data['dteday'] = pd.to_datetime(daily_data['dteday'])
drop_list = ['instant', 'season', 'yr', 'mnth', 'holiday', 'workingday',
             'weathersit', 'atemp', 'hum']
daily_data.drop(drop_list, inplace=True, axis=1)

shutil.rmtree(temp_dir)

# daily_data.head()  # PyLint throws an error if uncommented

# Set some parameters to apply to all plots. These can be overridden
# in each plot if desired
# Plot size to 14" x 7"
matplotlib.rc('figure', figsize=(14, 7))
# Font size to 14
matplotlib.rc('font', size=14)
# Do not display top and right frame lines
matplotlib.rc('axes.spines', top=False, right=False)
# Remove grid lines
matplotlib.rc('axes', grid=False)
# Set backgound color to white
matplotlib.rc('axes', facecolor='white')


def main():
    pass


if __name__ == "__main__":
    main()
