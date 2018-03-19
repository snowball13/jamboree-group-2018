import csv
import numpy as np


def extract_data(filename):
    # First row of file is used to create the keywords for the subsequent rows that
    # are read in as a dictionary
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        data = {f: [] for f in reader.fieldnames if f}
        for row in reader:
            for field, value in row.items():
                if field:
                    data[field].append(value)
    for fieldname in data:
        data[fieldname] = np.array(data[fieldname])
    return data


def get_test_data(data):
    np.random.seed(42)
    length = 0
    for fieldname in data:
        length = len(data[fieldname])
        break
    indices = np.random.choice(length - 1, int(0.2 * (length - 1)), replace=False)
    for fieldname in data:
        data[fieldname] = np.delete(data[fieldname], indices)
    return data


def get_validation_data(data):
    np.random.seed(42)
    length = 0
    for fieldname in data:
        length = len(data[fieldname])
        break
    indices = np.random.choice(length - 1, int(0.2 * (length - 1)), replace=False)
    for fieldname in data:
        data[fieldname] = data[fieldname][indices]
    return data

# filename = "AIR_data/Loss_data/Complete/Region_1_DR.csv"
# data = extract_data(filename)
# print(len(data['AIRSID']))
# data_test = get_test_data(data)
# print(len(data['AIRSID']))
