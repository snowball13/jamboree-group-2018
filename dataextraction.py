import csv
import numpy as np


def extract_data(files):
    # First row of file is used to create the keywords for the subsequent rows that
    # are read in as a dictionary
    data = []
    initialise = True
    for filename in files:
        with open(filename) as csvfile:
            reader = csv.DictReader(csvfile)
            if initialise is True:
                data = {f: [] for f in reader.fieldnames if f}
                initialise = False
            for row in reader:
                for field, value in row.items():
                    if field:
                        if value:
                            data[field].append(float(value))

    for fieldname in data:
        data[fieldname] = np.array(data[fieldname])
    return data

def extract_predictor_data(filename):
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        if initialise is True:
            data = {f: [] for f in reader.fieldnames if f}
            initialise = False
        for row in reader:
            for field, value in row.items():
                if field:
                    data[field].append(value)
    for fieldname in data:
        data[fieldname] = np.array(data[fieldname])
    return data


def get_test_data(data):
    # get length of fist column
    n = 0
    for firstcolumn in data:
        n = len(data[firstcolumn])
        break

    # get 20 percent of indices
    indices = get_datasplit_indices(n)

    # remove data of indices
    for fieldname in data:
        data[fieldname] = np.delete(data[fieldname], indices)
    return data


def get_validation_data(data):
    # get length of fist column
    n = 0
    for firstcolumn in data:
        n = len(data[firstcolumn])
        break

    # get 20 percent of indices
    indices = get_datasplit_indices(n)

    # get data of indices
    for fieldname in data:
        data[fieldname] = data[fieldname][indices]
    return data


def get_datasplit_indices(n):
    np.random.seed(42)
    indices = np.random.choice(n - 1, int(0.2 * (n - 1)), replace=False)
    return indices

def get_average_lob1(data):
    ids = sorted(set(data['AIRSID']))
    tally = np.zeros(len(ids))
    ei = 0
    for i in ids:
        subset = [j for j in range(len(data['AIRSID'])) if data['AIRSID'][j] == i]
        tally[ei] = sum(data['LOB1'][subset])
        ei += 1
    return ids, tally / 10000.


def get_correct_data_arrays(ids1, avg1, ids2, pred2):
    length = len(ids1)
    pred = np.zeros(length)
    pred[:] = np.nan
    print len(ids1)
    for j in range(len(ids1)):
        for k in range(len(ids2)):
            if ids1[j] == ids2[k]:
                pred[j] = pred2[k]
                break
    return pred
