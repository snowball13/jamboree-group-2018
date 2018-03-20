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


def get_test_data(data):

    n = len(data)

    # get 20 percent of indices
    indices = get_datasplit_indices(n)

    # remove data of indices
    output = np.delete(data, indices)

    return output


def get_validation_data(data):

    n = len(data)

    # get 20 percent of indices
    indices = get_datasplit_indices(n)

    output = data[indices]

    return output


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
    length = len(ids2)
    avg = np.zeros(length)
    for j in range(len(ids2)):
        for k in range(len(ids1)):
            if ids2[j] == ids1[k]:
                avg[j] = avg1[k]
                break
    return avg


def get_ids_list(data):
    ids = sorted(set(data['AIRSID']))
    return ids
