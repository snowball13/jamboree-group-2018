import glob
from dataextraction import *
# from jamboree import *
from AnalyseData import *

if __name__ == '__main__':
    # get all files
    # lossfiles = glob.glob("AIR_data/Loss_Data/Complete/Region_*_DR.csv")
    # lossfiles = sorted(lossfiles)
    # print(lossfiles)
    # data_loss = extract_data(lossfiles[0:1])
    # print(data_loss.keys())

    # ids, avg = get_average_lob1(data_loss)

    lossarray = np.load("AAL.npy")

    ids = lossarray[:, 0]
    avg = lossarray[:, 1]

    ids_pred, data_pred = get_prediction_data()

    avg = get_correct_data_arrays(ids, avg, ids_pred, data_pred)
    print(len(ids_pred), len(avg), len(data_pred))
    print(ids_pred.shape, avg.shape, data_pred.shape)

    testids = get_test_data(ids_pred)
    print(testids.shape)

    testav = get_test_data(avg)
    print(testav.shape)

    testpred = get_test_data(data_pred)
    print(testpred.shape)

    np.save("testids.npy", testids)
    np.save("testav.npy", testav)
    np.save("testpred.npy", testpred)

    valids = get_validation_data(ids_pred)
    print(valids.shape)

    valav = get_validation_data(avg)
    print(valav.shape)

    valpred = get_validation_data(data_pred)
    print(valpred.shape)

    np.save("valids.npy", valids)
    np.save("valav.npy", valav)
    np.save("valpreds.npy", valpred)

    # print(len(data['AIRSID']))
    # print(data.keys())
    # data_test = get_test_data(data)
    # print(len(data['AIRSID']))
