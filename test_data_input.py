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

    ids_pred, data_pred1, data_pred2, data_pred3, data_pred4 = get_prediction_data()

    avg = get_correct_data_arrays(ids, avg, ids_pred, data_pred1)
    print(len(ids_pred), len(avg), len(data_pred1))
    print(ids_pred.shape, avg.shape, data_pred1.shape)

    testids = get_test_data(ids_pred)
    print(testids.shape)

    testav = get_test_data(avg)
    print(testav.shape)

    # testpred = get_test_data(data_pred)
    # print(testpred.shape)

    testpred1 = get_test_data(data_pred1)
    print(testpred1.shape)

    testpred2 = get_test_data(data_pred2)
    print(testpred2.shape)

    testpred3 = get_test_data(data_pred3)
    print(testpred3.shape)

    testpred4 = get_test_data(data_pred4)
    print(testpred4.shape)

    np.save("testids.npy", testids)
    np.save("testav.npy", testav)

    # np.save("testpred.npy", testpred)

    np.save("testpred1.npy", testpred1)
    np.save("testpred2.npy", testpred2)
    np.save("testpred3.npy", testpred3)
    np.save("testpred4.npy", testpred4)

    valids = get_validation_data(ids_pred)
    print(valids.shape)

    valav = get_validation_data(avg)
    print(valav.shape)

    # valpred = get_validation_data(data_pred)
    # print(valpred.shape)

    valpred1 = get_validation_data(data_pred1)
    print(valpred1.shape)

    valpred2 = get_validation_data(data_pred2)
    print(valpred2.shape)

    valpred3 = get_validation_data(data_pred3)
    print(valpred3.shape)

    valpred4 = get_validation_data(data_pred4)
    print(valpred4.shape)

    np.save("valids.npy", valids)
    np.save("valav.npy", valav)

    # np.save("valpreds.npy", valpred)

    np.save("valpred1.npy", valpred1)
    np.save("valpred2.npy", valpred2)
    np.save("valpred3.npy", valpred3)
    np.save("valpred4.npy", valpred4)
