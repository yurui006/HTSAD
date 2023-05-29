import getargs
import getdata as gd
import logging

def trainer(servername, modelname, modelnum, device):
    model, args = getargs.get_args(modelname)
    args["modelnum"] = modelnum
    args["severname"] = servername
    trainpath = "servers/trainset_new/" + servername
    print("Analysing trainfile")
    logging.info("Analysing trainfile.")
    s_seq, time_seq, vc_seq, vn_seq = gd.log2tensors(trainpath)
    logging.info("{} logs".format(s_seq.shape[1]))
    model.train(args, device, s_seq, time_seq, vc_seq, vn_seq)



def tester(servername, modelname, modelnum, device, perepoch=0):
    model, args = getargs.get_args(modelname)
    test_args = getargs.get_test_args(modelname)
    test_args["modelnum"] = modelnum
    test_args["severname"] = servername
    test_args["BatchSize"] = 4096
    test_args["perepoch"] = perepoch
    test_args["epochnum"] = args["Epoches"]

    testpath = "servers/testset/" + servername
    print("Analysing testfile")
    logging.info("Analysing testfile.")
    s_seq, time_seq, vc_seq, vn_seq = gd.log2tensors(testpath)
    logging.info("{} logs".format(s_seq.shape[1]))
    model.test(test_args, device, s_seq, time_seq, vc_seq, vn_seq)

