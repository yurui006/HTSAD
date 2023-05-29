import torch
import torch.nn as nn
from torch import optim
import csv
import pickle
import os
from torch.utils.data import DataLoader
import getdata as gd
import warnings
import logging
warnings.filterwarnings("ignore")


torch.set_default_tensor_type(torch.DoubleTensor)


class HTSAD(nn.Module):
    def __init__(self, Dim, Embedding_sz, Output_sz, Eventfilter_sz, Event_num, Categorical_num, Numerical_num, Device):
        super().__init__()
        self.Dim = Dim  # dim of the output vector which is passed to the next step
        self.Embedding_sz = Embedding_sz  # dim of the vector after embedding, also dim of s and x
        self.Output_sz = Output_sz  # dim of the h and c vector, also dim of the vector before Dim vec
        self.Eventfilter_sz = Eventfilter_sz  # dim of the vec which is used to change s to es
        self.Device = Device
        self.Wx = nn.Parameter(torch.Tensor(Embedding_sz, Output_sz * 4))  # paras to multi with x
        self.Wh = nn.Parameter(torch.Tensor(Output_sz, Output_sz * 4))  # paras to multi with h
        self.Wc = nn.Parameter(torch.Tensor(3, Output_sz))  # paras to multi with c
        self.bias = nn.Parameter(torch.Tensor(Output_sz * 4))  # b
        self.Ve = nn.Parameter(torch.Tensor(Event_num, Embedding_sz))  # paras to multi with "type"
        self.Vc = nn.Parameter(torch.Tensor(Categorical_num, Embedding_sz))  # paras to multi with valuec
        self.Vn = nn.Parameter(torch.Tensor(Numerical_num, Embedding_sz))  # paras to multi with valuen
        self.linear = nn.Linear(Output_sz, Dim)  # layer to get Dim vec
        self.eventfilter1 = nn.Linear(Embedding_sz, Eventfilter_sz)
        self.eventfilter2 = nn.Tanh()
        self.eventfilter3 = nn.Linear(Eventfilter_sz, Output_sz)
        self.eventfilter4 = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for weight in self.parameters():
            weight.data.uniform_(-1, 1)

    def forward(self, event, time, vc, vn, init_states=None):
        """Assumes inputs is of shape (batchsize, sequencelength, features)"""
        """features = (event, time, valuec, valuen)"""
        batchsize, seq_len, _ = event.size()
        if init_states is None:
            h_t, c_t = (torch.randn(batchsize, self.Output_sz).to(self.Device),
                        torch.randn(batchsize, self.Output_sz).to(self.Device))
        else:
            h_t, c_t = init_states  # (batchsize, Output_sz)
        HS = self.Output_sz
        for t in range(seq_len):
            event_t = event[:, t, :]  # (batchsize, Event_num), type
            vc_t = vc[:, t, :]  # (batchsize, Categorical_num)
            vn_t = vn[:, t, :]  # (batchsize, Numerical_num)
            s = event_t @ self.Ve  # (batchsize, Embedding_sz)
            x_t = s + 2.0 * (vc_t @ self.Vc + torch.tanh(vn_t @ self.Vn))  # (batchsize, Embedding_sz)
            # batch the computations into a single matrix multiplication
            gates = x_t @ self.Wx + h_t @ self.Wh + self.bias  # (batchsize, 4*Output_sz)
            es = self.eventfilter2(self.eventfilter1(s))  # (batchsize, Eventfilter_sz)
            es = self.eventfilter4(self.eventfilter3(es))  # (batchsize, Output_sz)
            j_t = es  # (batchsize, Output_sz)
            g_t = torch.tanh(gates[:, HS * 2:HS * 3])  # (batchsize, Output_sz)
            i_t, f_t, o_t = (torch.sigmoid(gates[:, :HS] + c_t * self.Wc[0]),  # input, (batchsize, Output_sz)
                             torch.sigmoid(gates[:, HS:HS * 2] + c_t * self.Wc[1]),  # forget, (batchsize, Output_sz)
                             torch.sigmoid(gates[:, HS * 3:] + c_t * self.Wc[2]),  # output, (batchsize, Output_sz)
                             )
            c_t_hat = f_t * c_t + i_t * g_t  # (batchsize, Output_sz)
            c_t = j_t * c_t_hat + (1 - j_t) * c_t  # (batchsize, Output_sz)
            h_t_hat = o_t * torch.tanh(c_t_hat)  # (batchsize, Output_sz)
            h_t = j_t * h_t_hat + (1 - j_t) * h_t  # (batchsize, Output_sz)
        y_t = self.linear(h_t)  # (batchsize, Dim)
        return y_t


def train(args, device, s_seq, time_seq, vc_seq, vn_seq):
    layer = HTSAD(args["Dim"], args["Embedding_sz"], args["Output_sz"], args["Eventfilter_sz"],
                        args["class_num"], args["categorical_num"], args["numerical_num"], device).to(device)
    optimizer = optim.Adam(layer.parameters(), lr=args["LearningRate"], weight_decay=args["WeightDecay"])
    s_seq = s_seq.to(device)
    time_seq = time_seq.to(device)
    vc_seq = vc_seq.to(device)
    vn_seq = vn_seq.to(device)
    with torch.no_grad():
        center = layer(s_seq, time_seq, vc_seq, vn_seq)
    data_s, data_t, data_vc, data_vn = gd.split_tensors(s_seq, time_seq, vc_seq, vn_seq, args["SeqBatchsize"],
                                                        args["WindowSize"], args["Stride"])
    del s_seq, time_seq, vc_seq, vn_seq

    train_data = gd.SeqDataset(data_s[0], data_t[0], data_vc[0], data_vn[0])
    train_loader = DataLoader(train_data, batch_size=args["BatchSize"], shuffle=True)
    del data_s, data_t, data_vc, data_vn, train_data

    layer.train()
    loss_record = []
    for epoch in range(args["Epoches"]):
        loss_record.append("epoch:" + str(epoch))
        count = 0
        count_loss = 0
        for s, t, vc, vn in train_loader:
            count += 1
            optimizer.zero_grad()
            out = layer(s, t, vc, vn)
            loss = torch.sum((out - center) ** 2, dim=1).mean()
            if count % 100 == 1:
                progress = epoch / args["Epoches"] + count / (len(train_loader) * args["Epoches"])
                print("loss = {:.4f}\t{:.2%}".format(loss.item(), progress))
            loss_record.append(str(loss.item()))
            if loss.item() > args["LossThreshold"]:
                count_loss += 1
            loss.backward()
            optimizer.step()
        if count_loss / count < args["TrainThreshold"]:
            print("Stop early.")
            break

    modelnum = args["modelnum"]
    severname = args["severname"]
    if not os.path.exists('./HTSAD/{}/epoch{}/'.format(severname, modelnum)):
        os.makedirs('./HTSAD/{}/epoch{}/'.format(severname, modelnum))
    torch.save(layer.state_dict(), './HTSAD/{}/epoch{}/HTSADmodel.pth'.format(severname, modelnum))
    torch.save(optimizer.state_dict(), './HTSAD/{}/epoch{}/HTSADoptim.pth'.format(severname, modelnum))
    f = open('./HTSAD/{}/epoch{}/HTSADcenter&args.pkl'.format(severname, modelnum), 'wb')
    pickle.dump(center, f)
    pickle.dump(args, f)
    f.close()
    f = open("./HTSAD/{}/epoch{}/HTSADtrain.csv".format(severname, modelnum), "w")
    writer = csv.writer(f)
    for i in args:
        writer.writerow((i, args[i]))
    writer.writerow("")
    for i in loss_record:
        writer.writerow((str(i),))
    f.close()


def test(test_args, device, s_seq, time_seq, vc_seq, vn_seq):
    modelnum = test_args["modelnum"]
    severname = test_args["severname"]
    f = open('./HTSAD/{}/epoch{}/HTSADcenter&args.pkl'.format(severname, modelnum), 'rb')
    center = pickle.load(f).to(device)
    args = pickle.load(f)
    f.close()

    layer = HTSAD(args["Dim"], args["Embedding_sz"], args["Output_sz"], args["Eventfilter_sz"],
                        args["class_num"], args["categorical_num"], args["numerical_num"], device).to(device)

    layer.load_state_dict(torch.load('./HTSAD/{}/epoch{}/HTSADmodel.pth'.format(severname, modelnum)))
    layer.to(device)

    data_s, data_t, data_vc, data_vn = gd.split_tensors(s_seq, time_seq, vc_seq, vn_seq, test_args["SeqBatchsize"],
                                                        args["WindowSize"], args["Stride"])
    del s_seq, time_seq, vc_seq, vn_seq

    loss_seq = []
    TIMEs = []
    for i in range(len(data_s)):
        testdata_s, testdata_t, testdata_vc, testdata_vn = data_s[i], data_t[i], data_vc[i], data_vn[i]
        testdata_s = testdata_s.to(device)
        testdata_t = testdata_t.to(device)
        testdata_vc = testdata_vc.to(device)
        testdata_vn = testdata_vn.to(device)
        test_data = gd.SeqDataset(testdata_s, testdata_t, testdata_vc, testdata_vn)
        test_loader = DataLoader(test_data, batch_size=args["BatchSize"], shuffle=False)
        del test_data, testdata_s, testdata_t, testdata_vc, testdata_vn

        layer.eval()
        for s, t, vc, vn in test_loader:
            out = layer(s, t, vc, vn)
            loss = torch.sum((out - center) ** 2, dim=1)
            tempt = t.detach().cpu().numpy()
            tempt = tempt[:, -1, :]
            tempt = tempt.squeeze().tolist()
            if isinstance(tempt, float):
                tempt = [tempt]
            TIMEs += tempt
            cur_loss = loss.detach().cpu().numpy().tolist()
            loss_seq += cur_loss

    f = open("./HTSAD/{}/epoch{}/HTSADtest.csv".format(severname, modelnum), "w")
    writer = csv.writer(f)
    for i in loss_seq:
        writer.writerow((str(i),))
    f.close()
    f = open("./HTSAD/{}/epoch{}/HTSADtesttime.csv".format(severname, modelnum), "w")
    writer = csv.writer(f)
    for i in TIMEs:
        writer.writerow((str(i),))
    f.close()

