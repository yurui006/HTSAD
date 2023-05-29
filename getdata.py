import preprocess as pp
import datetime, time
import hashlib
import torch
from torch.utils.data import Dataset


class Record:
    def __init__(self):
        pass

    def trans(self):
        """Record-t, (s, vc, vn)"""
        funcname = "funcname"
        if funcname in pp.func.keys():
            return timestamp, pp.func[funcname](content)
        else:
            return timestamp, pp.svrother(content)


def readlogline(line):
    """logline - Record"""
    record = Record()
    return record


def readlog(path):
    """logfile - [Record]"""
    with open(path, "r") as f:
        done = 0
        logs = []
        while not done:
            line = f.readline()
            if line == "":
                done = 1
            else:
                currecord = readlogline(line)
                logs.append(currecord)
    return logs


def log_processor(records):
    """[Record] - [t, (s, vc, vn)]"""
    processed_log = []
    for item in records:
        processed_log.append(item.trans())
    return processed_log


def vec2tensors(logs):
    """[Record] - 4*tensor((1, length, dim))"""
    logvec = log_processor(logs)
    input_t = []
    input_s = []
    input_vc = []
    input_vn = []
    for item in logvec:
        input_t.append(item[0])
        tensors = pp.get_tensors(item[1][0], item[1][1], item[1][2])
        input_s.append(tensors[0])
        input_vc.append(tensors[1])
        input_vn.append(tensors[2])

    time_seq = torch.Tensor(input_t).view(-1, 1)
    s_seq = torch.cat(input_s, 1).transpose(0, 1).contiguous()
    vc_seq = torch.cat(input_vc, 1).transpose(0, 1).contiguous()
    vn_seq = torch.cat(input_vn, 1).transpose(0, 1).contiguous()

    s_seq = s_seq.view(1, -1, pp.class_num)
    time_seq = time_seq.view(1, -1, 1)
    vc_seq = vc_seq.view(1, -1, pp.categorical_num)
    vn_seq = vn_seq.view(1, -1, pp.numerical_num)

    return s_seq, time_seq, vc_seq, vn_seq


def split_seq(seq, window, stride):
    """tensor((1, length, dim)) - tensor((length-window+1, window, dim))"""
    length = seq.shape[1]
    num = (length - window) // stride + 1
    # print(str(num) + " sequences")
    splited_seqs = []
    for i in range(num):
        pos = i * stride
        splited_seqs.append(seq[:, pos: window + pos, :])
    splited_seqs = torch.cat(splited_seqs)
    return splited_seqs


def split_tensor(t, seq_batchsize):
    """tensor((length-window+1, length, dim)) - [tensor((seq_batchsize, length, dim))]"""
    tensors = []
    length = t.shape[0] // seq_batchsize
    for i in range(length):
        tensors.append(t[i * seq_batchsize: (i + 1) * seq_batchsize])
    tensors.append(t[length * seq_batchsize:])
    return tensors


def log2tensors(logpath):
    """logfile - 4*tensor((1, length, dim))"""
    logs = readlog(logpath)
    s_seq, time_seq, vc_seq, vn_seq = vec2tensors(logs)
    return s_seq, time_seq, vc_seq, vn_seq


def split_tensors(s_seq, time_seq, vc_seq, vn_seq, seq_batchsize, window, stride):
    """4*tensor((length-window+1, length, dim)) - 4*[tensor((seq_batchsize, length, dim))]"""
    data_s = split_seq(s_seq, window, stride)
    data_t = split_seq(time_seq, window, stride)
    data_vc = split_seq(vc_seq, window, stride)
    data_vn = split_seq(vn_seq, window, stride)
    data_s = split_tensor(data_s, seq_batchsize)
    data_t = split_tensor(data_t, seq_batchsize)
    data_vc = split_tensor(data_vc, seq_batchsize)
    data_vn = split_tensor(data_vn, seq_batchsize)
    return data_s, data_t, data_vc, data_vn


class SeqDataset(Dataset):
    def __init__(self, s, t, vc, vn):
        self.s_seq = s
        self.t_seq = t
        self.vc_seq = vc
        self.vn_seq = vn

    def __getitem__(self, index):
        return self.s_seq[index], self.t_seq[index], self.vc_seq[index], self.vn_seq[index]

    def __len__(self):
        return self.s_seq.shape[0]


def vecs2multi(vecs, windowsize):
    """[[](len=l)](len=seqlen) - tensor(seqlen/windowsize + 1, l)"""
    vecs = torch.tensor(vecs)
    print(vecs.shape)
    multi = torch.empty((0, vecs.shape[1]))
    i = 0
    while i + windowsize < vecs.shape[0]:
        temp = vecs[i:i+windowsize]
        max_values, _ = torch.max(temp, dim=0)
        multi = torch.cat((multi, max_values.unsqueeze(0)), dim=0)
        i += windowsize
    temp = vecs[i: -1]
    if temp.shape[0] != 0:
        max_values, _ = torch.max(temp, dim=0)
        multi = torch.cat((multi, max_values.unsqueeze(0)), dim=0)
    return multi