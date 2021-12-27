import torch
from utils.utils import data_prefetcher_two, cal_fam, setup_seed, calRes
from pretrainedmodels import xception
import utils.datasets_profiles as dp
from torch.utils.data import DataLoader#该接口主要用来将自定义的数据读取接口的输出或者PyTorch已有的数据读取接口的输入按照batch size封装成Tensor，后续只需要再包装成Variable即可作为模型的输入
from torch.optim import Adam
import numpy as np
import argparse
import random
import time

np.set_printoptions(precision=3)#precision: 保留几位小数，后面不会补0。supress: 对很大/小的数不使用科学计数法 (true)。formatter: 强制格式化，后面会补0

parser = argparse.ArgumentParser()#创建一个命令行解析器对象

parser.add_argument('--device', default="cuda:0", type=str)#解析的命令行为str类型，device就代表cuda0
parser.add_argument('--modelname', default="xception", type=str)
parser.add_argument('--distributed', default=False, action='store_true')
parser.add_argument('--upper', default="xbase", type=str,
                    help='the prefix used in save files')#存放文件时的前缀

parser.add_argument('--eH', default=120, type=int)#擦除块的高宽
parser.add_argument('--eW', default=120, type=int)

parser.add_argument('--batch_size', default=16, type=int)#mini-batch的大小
parser.add_argument('--max_batch', default=500000, type=int)#最大训练次数
parser.add_argument('--num_workers', default=4, type=int)#dataloader一次性创建num_worker个worker，（也可以说dataloader一次性创建num_worker个工作进程，worker也是普通的工作进程），并用batch_sampler将指定batch分配给指定worker，worker将它负责的batch加载进RAM。
parser.add_argument('--logbatch', default=3000, type=int)#3000batch为一个epoch
parser.add_argument('--savebatch', default=999, type=int)#训练999次保存一次模型
parser.add_argument('--seed', default=5, type=int)#初始化参数
parser.add_argument('--lr', default=0.0002, type=float, help='learning rate')#更新学习率

parser.add_argument('--pin_memory', '-p', default=False, action='store_true')#内存存储设置
parser.add_argument('--resume_model', default=None)#使用什么模型，是训练了的还是未训练的
parser.add_argument('--resume_optim', default=None)#使用什么优化器

parser.add_argument('--save_model', default=True, action='store_true')#判断
parser.add_argument('--save_optim', default=False, action='store_true')

args = parser.parse_args()

modelname = args.modelname
upper = args.upper


def Eval(model, lossfunc, dtloader):
    model.eval()
    sumloss = 0.
    y_true_all = None
    y_pred_all = None

    with torch.no_grad():
        for (j, batch) in enumerate(dtloader):
            x, y_true = batch
            y_pred = model.forward(x.cuda())

            loss = lossfunc(y_pred, y_true.cuda())
            sumloss += loss.detach()*len(x)

            y_pred = torch.nn.functional.softmax(
                y_pred.detach(), dim=1)[:, 1].flatten()

            if y_true_all is None:
                y_true_all = y_true
                y_pred_all = y_pred
            else:
                y_true_all = torch.cat((y_true_all, y_true))
                y_pred_all = torch.cat((y_pred_all, y_pred))

    return sumloss/len(y_true_all), y_true_all.detach(), y_pred_all.detach()

def Log(log):
    print(log)#输出基本信息
    f = open("./logs/"+upper+"_"+modelname+".log", "a")#创建一个文件，输出训练后的结果信息
    f.write(log+"\n")
#怎么不写一个log----------
    f.close()

if __name__ == "__main__":
    Log("\nModel:%s BatchSize:%d lr:%f" % (modelname, args.batch_size, args.lr))#进入函数
    torch.cuda.set_device(args.device)#选择gpu
    setup_seed(args.seed)#初始化参数

    print("cudnn.version:%s enabled:%s benchmark:%s deterministic:%s" % (torch.backends.cudnn.version(), torch.backends.cudnn.enabled, torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic))

    MAX_TPR_4 = 0.

    model = eval(modelname)(num_classes=1000).cuda()#为每一层设置BN和dropout
#为什么这儿是1000?

    if args.distributed:#将多块gpu并行工作
        model = torch.nn.DataParallel(model)

    optim = Adam(model.parameters(), lr=args.lr, weight_decay=0) # torch.optim是一个实现了多种优化算法的包  #导入优化器

    if args.resume_model is not None:#训练新的模型还是使用训练好了的模型
        model.load_state_dict(torch.load(args.resume_model))
    if args.resume_optim is not None:#训练默认的优化器还是别的
        optim.load_state_dict(torch.load(args.resume_optim))

    lossfunc = torch.nn.CrossEntropyLoss()#交叉熵损失，使正确预测的概率值最大


    dataset = dp.DFFD()#获取数据集

    trainsetR = dataset.getTrainsetR()
    trainsetF = dataset.getTrainsetF()

    validset = dataset.getValidset()

    testsetR = dataset.getTestsetR()#获取需要的数据集
    TestsetList, TestsetName = dataset.getsetlist(real=False, setType=2)
#选择用哪个造假数据集进行检测？
    setup_seed(args.seed)
#为什么又要初始化参数？

    traindataloaderR = DataLoader(  #往模型里面读取该数据集的数据       pytorch的数据往模型里输入的时候，不像tensorflow一样定义一下placeholder直接feeddict就可以，需要使用dataloader中转。使用dataloader了以后，可以通过dataloader的传入参数控制minibatch，shuffle，并行计算时使用的cpu核心数。而dataloader用的时候，也需要一个dataset，将数据整理成dataloader可以读得懂的结构

        trainsetR,
        batch_size=int(args.batch_size/2),
        shuffle=True,
        pin_memory=args.pin_memory,
        num_workers=int(args.num_workers)

    )

    traindataloaderF = DataLoader(
        trainsetF,
        batch_size=int(args.batch_size/2),
        shuffle=True,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers
    )

    validdataloader = DataLoader(
        validset,
        batch_size=args.batch_size*2,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers
    )

    testdataloaderR = DataLoader(
        testsetR,
        batch_size=args.batch_size*2,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers
    )

    testdataloaderList = []
    for tmptestset in TestsetList:
        testdataloaderList.append(
            DataLoader(
                tmptestset,
                batch_size=args.batch_size*2,
                pin_memory=args.pin_memory,
                num_workers=args.num_workers
            )#定义在哪个测试集上测试
        )

    print("Loaded model")

    batchind = 0
    e = 0
    sumcnt = 0
    sumloss = 0.
    while True:
        prefetcher = data_prefetcher_two(traindataloaderR, traindataloaderF)#加速数据读取
        # print(prefetcher)
        data, y_true = prefetcher.next()
        # print(y_true)
        while data is not None and batchind < args.max_batch:
            stime = time.time()
            # print(stime)
            sumcnt += len(data)
            # print(sumcnt)

            ''' ↓ the implementation of RFM ↓ '''
            model.eval()#不启用 BatchNormalization 和 Dropout，保证BN和dropout不发生变化，pytorch框架会自动把BN和Dropout固定住，不会取平均，而是用训练好的值，不然的话，一旦test的batch_size过小，很容易就会被BN层影响结果。
            mask = cal_fam(model, data)#返回一张batch的所有像素的差值
            imgmask = torch.ones_like(mask)#返回一个填充了标量值1的张量
            imgh = imgw = 224

            for i in range(len(mask)):#遍历每一张图片，擦除
                maxind = np.argsort(mask[i].cpu().numpy().flatten())[::-1]#numpy.argsort()函数对输入数组沿给定轴执行间接排序，并使用指定排序类型返回数据的索引数组
                pointcnt = 0                                               #应该是对可疑图的扰动值排序、降序。
                for pointind in maxind:
                    pointx = pointind//imgw
                    pointy = pointind % imgw   #得到像素点的坐标

                    if imgmask[i][0][pointx][pointy] == 1:

                        maskh = random.randint(1, args.eH)
                        maskw = random.randint(1, args.eW)

                        sh = random.randint(1, maskh)
                        sw = random.randint(1, maskw)#可疑擦除块的大小

                        top = max(pointx-sh, 0)
                        bot = min(pointx+(maskh-sh), imgh)
                        lef = max(pointy-sw, 0)
                        rig = min(pointy+(maskw-sw), imgw)#定义要擦除的区域

                        imgmask[i][:, top:bot, lef:rig] = torch.zeros_like(imgmask[i][:, top:bot, lef:rig])#擦除

                        pointcnt += 1
                        if pointcnt >= 3:#擦除三次
                            break
            data = imgmask * data + (1-imgmask) * (torch.rand_like(data)*2-1.)
            ''' ↑ the implementation of RFM ↑ '''
            model.train()
            y_pred = model.forward(data)
#正向传播
            loss = lossfunc(y_pred, y_true)
#计算损失
            flood = (loss-0.04).abs() + 0.04

            sumloss += loss.detach()*len(data)#计算所有的损失
            data, y_true = prefetcher.next()

            optim.zero_grad()  #把梯度置为0
            flood.backward()#更新梯度
            optim.step()#梯度被计算好之后，用这个函数来更新数据

            batchind += 1#记录训练次数
            print("Train %06d loss:%.5f avgloss:%.5f lr:%.6f time:%.4f" % (batchind, loss, sumloss/sumcnt, optim.param_groups[0]["lr"], time.time()-stime), end="\r")

            if batchind % args.logbatch == 0:#记录训练次数
                print()
                Log("epoch:%03d batch:%06d loss:%.5f avgloss:%.5f" % (e, batchind, loss, sumloss/sumcnt))

                loss_valid, y_true_valid, y_pred_valid = Eval(model, lossfunc, validdataloader)
                ap, acc, AUC, TPR_2, TPR_3, TPR_4 = calRes(y_true_valid, y_pred_valid)
                Log("AUC:%.6f TPR_2:%.6f TPR_3:%.6f TPR_4:%.6f %s" % (AUC, TPR_2, TPR_3, TPR_4, "validset"))#输出信息

                loss_r, y_true_r, y_pred_r = Eval(model, lossfunc, testdataloaderR)
                sumAUC = sumTPR_2 = sumTPR_3 = sumTPR_4 = 0
                for i, tmptestdataloader in enumerate(testdataloaderList):
                    loss_f, y_true_f, y_pred_f = Eval(model, lossfunc, tmptestdataloader)
                    ap, acc, AUC, TPR_2, TPR_3, TPR_4 = calRes(torch.cat((y_true_r, y_true_f)), torch.cat((y_pred_r, y_pred_f)))
                    sumAUC += AUC
                    sumTPR_2 += TPR_2
                    sumTPR_3 += TPR_3
                    sumTPR_4 += TPR_4
                    Log("AUC:%.6f TPR_2:%.6f TPR_3:%.6f TPR_4:%.6f %s" % (AUC, TPR_2, TPR_3, TPR_4, TestsetName[i]))
                if len(testdataloaderList) > 1:
                    Log("AUC:%.6f TPR_2:%.6f TPR_3:%.6f TPR_4:%.6f Test" %
                        (sumAUC/len(testdataloaderList), sumTPR_2/len(testdataloaderList), sumTPR_3/len(testdataloaderList), sumTPR_4/len(testdataloaderList)))
                    TPR_4 = (sumTPR_4)/len(testdataloaderList)
                    MAX_TPR_4 = TPR_4
                if batchind % args.savebatch == 0 or TPR_4 > MAX_TPR_4:#训练多少次保存一次模型的参数信息
                    # MAX_TPR_4 = TPR_4
                    if args.save_model:
                        torch.save(model.state_dict(), "models/" + upper+"_"+modelname+"_model_batch_"+str(batchind))
                    if args.save_optim:
                        torch.save(optim.state_dict(), "models/" + upper+"_"+modelname+"_optim_batch_"+str(batchind))

                print("-------------------------------------------")
        e += 1