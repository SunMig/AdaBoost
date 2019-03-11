from numpy import *
from boost import *
#简单的测试函数
def loadSimpData():
    dataMat=matrix([[1.,2.1],[2.,1.1],[1.3,1.],[1.,1.],[2.,1.]])
    classlabels=[1,1,-1,-1,1]
    return  dataMat,classlabels
#算法训练过程,返回的是弱分类器数组
def adaBoostTrains(dataArr,classLables,numIt=40):
    weakClassArr=[]
    m=shape(dataArr)[0]
    D=mat(ones((m,1))/m)#权重向量
    aggClassEst=mat(zeros((m,1))) #记录每个数据点的类别估计累计值
    for i in range(numIt):
        bestStump,error,classEst=buildStump(dataArr,classLables,D)#D是权重向量
        # print("D: ")
        # print(D.T)
        alpha=float(0.5*log((1.0-error)/max(error,1e-16))) #计算alpha值，用于更新权重向量
        bestStump['alpha']=alpha #每次循环把alpha值加入字典
        weakClassArr.append(bestStump) #把每次得到的弱分类器放入一个数组中
        # print("classEst: ")
        # print(classEst.T)
        expon=multiply(-1*alpha*mat(classLables).T,classEst)
        D=multiply(D,exp(expon))
        D=D/D.sum()#完成权重向量的更新
        aggClassEst+=alpha*classEst
        # print("aggClassEst: ")
        print(aggClassEst.T)
        aggErrors=multiply(sign(aggClassEst)!=mat(classLables).T,ones((m,1)))
        errorRote=aggErrors.sum()/m
        print("total error: "+str(errorRote))
        if errorRote==0.0:break
    return weakClassArr
#这里是AdaBoost算法的实现，输入分类数据和弱分类数组，对弱分类器进行加权求和
def adaClassify(datToClass,classifierArr):
    dataMatrix=mat(datToClass)
    m=shape(dataMatrix)[0]
    aggClassEst=mat(zeros((m,1))) #每个数据点的类别估计累计值
    for i in range(len(classifierArr)):
        classEst=stumpClassify(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        aggClassEst+=classifierArr[i]['alpha']*classEst
        print(aggClassEst)
    return sign(aggClassEst)