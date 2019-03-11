from adaboost import *
#测试文件
dataArr,labelArr=loadSimpData()
classifierArr=adaBoostTrains(dataArr,labelArr,30)
a=adaClassify([0,0],classifierArr)
print("数据类别是："+str(a))