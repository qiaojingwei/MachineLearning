# coding:utf-8
'''
    导入所需要的api，共五步，对应5个api
    1、获取数据
    2、数据基本处理
    2.1、数据集分割
    3、特征工程-数据标准化
    4、机器学习-线性回归
    5、模型评估
'''

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor   # 采用梯度下降法
from sklearn.metrics import mean_squared_error


def linear_model2():
    '''
    线性回归：梯度下降法
    :return: None
    '''
    # 1、获取数据
    boston = load_boston()
    # print(data)

    # 2、数据处理
    # 2.1 数据集划分
    x_train,x_test,y_train,y_test= train_test_split(boston.data,boston.target,test_size=0.2)

    # 3、特征工程—标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)

    # 4、机器学习-线性回归（正规方程）
    # estimator = SGDRegressor(max_iter=1000,learning_rate="constant",eta0=0.001)  # 一般不会直接指定学习率为常数
    estimator = SGDRegressor(max_iter=1000)     # ok!已经默认选择了optimal
    estimator.fit(x_train,y_train)

    # 5 模型评估
    # 5.1 获取回归系数、截距等参数值
    y_predict= estimator.predict(x_test)
    print("预测值：\n",y_predict)
    print("回归系数：\n",estimator.coef_)
    print("截距：\n",estimator.intercept_)

    # 5.2 评价
    # 均方误差
    mse = mean_squared_error(y_test,y_predict)
    print("均方误差为:\n",mse)
    return None

linear_model2()