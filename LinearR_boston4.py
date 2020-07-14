# coding:utf-8

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,RidgeCV   # 采用岭回归
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib


def linear_model3():
    '''
    线性回归：岭回归
    :return: None
    '''
    # 1、获取数据
    boston = load_boston()
    # print(data)

    # 2、数据处理
    # 2.1 数据集划分
    x_train,x_test,y_train,y_test= train_test_split(boston.data,boston.target,random_state=2,test_size=0.2)  # random_state为None时，每次生成的数据都是随机的；为整数时，每次生成的数据都相同

    # 3、特征工程—标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)

    # # 4、机器学习-线性回归（正规方程）
    # # 4.1、模型训练
    # estimator = Ridge(alpha=1.0)
    # estimator.fit(x_train,y_train)
    # # 4.2、模型保存
    # joblib.dump(estimator,"./test.pkl")
    # 4.3、模型加载
    estimator = joblib.load("./test.pkl")

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

linear_model3()