# -*- coding: utf-8 -*-
# @Description  : 评分卡模型，数据集：SupplyChain.csv;SupplyChain_Description.csv;字典.xlsx
# @Author       : chenyancan
# @Software     :PyCharm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import numpy as np
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score

# 计算IV，衡量自变量的预测能力
def cal_IV(df, feature, target):
    lst = []
    cols = ['Variable', 'Value', 'All', 'Bad']
    for i in range(df[feature].nunique()):  # unnique = unique的个数
        val = list(df[feature].unique())[i]
        # 统计feature,feature_value,这个value的个数，这个value导致target=1的个数
        temp1 = df[df[feature] == val].count()[feature]
        temp2 = df[(df[feature] == val) & df[target] == 1].count()[feature]
        lst.append([feature, val, temp1, temp2])
    # 计算字段
    data = pd.DataFrame(lst, columns=cols)
    data = data[data['Bad'] > 0]
    data['Share'] = data['All'] / data['All'].sum()
    data['Bad Rate'] = data['Bad'] / data['All']  # 这个value导致bad情况，在这个value个数的比例
    data['Distribution Bad'] = data['Bad'] / data['Bad'].sum()  # 这个value导致Bad 在所有Bad中的比例
    data['Distribution Good'] = (data['All'] - data['Bad']) / (data['All'].sum() - data['Bad'].sum())
    data['WOE'] = np.log(data['Distribution Bad'] / data['Distribution Good'])
    # data['IV'] = ((data['Distribution Bad'] - data['Distribution Good']) * data['WOE']).sum()
    data['IV'] = (data['Distribution Bad'] - data['Distribution Good']) * data['WOE']

    data = data.sort_values(by=['Variable', 'Value'], ascending=True)
    #     print(data)
    #     print(data['IV'].sum())
    return data['IV'].sum()

# 计算单个特征的woe值
def cal_WOE(df, feature, target):
    df_new = df
    df_woe = df_new.groupby(feature).agg({target:['sum','count']})
    df_woe.columns = list(map(''.join,df_woe.columns.values))
#         print(df_woe.columns)
    df_woe = df_woe.reset_index()
    df_woe = df_woe.rename(columns={target+'sum':'bad',target+'count': 'all'})

    df_woe['good'] = df_woe['all'] - df_woe['bad']
    df_woe = df_woe[[feature,'good','bad']]
    df_woe['bad_rate'] = df_woe['bad'] / df_woe['bad'].sum()
    df_woe['good_rate'] = df_woe['good'] / df_woe['good'].sum()
    # 计算woe
    df_woe['woe'] = np.log1p(df_woe['bad_rate'].divide(df_woe['good_rate']))
#     print(df_woe)
    # 拼接
    df_new = df_new.merge(df_woe, on =feature, how='left')
    return df_new

#计算WOE值
# def cal_WOE(df, features, target):
#     for feature in features:
#         df_woe = df.groupby(feature).agg({target:['sum','count']})
#         df_woe.columns = list(map(''.join,df_woe.columns.values))
#         df_woe = df_woe.reset_index()
#         df_woe = df_woe.rename(columns={target+'sum':'bad',target+'count': 'all'})
#
#         df_woe['good'] = df_woe['all'] - df_woe['bad']
#         df_woe = df_woe[[feature,'good','bad']]
#         df_woe['bad_rate'] = df_woe['bad'] / df_woe['bad'].sum()
#         df_woe['good_rate'] = df_woe['good'] / df_woe['good'].sum()
#         # 计算woe
#         # df_woe['woe'] = np.log(df_woe['bad_rate'].divide(df_woe['good_rate']))
#         df_woe['woe'] = np.log1p(df_woe['bad_rate']/df_woe['good_rate'])
#         #     print(df_woe)
#         #在后面拼接上 _feature,比如age
#         df_woe.columns = [c if c == feature else c + '_' + feature for c in list(df_woe.columns.values)]
#         # 拼接
#         df = df.merge(df_woe, on =feature, how='left')
#     return df


'''
    分类模型指标测试
'''
from sklearn.metrics import accuracy_score,recall_score,f1_score, confusion_matrix,r2_score
def model_stats_classify(model, X, x_train, x_test, y_train, y_test, name='Fraud'):
    print('Model used:', model)
    model = model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_pred, y_test)
    recall = recall_score(y_pred, y_test)  # 测试二分类使用
    condusion = confusion_matrix(y_pred, y_test)
    f1 = f1_score(y_test, y_pred)  # 测试二分类使用
    # r2 = r2_score(y_pred, y_test)

    # 添加表头
    print('{}-Accuray:{}%'.format(name, accuracy * 100))
    print('{}-Recall: {}%'.format(name, recall * 100))
    print('{}-Confusion Matrix:\n {}'.format(name, condusion))
    print('{}-F1 score:{}%'.format(name, f1 * 100))

    # result = pd.DataFrame({'模型名称': str(model),
    #                        '预测打分(%)': accuracy * 100,
    #                        '召回率(Recall)%': recall * 100,
    #                        'L1范数正则化(F1 score)%': f1 * 100,
    #                        '重要指标排名(feat_importance)': '',
    #                        '最佳积分(best_score)': ''},
    #                       index=[1])  # 自定义索引为：1 ，这里也可以不设置index

    result = pd.DataFrame({'模型名称': str(model),
                           '预测打分(%)': accuracy * 100,
                           '召回率(Recall)%': recall * 100,
                           'L1范数正则化(F1 score)%': f1 * 100,
                           '重要指标排名(feat_importance)': ''},
                          index=[1])  # 自定义索引为：1 ，这里也可以不设置index
    # 非通用指标
    try:
        important_col = model.feature_importances_.argsort()
        print('{} model.feature_importances_:{}%'.format(name, important_col))
        feat_importance = pd.DataFrame(
            {'Variables': X.columns[important_col], 'importance': model.feature_importances_[important_col]})
        plt.figure(figsize=(20, 10))
        result['重要指标排名(feat_importance)'] = feat_importance
        sns.catplot(x='Variables', y='importance', data=feat_importance, height=5, aspect=2, kind='bar')
        plt.xticks(rotation=90)
        plt.show()
    except:
        print("\033[1;34m 本模型不含feature_importances_\033[0m")
    # try:
    #     result['最佳积分(best_score)'] = model.best_score
    #     print('{} model.best_score:{}%'.format(name, model.best_score))
    # except:
    #     print("\033[1;34m 本模型不含best_score\033[0m")
    return result

'''
    分类模型执行
'''
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC,LinearSVC
import xgboost as xgb
import lightgbm as lgb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
def fit_models(X,y,name='Fraud',type='classify'):
    model_result_dfs = pd.DataFrame()
    # 切割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print('X_train:{}, X_test:{}, y_train:{}, y_test:{}'.format(X_train.shape,X_test.shape,y_train.shape,y_test.shape))

    # 数据规范化，全部映射到-1~1之间
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    if type == 'classify': #分类
        models = ['LogisticRegression','GaussianNB', 'BernoulliNB', 'LinearSVC','LinearDiscriminantAnalysis','DecisionTreeClassifier','RandomForestClassifier',\
                 'xgb.XGBClassifier','lgb.LGBMClassifier']
        for model_name in models:
            # 模型定义
            model = eval(model_name + '()')
            # 指标计算
            model_report_df = model_stats_classify(model, X, X_train, X_test, y_train, y_test, name)
            model_result_dfs = model_result_dfs.append(model_report_df,ignore_index=True)
    # else:#回归
    #     models = regressor_models()
    #     for i,model in enumerate(models):
    #         model_report_df = model_stats_regressor(model, X, X_train, X_test, y_train, y_test, name)
    #         model_result_dfs = model_result_dfs.append(model_report_df)
    return model_result_dfs

class ScoreCard:
    '''
        初始化评分卡基准线
    '''
    def __init__(self,A,B):
        self.df_bin_to_woe = None
        self.A = A
        self.B = B
    '''
        加载原始数据集
    '''
    def load_data(self,path):
        self.df_train = pd.read_csv(path)
        self.df_train = self.df_train.iloc[:, 1:]
        df_train = self.df_train
        sns.countplot(x='SeriousDlqin2yrs', data=df_train)
        print('违约率:{}'.format(df_train['SeriousDlqin2yrs'].sum() / len(df_train)))
    '''
        数据清洗，数据填充
    '''
    def data_clearn(self):
        # 查看缺失值
        null_num = self.df_train.isnull().sum()
        null_num_rate = pd.DataFrame({'列名':null_num.index,'缺失值个数':null_num.values, '比例':null_num.values / len(self.df_train)})
        print(null_num_rate)
        # 使用中位数对缺失值进行填充
        self.df_train = self.df_train.fillna(self.df_train.median())

        # 数据探索
        print(self.df_train['RevolvingUtilizationOfUnsecuredLines'].describe())
        # 直方图探索
        sns.distplot(self.df_train['RevolvingUtilizationOfUnsecuredLines'])
    '''
        数据分箱
    '''
    def data_binning(self):
        # 对年龄进行分箱
        age_bins = [-math.inf, 25, 40, 50, 60, 70, math.inf]
        self.df_train['bin_age'] = pd.cut(self.df_train['age'], bins=age_bins)
        self.df_train[['age', 'bin_age']]

        # 家庭成员数 NumberOfDependents
        NumberOfDependents_bin = [-math.inf, 2, 4, 6, 8, math.inf]
        self.df_train['bin_NumberOfDependents'] = pd.cut(self.df_train['NumberOfDependents'], bins=NumberOfDependents_bin)
        self.df_train[['NumberOfDependents', 'bin_NumberOfDependents']]

        # 对于3种逾期次数，即NumberOfTime30-59DaysPastDueNotWorse，NumberOfTime60-89DaysPastDueNotWorse，NumberOfTimes90DaysLate，分成10段
        dpd_bins = [-math.inf, 1, 2, 3, 4, 5, 6, 7, 8, 9, math.inf]
        self.df_train['bin_NumberOfTime30-59DaysPastDueNotWorse'] = pd.cut(self.df_train['NumberOfTime30-59DaysPastDueNotWorse'],
                                                                      bins=NumberOfDependents_bin)
        self.df_train['bin_NumberOfTime60-89DaysPastDueNotWorse'] = pd.cut(self.df_train['NumberOfTime60-89DaysPastDueNotWorse'],
                                                                      bins=NumberOfDependents_bin)
        self.df_train['bin_NumberOfTimes90DaysLate'] = pd.cut(self.df_train['NumberOfTimes90DaysLate'],
                                                         bins=NumberOfDependents_bin)

        # RevolvingUtilizationOfUnsecuredLines, DebtRatio, MonthlyIncome, NumberOfOpenCreditLinesAndLoans, NumberRealEstateLoansOrLines
        # 等频5段 0-20%分位数，20%-40%分位数
        # 如果遇到相同的数据吗，横跨2个箱子，那么就去掉一个
        self.df_train['bin_RevolvingUtilizationOfUnsecuredLines'] = pd.qcut(self.df_train['RevolvingUtilizationOfUnsecuredLines'],
                                                                       q=5, duplicates='drop')
        self.df_train['bin_DebtRatio'] = pd.qcut(self.df_train['DebtRatio'], q=5, duplicates='drop')
        self.df_train['bin_MonthlyIncome'] = pd.qcut(self.df_train['MonthlyIncome'], q=5, duplicates='drop')
        self.df_train['bin_NumberOfOpenCreditLinesAndLoans'] = pd.qcut(self.df_train['NumberOfOpenCreditLinesAndLoans'], q=5,
                                                                  duplicates='drop')
        self.df_train['bin_NumberRealEstateLoansOrLines'] = pd.qcut(self.df_train['NumberRealEstateLoansOrLines'], q=5,
                                                               duplicates='drop')
        print(self.df_train[['bin_RevolvingUtilizationOfUnsecuredLines', 'bin_DebtRatio', 'bin_MonthlyIncome',
                        'bin_NumberOfOpenCreditLinesAndLoans', 'bin_NumberRealEstateLoansOrLines']])
        print(self.df_train['bin_NumberRealEstateLoansOrLines'].value_counts())
        print(self.df_train['NumberRealEstateLoansOrLines'].value_counts())
        # 统计分箱字段
        bin_cols = [c for c in self.df_train.columns.values if c.startswith('bin_')]
        print('分箱字段:{}'.format(bin_cols))
    '''
        通过IV值和WOE值筛选特征
    '''
    def getFeature(self):
        # 查看指标SeriousDlqin2yrs对target(bin_age)的影响
        # cal_IV(self.df_train, 'bin_age', 'SeriousDlqin2yrs')

        # 统计分箱字段
        bin_cols = [c for c in self.df_train.columns.values if c.startswith('bin_')]
        # bin_cols

        # 通过IV寻找所有合适的指标项
        select_bin_features = []
        for col in bin_cols:
            temp_IV = cal_IV(self.df_train, col, 'SeriousDlqin2yrs')
            if temp_IV > 0.1 : #and temp_IV <= 0.5:
                print('字段:{},IV值:{}'.format(col, temp_IV))
                select_bin_features.append(col)
        self.woe_train = self.df_train

        # woe特征追加
        woe_select_feature = []
        for i, select_bin_feature in enumerate(select_bin_features):
            self.woe_train['woe_' + select_bin_feature] = cal_WOE(self.df_train, select_bin_feature, 'SeriousDlqin2yrs')['woe']
            # 返回特征值数组
            woe_select_feature.append('woe_' + select_bin_feature)
        # 返回X,y

        # feature,bin,woe值列表
        self.df_bin_to_woe = pd.DataFrame(columns=['features', 'bin', 'woe'])
        self.select_features = []
        for f in select_bin_features:
            f = f[4:]
            self.select_features.append(f)
            b = 'bin_' + f
            w = 'woe_bin_' + f
            # 对于每个feature字段找到相应的bin和woe
            df = self.df_train[[w, b]].drop_duplicates()
            print(df)
            df.columns = ['woe', 'bin']
            df['features'] = f
            df = df[['features', 'bin', 'woe']]
            self.df_bin_to_woe = pd.concat([self.df_bin_to_woe, df])

        return self.woe_train[woe_select_feature],self.df_train['SeriousDlqin2yrs']
    '''
        简单逻辑回归模型和贝叶斯优化
    '''
    def logistic_by_bys(self,x_train,y_train):
        # best_param = {'target': 0.16283227307124037, 'params': {'C': 6.8910345173078795, 'max_iter': 356.66284309481864}}
        # best_param_C = best_param['params']['C']
        # best_param_max_iter = best_param['params']['max_iter']
        # print('best_param_C:{};best_param_max_iter:{}'.format(best_param_C,best_param_max_iter))
        # cross_val_score(LogisticRegression(C=best_param_C, max_iter=best_param_max_iter), x_train, y_train, scoring='f1', cv=5).mean()

        def lr_cv(max_iter, C):
            result = cross_val_score(LogisticRegression(C=C, max_iter=max_iter), x_train, y_train, scoring='f1',
                                     cv=5).mean()
            return result
        # 使用贝叶斯超参数优化
        lr_op = BayesianOptimization(
            f=lr_cv,
            pbounds={'C': (0.01, 10), 'max_iter': (50, 500)}
        )
        lr_op.maximize()
        best_param = lr_op.max
        best_param_C = best_param['params']['C']
        best_param_max_iter = best_param['params']['max_iter']
        # 模型拟合和coef值
        # best_param_C = 6.8910345173078795
        # best_param_max_iter = 356.66284309481864
        model = LogisticRegression(random_state=33, class_weight='balanced', max_iter=best_param_max_iter, C=best_param_C)
        model.fit(x_train, y_train)
        return model

    '''
        评分卡模型转换
    '''
    def generate_soorecard(self,model_coef):
        binning_df = self.df_bin_to_woe
        features = self.select_features
        lst = []
        cols = ['Variable', 'Binning', 'Score']
        # 模型系数
        coef = model_coef[0]
        for i in range(len(features)):
            f = features[i]
            # 得到这个feature的WOE规则
            df = binning_df[binning_df['features'] == f]
            for index, row in df.iterrows():
                lst.append([f, row['bin'], int(round(-coef[i] * row['woe'] * self.B))])
        data = pd.DataFrame(lst, columns=cols)
        return data
    '''
        评分卡模型
    '''
    def cal_score(self,df, score_card):
        # map_to_score 按照评分卡规则进行计算
        df['score'] = df.apply(map_to_score, args=(score_card,), axis=1)
        df['score'] = df['score'].astype(int)
        df['score'] = df['score'] + self.A
        return df
    '''
        查看信用好坏个人
        @:param person = 0,比较好的5个人; person = 1,比较差的5个人
    '''
    def simple_query(self,num=5,person=0):
        # 随机选择Good的5个人
        result = self.df_train[self.df_train['SeriousDlqin2yrs'] == person].sample(num)
        # df_woe[woe_cols]
        result = result[self.select_features]
        # 随机选的人，进行评分
        return self.cal_score(result, score_card)

# 把数据映射到分箱中
def str_to_int(s):
    if s=='-inf':
        return -999999
    if s=='inf':
        return 999999
    return float(s)


# 讲value影响到bin
def map_value_to_bin(feature_value, frature_to_bin):
    for index, row in frature_to_bin.iterrows():
        bins = str(row['Binning'])
        left_open = bins[0] == '('
        right_open = bins[-1] == ')'
        binnings = bins[1:-1].split(',')
        in_range = True
        temp = str_to_int(binnings[0])
        temp2 = str_to_int(binnings[1])
        # 检查左括号
        if left_open:
            if feature_value <= temp:
                in_range = False
        else:
            if feature_value <= temp:
                in_range = False

        # 检查右括号
        if right_open:
            if feature_value >= temp2:
                in_range = False
        else:
            if feature_value > temp2:
                in_range = False
        if in_range:
            return row['Binning']

# df 待转换的样本，score_card 评分卡规则
def map_to_score(df,score_card):
    scored_columns = list(score_card['Variable'].unique())
    score = 0
    for col in scored_columns:
        # 取出评分规则
        feature_to_bin = score_card[score_card['Variable'] == col]
        # 取出具体的feature_value
        feature_value = df[col]
        selected_bin = map_value_to_bin(feature_value,feature_to_bin)
        temp_score = feature_to_bin[feature_to_bin['Binning'] == selected_bin]
        score += temp_score['Score'].values[0]
    return score



'''
    1 通过IV值和WOE值筛选特征
    2 使用逻辑回归及其他模型对是否会逾期90天以上进行预测（二分类问题）
'''
if __name__ == "__main__":
    A = 650  # 基础分，业内常见的500,600,650
    # B =72.13
    B = 20  # 刻度，业内常见20,50
    scoreCardTest = ScoreCard(A,B)
    # 数据加载
    scoreCardTest.load_data('test_data/cs-training.csv')
    # 数据清洗
    scoreCardTest.data_clearn()
    # 数据分箱
    scoreCardTest.data_binning()
    # 特征抽取
    X,y = scoreCardTest.getFeature()
    # 逻辑回归交叉熵+贝叶斯调参
    model = scoreCardTest.logistic_by_bys(X,y)
    # 评分卡
    score_card = scoreCardTest.generate_soorecard(model.coef_)
    # 高分的人
    good_person = scoreCardTest.simple_query(person = 0)
    # 低分的人
    bad_person = scoreCardTest.simple_query(person = 1)
    # 其他二分类预测对比（逻辑回归等）
    predict_report = fit_models(X,y,name='WOE模型预测违约')