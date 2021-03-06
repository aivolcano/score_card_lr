
### 任务目标：基于逻辑回归的评分卡模型开发krjrick模型/ BCG矩阵

* 评分卡原理支持
评分卡的核心是：给用户身上的标签汇总出一个分数。用户画像提供多个标签（特征），这些标签是散乱，评分卡则是制定出一个规则，把用户画像放在规则之下，计算出一个评分。相当于商品上所有的卖点最终都以价格（一个数字）来体现出该商品的价值。

* 2个评分卡意味着2套规则，可以建立2个逻辑回归，分属x轴和y轴

![image](https://user-images.githubusercontent.com/68730894/115492023-96461680-a293-11eb-89eb-0bf3af3d7d15.png)

参考了Kraljic采购定位模型，分别以供应风险程度和利润额为label，使用2个逻辑回归，再使用供应风险评分卡评分的中位数和采购金额评分的中位树为x轴和y轴，这样整个空间分为4个部分，对应[ 战略物资 ]、[ 杠杆物资 ]、[ 瓶颈物资 ]、 [ 一般物资 ]。

![image](https://user-images.githubusercontent.com/68730894/115150100-7ea33e00-a099-11eb-93aa-74b5e8623623.png)

### 开发步骤
* 使用WOE对特征进行分箱
* 通过IV值筛选变量
IV值介于0.1到0.8之间，如果超过0.8该变量的区分度不强。IV值通过woe计算得到。
* 计算筛选后变量的woe方便分箱
* 得到WOE规则
* 使用WOE+LR建模
f1_score = 0.9562; accuracy_score = 0.9555
* 调参优化：网格调参优化
* 生成评分卡generator_score_card
通过model.coef_转为评分卡，假设p为风险（risk=1）的概率，那么正常（label=0）的概率为1-p。

此时，优势比：odds = p / (1-p)

风险概率 p = odds / (1 + odds), 逻辑回归 p = 1 / (1 + e^(-θ^Tx))


### 数据集是我们自己生成的数据集
与风险有关的特征是risk1 - risk24，risk_label是0-1特征；利润有关的特征是money1 - money24，money_label是0-1特征

我上传了我使用的数据集materical_group.xlsx，也可以根据material_group_data_generate.ipynb 和 赋予数据规律性.ipynb生成自己的专属数据集

我建立了money利润评分卡 和 risk 风险评分卡 两个逻辑回归模型



