# Titanic 旅客生存概率预测系统

## 项目简介

基于深度学习的Titanic旅客生存概率预测系统，使用多层感知机(MLP/DNN)模型进行预测，提供友好的Web界面进行旅客信息输入、预测和查询。

## 技术栈

- **深度学习框架**: TensorFlow/Keras
- **机器学习工具**: scikit-learn
- **数据处理**: pandas, numpy
- **可视化**: matplotlib
- **Web界面**: Streamlit
- **数据库**: SQLite

## 模型说明

### 模型结构

使用多层感知机(MLP/DNN)，结构如下：

- **输入层**: 12个特征节点
- **隐藏层1**: 64个神经元，ReLU激活函数，BatchNormalization，Dropout(0.3)
- **隐藏层2**: 32个神经元，ReLU激活函数，BatchNormalization，Dropout(0.3)
- **输出层**: 1个神经元，Sigmoid激活函数

### 选择此模型的原因

1. **MLP结构简单但强大**：适合处理结构化数据
2. **两层隐藏层**：足够捕捉特征间的非线性关系
3. **ReLU激活函数**：解决梯度消失问题，加速收敛
4. **BatchNormalization**：加速训练并提高稳定性
5. **Dropout**：防止过拟合，提高泛化能力
6. **Sigmoid输出层**：适合二分类问题，输出概率值

## 项目结构

```
实训项目/
├── app.py                  # Streamlit Web应用主文件
├── titanic_model.py        # 深度学习模型定义和训练
├── data_preprocessing.py   # 数据预处理和特征工程
├── database.py             # SQLite数据库操作
├── config.py               # 项目配置文件
├── requirements.txt        # Python依赖包
├── data/                   # 数据目录
├── models/                 # 模型保存目录
└── titanic.db              # SQLite数据库文件
```

## 安装步骤

1. 安装Python依赖：

```bash
pip install -r requirements.txt
```

## 使用方法

### 方式一：直接运行Streamlit应用（推荐）

```bash
streamlit run app.py
```

首次运行时，系统会自动下载数据并训练模型。

### 方式二：先训练模型再运行

```bash
# 训练模型
python titanic_model.py

# 运行应用
streamlit run app.py
```

## 功能说明

### 1. 预测生存概率
- 输入旅客信息（客舱等级、姓名、性别、年龄等）
- 系统自动预测生存概率
- 结果保存到数据库

### 2. 查看旅客信息
- 查看所有预测过的旅客
- 筛选查看生存旅客
- 表格形式展示详细信息

### 3. 系统统计
- 总预测次数
- 生存/遇难人数统计
- 平均生存概率
- 可视化分析图表

## 特征工程

系统对原始数据进行了以下特征工程：

1. **提取称呼**：从姓名中提取Mr/Mrs/Miss等称呼
2. **家庭大小**：计算兄弟姐妹+父母小孩+1
3. **是否独自一人**：家庭大小为1时标记为1
4. **年龄分箱**：将年龄分为Child/Teen/YoungAdult/Adult/Senior
5. **票价分箱**：将票价分为Low/Medium/High/VeryHigh
6. **缺失值处理**：使用中位数/众数填充缺失值
7. **标签编码**：对分类特征进行编码
8. **标准化**：对数值特征进行标准化

## 预期精度

模型在测试集上的准确率通常不低于0.75，超过要求的0.70。

## 注意事项

- 首次运行需要下载训练数据和训练模型，可能需要几分钟
- 模型文件保存在 `models/` 目录下
- 数据库文件保存在 `titanic.db`
"# -" 
