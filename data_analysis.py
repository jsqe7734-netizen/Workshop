import pandas as pd
import os


def check_titanic_data():
    """
    检查Titanic数据集的质量和完整性
    """
    print("=" * 60)
    print("Titanic 数据集质量检查报告")
    print("=" * 60)
    
    data_dir = r'd:\实训项目\data\Titanic生存率'
    
    train_path = os.path.join(data_dir, 'mytrain.csv')
    test_path = os.path.join(data_dir, 'mytest.csv')
    
    # 检查文件是否存在
    print("\n1. 文件检查:")
    print(f"   训练数据文件: {'[OK]' if os.path.exists(train_path) else '[FAIL]'}")
    print(f"   测试数据文件: {'[OK]' if os.path.exists(test_path) else '[FAIL]'}")
    
    if not os.path.exists(train_path):
        print("\n错误: 训练数据文件不存在!")
        return
    
    # 加载数据
    train_df = pd.read_csv(train_path)
    print(f"\n2. 数据基本信息:")
    print(f"   训练集样本数: {len(train_df)}")
    print(f"   特征数量: {len(train_df.columns)}")
    print(f"   列名: {list(train_df.columns)}")
    
    # 数据类型检查
    print(f"\n3. 数据类型:")
    print(train_df.dtypes)
    
    # 缺失值检查
    print(f"\n4. 缺失值统计:")
    missing = train_df.isnull().sum()
    missing_pct = (train_df.isnull().sum() / len(train_df) * 100).round(2)
    
    for col in train_df.columns:
        if missing[col] > 0:
            print(f"   {col:12s}: {missing[col]:4d} ({missing_pct[col]:5.2f}%)")
        else:
            print(f"   {col:12s}: 0")
    
    # 数值特征统计
    print(f"\n5. 数值特征统计:")
    numerical_cols = ['Age', 'SibSp', 'Parch', 'Fare']
    print(train_df[numerical_cols].describe())
    
    # 分类特征统计
    print(f"\n6. 分类特征统计:")
    print(f"\n   Survived (生存情况):")
    print(train_df['Survived'].value_counts())
    print(f"   生存率: {train_df['Survived'].mean()*100:.2f}%")
    
    print(f"\n   Pclass (客舱等级):")
    print(train_df['Pclass'].value_counts().sort_index())
    
    print(f"\n   Sex (性别):")
    print(train_df['Sex'].value_counts())
    
    print(f"\n   Embarked (登船港口):")
    print(train_df['Embarked'].value_counts())
    
    # 特殊值检查
    print(f"\n7. 特殊值检查:")
    
    # 年龄分布
    print(f"   年龄范围: {train_df['Age'].min():.1f} - {train_df['Age'].max():.1f}")
    print(f"   年龄中位数: {train_df['Age'].median():.1f}")
    
    # 票价分布
    print(f"   票价范围: {train_df['Fare'].min():.2f} - {train_df['Fare'].max():.2f}")
    print(f"   票价中位数: {train_df['Fare'].median():.2f}")
    print(f"   票价均值: {train_df['Fare'].mean():.2f}")
    
    # 家庭规模
    train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
    print(f"\n   家庭规模分布:")
    print(train_df['FamilySize'].value_counts().sort_index())
    
    # 提取姓名中的称呼
    print(f"\n8. 姓名称呼分析:")
    titles = train_df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    print(titles.value_counts())
    
    # 舱号分析
    print(f"\n9. 船舱号分析:")
    cabin_count = train_df['Cabin'].notna().sum()
    print(f"   有船舱号记录: {cabin_count} ({cabin_count/len(train_df)*100:.2f}%)")
    
    if train_df['Cabin'].notna().any():
        # 提取舱位类型
        cabin_type = train_df['Cabin'].dropna().str[0]
        print(f"   舱位类型分布:")
        print(cabin_type.value_counts())
    
    print("\n" + "=" * 60)
    print("数据质量评估:")
    print("=" * 60)
    
    # 计算数据质量分数
    total_cells = train_df.shape[0] * train_df.shape[1]
    missing_cells = train_df.isnull().sum().sum()
    quality_score = (1 - missing_cells/total_cells) * 100
    
    print(f"\n数据完整度: {quality_score:.2f}%")
    
    if quality_score >= 95:
        print("数据质量: 优秀 [OK]")
    elif quality_score >= 85:
        print("数据质量: 良好")
    elif quality_score >= 70:
        print("数据质量: 一般，需要处理缺失值")
    else:
        print("数据质量: 较差，需要大量数据清洗")
    
    print("\n关键发现:")
    print("- Age列有较多缺失值，需要用中位数或均值填充")
    print("- Cabin列缺失严重，可能需要从模型中排除或特殊处理")
    print("- Embarked列有少量缺失，需要用众数填充")
    print("- 可以从Name中提取Title作为新特征")
    print("- FamilySize特征可以通过SibSp和Parch计算得出")
    
    print("\n" + "=" * 60)
    
    # 如果有测试集，也检查测试集
    if os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        print("\n测试集信息:")
        print(f"   测试集样本数: {len(test_df)}")
        print(f"   测试集缺失值:")
        test_missing = test_df.isnull().sum()
        for col in test_df.columns:
            if test_missing[col] > 0:
                print(f"      {col}: {test_missing[col]}")


if __name__ == "__main__":
    check_titanic_data()
