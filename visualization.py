import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def visualize_titanic_data():
    """
    可视化Titanic数据集的分布和特征关系
    """
    print("开始数据可视化分析...")
    
    # 设置中文字体和样式
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_style("whitegrid")
    
    # 加载数据
    data_path = r'd:\实训项目\data\Titanic生存率\mytrain.csv'
    df = pd.read_csv(data_path)
    
    # 创建输出目录
    output_dir = r'd:\实训项目\data\可视化分析'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"数据加载完成，共 {len(df)} 条记录")
    
    # 创建综合图表
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Titanic 数据集分析报告', fontsize=20, fontweight='bold')
    
    # 1. 生存率分布
    ax1 = axes[0, 0]
    survival_counts = df['Survived'].value_counts()
    colors = ['#ff6b6b', '#51cf66']
    bars = ax1.bar(['Not Survived', 'Survived'], [survival_counts[0], survival_counts[1]], color=colors)
    ax1.set_title('Overall Survival Rate', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count')
    for bar, count in zip(bars, [survival_counts[0], survival_counts[1]]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'{count}\n({count/len(df)*100:.1f}%)', ha='center', fontsize=10)
    
    # 2. 性别与生存率
    ax2 = axes[0, 1]
    sex_survival = df.groupby('Sex')['Survived'].mean() * 100
    colors_sex = ['#339af0', '#f06595']
    bars = ax2.bar(['Female', 'Male'], [sex_survival['female'], sex_survival['male']], color=colors_sex)
    ax2.set_title('Survival Rate by Gender', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Survival Rate (%)')
    for bar, rate in zip(bars, [sex_survival['female'], sex_survival['male']]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{rate:.1f}%', ha='center', fontsize=10, fontweight='bold')
    
    # 3. 客舱等级与生存率
    ax3 = axes[0, 2]
    pclass_survival = df.groupby('Pclass')['Survived'].mean() * 100
    colors_pclass = ['#ffd43b', '#fab005', '#e67700']
    bars = ax3.bar(['1st Class', '2nd Class', '3rd Class'], 
                   [pclass_survival[1], pclass_survival[2], pclass_survival[3]], 
                   color=colors_pclass)
    ax3.set_title('Survival Rate by Passenger Class', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Survival Rate (%)')
    for bar, rate in zip(bars, [pclass_survival[1], pclass_survival[2], pclass_survival[3]]):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{rate:.1f}%', ha='center', fontsize=10, fontweight='bold')
    
    # 4. 年龄分布
    ax4 = axes[1, 0]
    df['Age'].hist(bins=30, ax=ax4, color='#74c0fc', edgecolor='black', alpha=0.7)
    ax4.set_title('Age Distribution', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Age')
    ax4.set_ylabel('Frequency')
    ax4.axvline(df['Age'].median(), color='red', linestyle='--', linewidth=2, label=f'Median: {df["Age"].median():.1f}')
    ax4.legend()
    
    # 5. 年龄与生存
    ax5 = axes[1, 1]
    survived = df[df['Survived'] == 1]['Age']
    not_survived = df[df['Survived'] == 0]['Age']
    ax5.hist([survived, not_survived], bins=30, label=['Survived', 'Not Survived'], 
             color=['#51cf66', '#ff6b6b'], alpha=0.7, edgecolor='black')
    ax5.set_title('Age Distribution by Survival', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Age')
    ax5.set_ylabel('Frequency')
    ax5.legend()
    
    # 6. 票价分布
    ax6 = axes[1, 2]
    df['Fare'].hist(bins=50, ax=ax6, color='#ffd43b', edgecolor='black', alpha=0.7)
    ax6.set_title('Fare Distribution', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Fare')
    ax6.set_ylabel('Frequency')
    ax6.axvline(df['Fare'].median(), color='red', linestyle='--', linewidth=2, 
               label=f'Median: {df["Fare"].median():.2f}')
    ax6.legend()
    
    # 7. 登船港口与生存率
    ax7 = axes[2, 0]
    embarked_survival = df.groupby('Embarked')['Survived'].mean() * 100
    colors_embarked = ['#845ef7', '#20c997', '#339af0']
    embark_mapping = {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}
    bars = ax7.bar([embark_mapping[x] for x in embarked_survival.index], 
                   embarked_survival.values, color=colors_embarked)
    ax7.set_title('Survival Rate by Embarked Port', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Survival Rate (%)')
    for bar, rate in zip(bars, embarked_survival.values):
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{rate:.1f}%', ha='center', fontsize=10, fontweight='bold')
    
    # 8. 家庭规模与生存
    ax8 = axes[2, 1]
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    family_survival = df.groupby('FamilySize')['Survived'].mean() * 100
    ax8.bar(family_survival.index, family_survival.values, color='#20c997', edgecolor='black')
    ax8.set_title('Survival Rate by Family Size', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Family Size')
    ax8.set_ylabel('Survival Rate (%)')
    
    # 9. 性别和客舱等级的交互效应
    ax9 = axes[2, 2]
    pivot_table = df.pivot_table(values='Survived', index='Sex', columns='Pclass', aggfunc='mean') * 100
    pivot_table.index = ['Female', 'Male']
    pivot_table.columns = ['1st Class', '2nd Class', '3rd Class']
    sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax9, 
                cbar_kws={'label': 'Survival Rate (%)'}, linewidths=0.5)
    ax9.set_title('Survival Rate by Gender & Class', fontsize=12, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 保存图表
    output_path = os.path.join(output_dir, 'titanic_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"综合分析图表已保存至: {output_path}")
    
    # 创建相关性热力图
    fig2, ax = plt.subplots(figsize=(12, 8))
    
    # 准备相关性分析数据
    df_corr = df.copy()
    df_corr['Sex'] = df_corr['Sex'].map({'male': 0, 'female': 1})
    df_corr['Embarked'] = df_corr['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    correlation_matrix = df_corr[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].corr()
    
    sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    
    corr_output_path = os.path.join(output_dir, 'correlation_matrix.png')
    plt.savefig(corr_output_path, dpi=300, bbox_inches='tight')
    print(f"相关性矩阵已保存至: {corr_output_path}")
    
    # 生成统计分析报告
    print("\n" + "=" * 60)
    print("关键发现总结:")
    print("=" * 60)
    
    print("\n1. 生存率分析:")
    print(f"   - 整体生存率: {df['Survived'].mean()*100:.2f}%")
    print(f"   - 女性生存率: {df[df['Sex'] == 'female']['Survived'].mean()*100:.2f}%")
    print(f"   - 男性生存率: {df[df['Sex'] == 'male']['Survived'].mean()*100:.2f}%")
    
    print("\n2. 客舱等级影响:")
    for pclass in [1, 2, 3]:
        survival_rate = df[df['Pclass'] == pclass]['Survived'].mean() * 100
        print(f"   - {pclass}等舱生存率: {survival_rate:.2f}%")
    
    print("\n3. 年龄影响:")
    children = df[df['Age'] < 15]['Survived'].mean() * 100
    adults = df[(df['Age'] >= 15) & (df['Age'] < 60)]['Survived'].mean() * 100
    elderly = df[df['Age'] >= 60]['Survived'].mean() * 100
    print(f"   - 儿童(0-15岁)生存率: {children:.2f}%")
    print(f"   - 成年人(15-60岁)生存率: {adults:.2f}%")
    print(f"   - 老年人(60岁以上)生存率: {elderly:.2f}%")
    
    print("\n4. 票价影响:")
    high_fare = df[df['Fare'] > 50]['Survived'].mean() * 100
    low_fare = df[df['Fare'] <= 50]['Survived'].mean() * 100
    print(f"   - 高票价(>50)生存率: {high_fare:.2f}%")
    print(f"   - 低票价(<=50)生存率: {low_fare:.2f}%")
    
    print("\n5. 家庭规模影响:")
    alone = df[df['FamilySize'] == 1]['Survived'].mean() * 100
    small_family = df[(df['FamilySize'] >= 2) & (df['FamilySize'] <= 4)]['Survived'].mean() * 100
    large_family = df[df['FamilySize'] > 4]['Survived'].mean() * 100
    print(f"   - 独自旅行生存率: {alone:.2f}%")
    print(f"   - 小家庭(2-4人)生存率: {small_family:.2f}%")
    print(f"   - 大家庭(>4人)生存率: {large_family:.2f}%")
    
    print("\n" + "=" * 60)
    print("数据可视化完成!")
    print("=" * 60)


if __name__ == "__main__":
    visualize_titanic_data()
