import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from config import MODEL_DIR
from titanic_model import TitanicDNNModel, train_and_save_model
from database import TitanicDatabase
import time

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


@st.cache_resource
def load_model():
    model_path = os.path.join(MODEL_DIR, 'titanic_dnn_model.keras')
    preprocessor_path = os.path.join(MODEL_DIR, 'titanic_dnn_model_preprocessor.pkl')
    accuracy_path = os.path.join(MODEL_DIR, 'model_accuracy.txt')
    
    if os.path.exists(model_path) and os.path.exists(preprocessor_path):
        try:
            model = TitanicDNNModel.load(model_path, preprocessor_path)
            
            # 加载模型准确率
            model.accuracy = None
            if os.path.exists(accuracy_path):
                with open(accuracy_path, 'r') as f:
                    model.accuracy = float(f.read())
            
            return model
        except Exception as e:
            st.error(f"加载模型失败: {e}")
            return None
    return None


@st.cache_resource
def get_database():
    return TitanicDatabase()


def main():
    st.set_page_config(page_title="Titanic 生存预测系统", page_icon="🚢", layout="wide")
    
    st.title("🚢 Titanic 旅客生存概率预测系统")
    st.markdown("---")
    
    db = get_database()
    model = load_model()
    
    if model is None:
        st.warning("模型未找到，正在训练新模型...")
        with st.spinner("训练模型中，请稍候..."):
            try:
                model, accuracy = train_and_save_model()
                st.success(f"✓ 模型训练完成！准确率: {accuracy:.4f}")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"训练失败: {e}")
                return
    
    st.sidebar.title("导航")
    page = st.sidebar.radio("选择功能", [
        "预测生存概率", 
        "批量预测", 
        "查看旅客信息", 
        "系统统计",
        "数据质量检查",
        "数据可视化"
    ])
    
    if page == "预测生存概率":
        show_prediction_page(model, db)
    elif page == "批量预测":
        show_batch_prediction_page(model, db)
    elif page == "查看旅客信息":
        show_passengers_page(db)
    elif page == "系统统计":
        show_statistics_page(db, model)
    elif page == "数据质量检查":
        show_data_quality_page()
    else:
        show_visualization_page()


def show_prediction_page(model, db):
    st.header("🔮 旅客生存概率预测")
    st.markdown("请输入旅客信息，系统将预测其生存概率")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pclass = st.selectbox("客舱等级 (Pclass)", [1, 2, 3], index=2, help="1=一等舱, 2=二等舱, 3=三等舱")
        name = st.text_input("姓名 (Name)", value="Mr. John Doe")
        sex = st.selectbox("性别 (Sex)", ["male", "female"], index=0)
        age = st.number_input("年龄 (Age)", min_value=0.0, max_value=100.0, value=28.0, step=0.5)
        sibsp = st.number_input("堂兄弟/妹个数 (SibSp)", min_value=0, max_value=10, value=0, step=1)
    
    with col2:
        parch = st.number_input("父母与小孩个数 (Parch)", min_value=0, max_value=10, value=0, step=1)
        ticket = st.text_input("船票号 (Ticket)", value="A/5 21171")
        fare = st.number_input("船票价格 (Fare)", min_value=0.0, max_value=550.0, value=14.5, step=0.5)
        cabin = st.text_input("船舱号 (Cabin)", value="")
        embarked = st.selectbox("上船港口 (Embarked)", ["S", "C", "Q"], index=0, help="S=南安普顿, C=瑟堡, Q=皇后镇")
    
    if st.button("预测生存概率", type="primary"):
        with st.spinner("正在预测..."):
            passenger_data = {
                'Pclass': pclass,
                'Name': name,
                'Sex': sex,
                'Age': age,
                'SibSp': sibsp,
                'Parch': parch,
                'Ticket': ticket,
                'Fare': fare,
                'Cabin': cabin,
                'Embarked': embarked
            }
        
            df = pd.DataFrame([passenger_data])
            X = model.preprocessor.transform(df)
            survival_prob = model.predict(X)[0]
            
            time.sleep(0.5)
            
            st.markdown("---")
            st.subheader("预测结果")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                if survival_prob >= 0.5:
                    st.success(f"### ✅ 预测生存")
                    st.metric("生存概率", f"{survival_prob:.2%}")
                else:
                    st.error(f"### ❌ 预测遇难")
                    st.metric("生存概率", f"{survival_prob:.2%}")
            
            with col2:
                st.progress(float(survival_prob))
            
            db.add_passenger(passenger_data, float(survival_prob))
            st.success("✓ 旅客信息已保存到数据库")
            
            st.markdown("### 旅客信息")
            st.json(passenger_data)


def show_batch_prediction_page(model, db):
    st.header("📦 批量预测")
    st.markdown("上传CSV文件进行批量预测")
    
    # 上传文件
    uploaded_file = st.file_uploader("上传CSV文件", type="csv")
    
    if uploaded_file is not None:
        try:
            # 读取CSV文件
            df = pd.read_csv(uploaded_file)
            st.subheader("上传的数据")
            st.dataframe(df.head(10))
            st.write(f"共 {len(df)} 条记录")
            
            # 检查必要的列
            required_columns = ['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
            missing_cols = [col for col in required_columns if col not in df.columns]
            
            if missing_cols:
                st.error(f"缺少必要的列: {', '.join(missing_cols)}")
                st.info("CSV文件需要包含以下列: Pclass, Name, Sex, Age, SibSp, Parch, Fare, Embarked")
                return
            
            # 批量预测
            if st.button("开始批量预测", type="primary"):
                with st.spinner("正在批量预测..."):
                    # 预处理数据
                    X = model.preprocessor.transform(df)
                    
                    # 预测
                    probabilities = model.predict(X)
                    
                    # 添加预测结果
                    df['SurvivalProbability'] = probabilities
                    df['PredictedSurvived'] = (probabilities >= 0.5).astype(int)
                    
                    # 保存到数据库
                    saved_count = 0
                    for _, row in df.iterrows():
                        passenger_data = {
                            'Pclass': row['Pclass'],
                            'Name': row['Name'],
                            'Sex': row['Sex'],
                            'Age': row['Age'],
                            'SibSp': row['SibSp'],
                            'Parch': row['Parch'],
                            'Ticket': row.get('Ticket', ''),
                            'Fare': row['Fare'],
                            'Cabin': row.get('Cabin', ''),
                            'Embarked': row['Embarked']
                        }
                        db.add_passenger(passenger_data, float(row['SurvivalProbability']))
                        saved_count += 1
                    
                    time.sleep(0.5)
                    
                    st.success(f"✓ 批量预测完成！共处理 {len(df)} 条记录，已保存 {saved_count} 条到数据库")
                    
                    # 显示预测结果
                    st.subheader("预测结果")
                    df_display = df.copy()
                    df_display['SurvivalProbability'] = df_display['SurvivalProbability'].apply(lambda x: f"{x:.2%}")
                    df_display['PredictedSurvived'] = df_display['PredictedSurvived'].map({1: '✅ 生存', 0: '❌ 遇难'})
                    
                    st.dataframe(
                        df_display,
                        use_container_width=True,
                        column_config={
                            'Pclass': '客舱等级',
                            'Name': '姓名',
                            'Sex': '性别',
                            'Age': '年龄',
                            'SibSp': '兄弟姐妹数',
                            'Parch': '父母小孩数',
                            'Ticket': '船票号',
                            'Fare': '票价',
                            'Cabin': '船舱号',
                            'Embarked': '登船港口',
                            'SurvivalProbability': '生存概率',
                            'PredictedSurvived': '预测结果'
                        }
                    )
                    
                    # 统计信息
                    survived_count = df['PredictedSurvived'].sum()
                    not_survived_count = len(df) - survived_count
                    avg_prob = df['SurvivalProbability'].mean()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("总人数", len(df))
                    with col2:
                        st.metric("预测生存", survived_count)
                    with col3:
                        st.metric("平均生存概率", f"{avg_prob:.2%}")
            
        except Exception as e:
            st.error(f"处理失败: {e}")
    
    # 显示示例格式
    st.markdown("---")
    st.subheader("📝 CSV文件格式示例")
    sample_data = {
        'Pclass': [1, 2, 3],
        'Name': ['Smith, Mr. John', 'Jones, Mrs. Mary', 'Brown, Mr. William'],
        'Sex': ['male', 'female', 'male'],
        'Age': [30, 25, 45],
        'SibSp': [0, 1, 0],
        'Parch': [0, 0, 2],
        'Ticket': ['A/5 21171', 'PC 17599', '373450'],
        'Fare': [7.25, 71.2833, 8.05],
        'Cabin': ['', 'C85', ''],
        'Embarked': ['S', 'C', 'S']
    }
    sample_df = pd.DataFrame(sample_data)
    st.dataframe(sample_df)
    
    # 提供下载示例
    csv = sample_df.to_csv(index=False)
    st.download_button(
        label="下载示例CSV",
        data=csv,
        file_name='titanic_batch_sample.csv',
        mime='text/csv'
    )


def show_passengers_page(db):
    st.header("📋 旅客信息查询")
    
    view_option = st.radio("查看类型", ["所有旅客", "生存旅客"], horizontal=True)
    
    if view_option == "所有旅客":
        df = db.get_all_passengers()
    else:
        df = db.get_survived_passengers()
    
    if len(df) == 0:
        st.info("暂无旅客数据，请先进行预测")
    else:
        # 搜索功能
        st.subheader("🔍 搜索旅客")
        search_col1, search_col2, search_col3 = st.columns([2, 1, 1])
        
        with search_col1:
            search_name = st.text_input("按姓名搜索", placeholder="输入旅客姓名...")
        
        with search_col2:
            search_pclass = st.selectbox("按客舱等级筛选", ["全部", 1, 2, 3])
        
        with search_col3:
            search_sex = st.selectbox("按性别筛选", ["全部", "male", "female"])
        
        # 应用筛选条件
        filtered_df = df.copy()
        
        if search_name:
            filtered_df = filtered_df[filtered_df['name'].str.contains(search_name, case=False, na=False)]
        
        if search_pclass != "全部":
            filtered_df = filtered_df[filtered_df['pclass'] == search_pclass]
        
        if search_sex != "全部":
            filtered_df = filtered_df[filtered_df['sex'] == search_sex]
        
        if len(filtered_df) == 0:
            st.warning("未找到匹配的旅客记录")
        else:
            df_display = filtered_df.copy()
            df_display['survival_probability'] = df_display['survival_probability'].apply(lambda x: f"{x:.2%}")
            df_display['predicted_survived'] = df_display['predicted_survived'].map({1: '✅ 生存', 0: '❌ 遇难'})
            
            # 显示数据表格
            st.dataframe(
                df_display,
                use_container_width=True,
                column_config={
                    'id': 'ID',
                    'pclass': '客舱等级',
                    'name': '姓名',
                    'sex': '性别',
                    'age': '年龄',
                    'sibsp': '兄弟姐妹数',
                    'parch': '父母小孩数',
                    'ticket': '船票号',
                    'fare': '票价',
                    'cabin': '船舱号',
                    'embarked': '登船港口',
                    'survival_probability': '生存概率',
                    'predicted_survived': '预测结果',
                    'created_at': '创建时间'
                }
            )
            
            st.write(f"共 {len(filtered_df)} 条记录（总计 {len(df)} 条）")
        
        st.markdown("---")
        st.subheader("🗑️ 删除旅客记录")
        
        passenger_ids = df['id'].tolist()
        passenger_names = df['name'].tolist()
        options = [f"ID: {pid} - {name}" for pid, name in zip(passenger_ids, passenger_names)]
        
        # 初始化 session_state
        if 'delete_trigger' not in st.session_state:
            st.session_state.delete_trigger = 0
        
        # 使用唯一的 key 确保 multiselect 每次都能正确重置
        multiselect_key = f"delete_multiselect_{st.session_state.delete_trigger}"
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ 全选"):
                st.session_state.selected_options = options.copy()
                st.session_state.delete_trigger += 1
                st.rerun()
        
        with col2:
            if st.button("❌ 取消全选"):
                st.session_state.selected_options = []
                st.session_state.delete_trigger += 1
                st.rerun()
        
        # 获取当前选中的选项
        if 'selected_options' not in st.session_state:
            st.session_state.selected_options = []
        
        selected_options = st.multiselect(
            "选择要删除的旅客记录（可多选）",
            options=options,
            default=st.session_state.selected_options,
            key=multiselect_key
        )
        
        # 更新 session_state
        st.session_state.selected_options = selected_options
        
        selected_ids = []
        selected_names = []
        for opt in selected_options:
            idx = options.index(opt)
            selected_ids.append(passenger_ids[idx])
            selected_names.append(passenger_names[idx])
        
        if len(selected_ids) > 0:
            st.warning(f"⚠️ 即将删除以下 {len(selected_ids)} 条记录：")
            st.write(", ".join(selected_names))
            
            if st.button(f"❌ 确认删除 {len(selected_ids)} 条记录", type="primary"):
                success_count = 0
                for pid in selected_ids:
                    if db.delete_passenger(pid):
                        success_count += 1
                
                if success_count == len(selected_ids):
                    st.success(f"✅ 成功删除全部 {success_count} 条记录！")
                    st.session_state.selected_options = []
                    st.session_state.delete_trigger += 1
                elif success_count > 0:
                    st.warning(f"⚠️ 成功删除 {success_count}/{len(selected_ids)} 条记录")
                    st.session_state.selected_options = []
                    st.session_state.delete_trigger += 1
                else:
                    st.error("❌ 删除失败！")
                
                st.rerun()


def show_statistics_page(db, model=None):
    st.header("📊 系统统计")
    
    # 显示模型信息
    st.subheader("🤖 模型信息")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if model and model.accuracy is not None:
            st.metric("模型准确率", f"{model.accuracy:.2%}")
        else:
            st.metric("模型准确率", "未知")
    
    with col2:
        st.metric("模型类型", "MLP/DNN")
    
    with col3:
        st.metric("输入特征数", 12)
    
    with col4:
        st.metric("隐藏层", "2层")
    
    st.markdown("---")
    
    # 数据库统计
    stats = db.get_statistics()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("总预测次数", stats['total'])
    
    with col2:
        st.metric("预测生存", stats['survived'])
    
    with col3:
        st.metric("预测遇难", stats['not_survived'])
    
    with col4:
        st.metric("平均生存概率", f"{stats['avg_survival_probability']:.2%}")
    
    if stats['total'] > 0:
        df = db.get_all_passengers()
        
        st.markdown("---")
        st.subheader("生存概率分布")
        
        fig_col1, fig_col2 = st.columns(2)
        
        with fig_col1:
            survival_counts = df['predicted_survived'].value_counts()
            survival_df = pd.DataFrame({
                '结果': ['遇难', '生存'],
                '人数': [survival_counts.get(0, 0), survival_counts.get(1, 0)]
            })
            st.bar_chart(survival_df.set_index('结果'))
        
        with fig_col2:
            st.subheader("按客舱等级统计")
            pclass_stats = df.groupby('pclass')['predicted_survived'].mean()
            pclass_df = pd.DataFrame({
                '客舱等级': ['一等舱', '二等舱', '三等舱'],
                '生存率': [
                    pclass_stats.get(1, 0) * 100,
                    pclass_stats.get(2, 0) * 100,
                    pclass_stats.get(3, 0) * 100
                ]
            })
            st.bar_chart(pclass_df.set_index('客舱等级'))


def show_data_quality_page():
    st.header("📋 数据质量检查")
    st.markdown("检查数据集的质量和完整性")
    
    # 数据上传功能
    st.subheader("📁 数据上传")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_file = st.file_uploader("上传CSV数据文件", type="csv")
    
    with col2:
        use_default = st.button("使用默认数据")
    
    # 处理数据加载
    data_dir = r'd:\实训项目\data\Titanic生存率'
    train_path = os.path.join(data_dir, 'mytrain.csv')
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ 成功加载上传的文件，共 {len(df)} 条记录")
    elif use_default or os.path.exists(train_path):
        df = pd.read_csv(train_path)
        st.info(f"📂 使用默认数据集，共 {len(df)} 条记录")
    else:
        st.error("错误: 训练数据文件不存在，请上传数据文件!")
        return
    
    st.subheader("📊 数据基本信息")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("样本数量", len(df))
    with col2:
        st.metric("特征数量", len(df.columns))
    with col3:
        st.metric("数据完整度", f"{(1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100:.2f}%")
    
    st.subheader("📁 数据类型")
    st.dataframe(df.dtypes.reset_index().rename(columns={'index': '特征', 0: '类型'}))
    
    st.subheader("⚠️ 缺失值统计")
    missing = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        '缺失数量': missing,
        '缺失比例': missing_pct.apply(lambda x: f"{x}%")
    })
    st.dataframe(missing_df[missing > 0])
    
    st.subheader("📈 数值特征统计")
    numerical_cols = ['Age', 'SibSp', 'Parch', 'Fare']
    st.dataframe(df[numerical_cols].describe().round(2))
    
    st.subheader("🏷️ 分类特征统计")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**生存情况**")
        st.dataframe(df['Survived'].value_counts())
        st.metric("生存率", f"{df['Survived'].mean()*100:.2f}%")
    
    with col2:
        st.markdown("**客舱等级**")
        st.dataframe(df['Pclass'].value_counts().sort_index())
    
    with col3:
        st.markdown("**性别**")
        st.dataframe(df['Sex'].value_counts())
    
    with col4:
        st.markdown("**登船港口**")
        st.dataframe(df['Embarked'].value_counts())
    
    st.subheader("💡 关键发现")
    st.info("""
    - **Age列**有较多缺失值，需要用中位数或均值填充
    - **Cabin列**缺失严重，可能需要从模型中排除或特殊处理
    - **Embarked列**有少量缺失，需要用众数填充
    - 可以从Name中提取Title作为新特征
    - FamilySize特征可以通过SibSp和Parch计算得出
    """)


def show_visualization_page():
    st.header("📊 数据可视化分析")
    st.markdown("可视化数据集的分布和特征关系")
    
    # 数据上传功能
    st.subheader("📁 数据上传")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_file = st.file_uploader("上传CSV数据文件", type="csv")
    
    with col2:
        use_default = st.button("使用默认数据")
    
    # 处理数据加载
    data_path = r'd:\实训项目\data\Titanic生存率\mytrain.csv'
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ 成功加载上传的文件，共 {len(df)} 条记录")
    elif use_default or os.path.exists(data_path):
        df = pd.read_csv(data_path)
        st.info(f"📂 使用默认数据集，共 {len(df)} 条记录")
    else:
        st.error("错误: 训练数据文件不存在，请上传数据文件!")
        return
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Titanic 数据集分析报告', fontsize=20, fontweight='bold')
    
    ax1 = axes[0, 0]
    survival_counts = df['Survived'].value_counts()
    colors = ['#ff6b6b', '#51cf66']
    bars = ax1.bar(['遇难', '生存'], [survival_counts[0], survival_counts[1]], color=colors)
    ax1.set_title('整体生存率', fontsize=12, fontweight='bold')
    ax1.set_ylabel('人数')
    for bar, count in zip(bars, [survival_counts[0], survival_counts[1]]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'{count}\n({count/len(df)*100:.1f}%)', ha='center', fontsize=10)
    
    ax2 = axes[0, 1]
    sex_survival = df.groupby('Sex')['Survived'].mean() * 100
    colors_sex = ['#339af0', '#f06595']
    bars = ax2.bar(['女性', '男性'], [sex_survival['female'], sex_survival['male']], color=colors_sex)
    ax2.set_title('性别与生存率', fontsize=12, fontweight='bold')
    ax2.set_ylabel('生存率 (%)')
    for bar, rate in zip(bars, [sex_survival['female'], sex_survival['male']]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{rate:.1f}%', ha='center', fontsize=10, fontweight='bold')
    
    ax3 = axes[0, 2]
    pclass_survival = df.groupby('Pclass')['Survived'].mean() * 100
    colors_pclass = ['#ffd43b', '#fab005', '#e67700']
    bars = ax3.bar(['一等舱', '二等舱', '三等舱'], 
                   [pclass_survival[1], pclass_survival[2], pclass_survival[3]], 
                   color=colors_pclass)
    ax3.set_title('客舱等级与生存率', fontsize=12, fontweight='bold')
    ax3.set_ylabel('生存率 (%)')
    for bar, rate in zip(bars, [pclass_survival[1], pclass_survival[2], pclass_survival[3]]):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{rate:.1f}%', ha='center', fontsize=10, fontweight='bold')
    
    ax4 = axes[1, 0]
    df['Age'].hist(bins=30, ax=ax4, color='#74c0fc', edgecolor='black', alpha=0.7)
    ax4.set_title('年龄分布', fontsize=12, fontweight='bold')
    ax4.set_xlabel('年龄')
    ax4.set_ylabel('频数')
    ax4.axvline(df['Age'].median(), color='red', linestyle='--', linewidth=2, label=f'中位数: {df["Age"].median():.1f}')
    ax4.legend()
    
    ax5 = axes[1, 1]
    survived = df[df['Survived'] == 1]['Age']
    not_survived = df[df['Survived'] == 0]['Age']
    ax5.hist([survived, not_survived], bins=30, label=['生存', '遇难'], 
             color=['#51cf66', '#ff6b6b'], alpha=0.7, edgecolor='black')
    ax5.set_title('年龄与生存关系', fontsize=12, fontweight='bold')
    ax5.set_xlabel('年龄')
    ax5.set_ylabel('频数')
    ax5.legend()
    
    ax6 = axes[1, 2]
    df['Fare'].hist(bins=50, ax=ax6, color='#ffd43b', edgecolor='black', alpha=0.7)
    ax6.set_title('票价分布', fontsize=12, fontweight='bold')
    ax6.set_xlabel('票价')
    ax6.set_ylabel('频数')
    ax6.axvline(df['Fare'].median(), color='red', linestyle='--', linewidth=2, 
               label=f'中位数: {df["Fare"].median():.2f}')
    ax6.legend()
    
    ax7 = axes[2, 0]
    embarked_survival = df.groupby('Embarked')['Survived'].mean() * 100
    colors_embarked = ['#845ef7', '#20c997', '#339af0']
    embark_mapping = {'C': '瑟堡', 'Q': '皇后镇', 'S': '南安普顿'}
    bars = ax7.bar([embark_mapping[x] for x in embarked_survival.index], 
                   embarked_survival.values, color=colors_embarked)
    ax7.set_title('登船港口与生存率', fontsize=12, fontweight='bold')
    ax7.set_ylabel('生存率 (%)')
    for bar, rate in zip(bars, embarked_survival.values):
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{rate:.1f}%', ha='center', fontsize=10, fontweight='bold')
    
    ax8 = axes[2, 1]
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    family_survival = df.groupby('FamilySize')['Survived'].mean() * 100
    ax8.bar(family_survival.index, family_survival.values, color='#20c997', edgecolor='black')
    ax8.set_title('家庭规模与生存率', fontsize=12, fontweight='bold')
    ax8.set_xlabel('家庭规模')
    ax8.set_ylabel('生存率 (%)')
    
    ax9 = axes[2, 2]
    pivot_table = df.pivot_table(values='Survived', index='Sex', columns='Pclass', aggfunc='mean') * 100
    pivot_table.index = ['女性', '男性']
    pivot_table.columns = ['一等舱', '二等舱', '三等舱']
    sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax9, 
                cbar_kws={'label': '生存率 (%)'}, linewidths=0.5)
    ax9.set_title('性别与客舱等级交互效应', fontsize=12, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    st.pyplot(fig)
    
    st.subheader("🔗 特征相关性矩阵")
    df_corr = df.copy()
    df_corr['Sex'] = df_corr['Sex'].map({'male': 0, 'female': 1})
    df_corr['Embarked'] = df_corr['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    correlation_matrix = df_corr[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].corr()
    
    fig2, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
    plt.title('特征相关性矩阵', fontsize=16, fontweight='bold', pad=20)
    st.pyplot(fig2)


if __name__ == "__main__":
    main()
