import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os
from config import DATA_DIR, MODEL_DIR


class TitanicPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.is_fitted = False
        self.fare_bins = None  # 保存训练时的票价分位数边界

    def _extract_title(self, name):
        try:
            # 处理姓名格式，提取称呼
            parts = name.split(',')
            if len(parts) >= 2:
                title_part = parts[1].strip()
                title_parts = title_part.split('.')
                if len(title_parts) >= 1:
                    title = title_parts[0].strip()
                else:
                    title = 'Other'
            else:
                # 如果没有逗号，尝试直接找点号
                parts = name.split('.')
                if len(parts) >= 2:
                    title = parts[0].strip().split()[-1]
                else:
                    title = 'Other'
            
            title_mapping = {
                'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
                'Dr': 'Officer', 'Rev': 'Officer', 'Col': 'Officer',
                'Major': 'Officer', 'Mlle': 'Miss', 'Ms': 'Mrs', 'Mme': 'Mrs',
                'Capt': 'Officer', 'Don': 'Royalty', 'Sir': 'Royalty',
                'the Countess': 'Royalty', 'Dona': 'Royalty', 'Lady': 'Royalty', 'Jonkheer': 'Royalty'
            }
            return title_mapping.get(title, 'Other')
        except Exception:
            return 'Other'

    def fit_transform(self, df):
        df = df.copy()
        
        df['Title'] = df['Name'].apply(self._extract_title)
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        
        df['Age'] = df['Age'].fillna(df['Age'].median())
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
        
        df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 18, 30, 50, 100], labels=['Child', 'Teen', 'YoungAdult', 'Adult', 'Senior'])
        df['FareBin'], self.fare_bins = pd.qcut(df['Fare'], 4, labels=['Low', 'Medium', 'High', 'VeryHigh'], retbins=True)
        
        categorical_features = ['Sex', 'Embarked', 'Title', 'AgeBin', 'FareBin', 'Pclass']
        for feature in categorical_features:
            le = LabelEncoder()
            df[feature] = le.fit_transform(df[feature].astype(str))
            self.label_encoders[feature] = le
        
        features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 
                    'Title', 'FamilySize', 'IsAlone', 'AgeBin', 'FareBin']
        self.feature_names = features
        
        X = df[features].values
        X_scaled = self.scaler.fit_transform(X)
        self.is_fitted = True
        
        return X_scaled

    def transform(self, df):
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted yet. Call fit_transform first.")
        
        df = df.copy()
        
        df['Title'] = df['Name'].apply(self._extract_title)
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        
        df['Age'] = df['Age'].fillna(df['Age'].median() if 'Age' in df else 28)
        df['Fare'] = df['Fare'].fillna(df['Fare'].median() if 'Fare' in df else 14)
        df['Embarked'] = df['Embarked'].fillna('S')
        
        df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 18, 30, 50, 100], labels=['Child', 'Teen', 'YoungAdult', 'Adult', 'Senior'])
        
        if self.fare_bins is not None:
            df['FareBin'] = pd.cut(df['Fare'], bins=self.fare_bins, labels=['Low', 'Medium', 'High', 'VeryHigh'], include_lowest=True)
        else:
            df['FareBin'] = pd.cut(df['Fare'], bins=[0, 14.4542, 31, 51.8625, float('inf')], labels=['Low', 'Medium', 'High', 'VeryHigh'])
        
        for feature, le in self.label_encoders.items():
            df[feature] = df[feature].astype(str)
            unseen = set(df[feature]) - set(le.classes_)
            if unseen:
                for u in unseen:
                    le.classes_ = np.append(le.classes_, u)
            df[feature] = le.transform(df[feature])
        
        X = df[self.feature_names].values
        X_scaled = self.scaler.transform(X)
        
        return X_scaled

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def download_sample_data():
    """
    加载Titanic数据集
    优先使用本地数据文件，如果不存在则从网络下载
    """
    local_train_path = os.path.join(DATA_DIR, 'Titanic生存率', 'mytrain.csv')
    local_test_path = os.path.join(DATA_DIR, 'Titanic生存率', 'mytest.csv')
    
    if os.path.exists(local_train_path):
        print(f"加载本地训练数据: {local_train_path}")
        train_df = pd.read_csv(local_train_path)
        print(f"训练数据形状: {train_df.shape}")
        
        if os.path.exists(local_test_path):
            print(f"加载本地测试数据: {local_test_path}")
            test_df = pd.read_csv(local_test_path)
            print(f"测试数据形状: {test_df.shape}")
            return train_df, test_df
        
        return train_df, None
    
    print("本地数据文件不存在，从网络下载...")
    train_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(train_url)
    save_path = os.path.join(DATA_DIR, 'titanic.csv')
    df.to_csv(save_path, index=False)
    return df, None
