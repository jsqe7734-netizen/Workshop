import pandas as pd
import numpy as np
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class TitanicModel:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.accuracy = None
    
    def build_preprocessor(self):
        numerical_features = ['Age', 'SibSp', 'Parch', 'Fare']
        categorical_features = ['Pclass', 'Sex', 'Embarked']
        
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])
    
    def train(self, X, y):
        self.build_preprocessor()
        
        self.model = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ))
        ])
        
        self.model.fit(X, y)
        self.accuracy = self.model.score(X, y)
        return self.accuracy
    
    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict_proba(X)[:, 1]
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)
    
    def save(self, model_path, preprocessor_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.model, model_path)
    
    @classmethod
    def load(cls, model_path, preprocessor_path=None):
        instance = cls()
        instance.model = joblib.load(model_path)
        instance.preprocessor = instance.model.named_steps['preprocessor']
        return instance

def train_and_save_model():
    from sklearn.impute import SimpleImputer
    
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'Titanic生存率', 'mytrain.csv')
    df = pd.read_csv(data_path)
    
    X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    y = df['Survived']
    
    model = TitanicModel()
    accuracy = model.train(X, y)
    
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    model_path = os.path.join(model_dir, 'titanic_rf_model.pkl')
    preprocessor_path = os.path.join(model_dir, 'titanic_rf_preprocessor.pkl')
    
    model.save(model_path, preprocessor_path)
    
    accuracy_path = os.path.join(model_dir, 'model_accuracy.txt')
    with open(accuracy_path, 'w') as f:
        f.write(str(accuracy))
    
    return model, accuracy

if __name__ == "__main__":
    model, accuracy = train_and_save_model()
    print(f"Model trained with accuracy: {accuracy:.4f}")