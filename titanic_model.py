import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from config import MODEL_DIR, DATA_DIR
from data_preprocessing import TitanicPreprocessor, download_sample_data


class TitanicDNNModel:
    def __init__(self, input_dim=12):
        self.input_dim = input_dim
        self.model = None
        self.history = None
        self.preprocessor = TitanicPreprocessor()

    def build_model(self):
        """
        构建多层感知机(MLP/DNN)模型
        模型结构说明：
        - 输入层：12个特征节点
        - 隐藏层1：64个神经元，ReLU激活函数，BatchNormalization和Dropout防止过拟合
        - 隐藏层2：32个神经元，ReLU激活函数，BatchNormalization和Dropout防止过拟合
        - 输出层：1个神经元，Sigmoid激活函数输出0-1之间的生存概率
        
        选择此模型的原因：
        1. MLP结构简单但强大，适合处理结构化数据
        2. 两层隐藏层足够捕捉特征间的非线性关系
        3. ReLU激活函数解决梯度消失问题，加速收敛
        4. BatchNormalization加速训练并提高稳定性
        5. Dropout防止过拟合，提高泛化能力
        6. Sigmoid输出层适合二分类问题，输出概率值
        """
        model = Sequential([
            Dense(64, activation='relu', input_dim=self.input_dim),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        self.best_threshold = 0.5  # 最佳分类阈值
        return model

    def train(self, X, y, epochs=100, batch_size=32, validation_split=0.2):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # 计算类别权重，处理不平衡数据
        class_counts = np.bincount(y_train)
        weight_for_0 = (1 / class_counts[0]) * (len(y_train) / 2.0)
        weight_for_1 = (1 / class_counts[1]) * (len(y_train) / 2.0)
        class_weight = {0: weight_for_0, 1: weight_for_1}
        print(f"类别权重: 遇难={weight_for_0:.2f}, 生存={weight_for_1:.2f}")
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        checkpoint_path = os.path.join(MODEL_DIR, 'best_model.keras')
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            class_weight=class_weight,
            callbacks=[early_stopping, checkpoint],
            verbose=1
        )
        
        # 寻找最佳分类阈值
        self._find_best_threshold(X_val, y_val)
        
        return self.history
    
    def _find_best_threshold(self, X_val, y_val):
        y_pred_proba = self.model.predict(X_val, verbose=0).flatten()
        
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in np.arange(0.1, 0.9, 0.01):
            y_pred = (y_pred_proba >= threshold).astype(int)
            f1 = f1_score(y_val, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        self.best_threshold = best_threshold
        print(f"最佳分类阈值: {best_threshold:.2f} (F1={best_f1:.4f})")

    def evaluate(self, X_test, y_test):
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba >= self.best_threshold).astype(int).flatten()
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"模型准确率: {accuracy:.4f}")
        print(f"使用阈值: {self.best_threshold:.2f}")
        print("\n分类报告:")
        print(classification_report(y_test, y_pred))
        print("\n混淆矩阵:")
        print(confusion_matrix(y_test, y_pred))
        
        return accuracy

    def predict(self, X):
        proba = self.model.predict(X, verbose=0).flatten()
        return proba
    
    def predict_class(self, X):
        proba = self.predict(X)
        return (proba >= self.best_threshold).astype(int), proba

    def plot_training_history(self):
        if self.history is None:
            print("请先训练模型")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(self.history.history['accuracy'], label='训练准确率')
        ax1.plot(self.history.history['val_accuracy'], label='验证准确率')
        ax1.set_title('模型准确率')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('准确率')
        ax1.legend()
        
        ax2.plot(self.history.history['loss'], label='训练损失')
        ax2.plot(self.history.history['val_loss'], label='验证损失')
        ax2.set_title('模型损失')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('损失')
        ax2.legend()
        
        plt.tight_layout()
        save_path = os.path.join(MODEL_DIR, 'training_history.png')
        plt.savefig(save_path)
        print(f"训练历史图表已保存至: {save_path}")
        plt.close()

    def save(self, filepath):
        self.model.save(filepath)
        preprocessor_path = filepath.replace('.keras', '_preprocessor.pkl')
        self.preprocessor.save(preprocessor_path)
        
        # 保存阈值
        threshold_path = filepath.replace('.keras', '_threshold.txt')
        with open(threshold_path, 'w') as f:
            f.write(str(self.best_threshold))

    @classmethod
    def load(cls, model_path, preprocessor_path):
        model = tf.keras.models.load_model(model_path)
        preprocessor = TitanicPreprocessor.load(preprocessor_path)
        
        instance = cls(input_dim=preprocessor.feature_names.__len__())
        instance.model = model
        instance.preprocessor = preprocessor
        
        # 加载阈值
        threshold_path = model_path.replace('.keras', '_threshold.txt')
        if os.path.exists(threshold_path):
            with open(threshold_path, 'r') as f:
                instance.best_threshold = float(f.read())
        else:
            instance.best_threshold = 0.5
        
        return instance


def train_and_save_model():
    print("=== Titanic 生存预测模型训练 ===")
    
    print("\n1. 加载数据...")
    result = download_sample_data()
    
    if isinstance(result, tuple):
        train_df, test_df = result
        print(f"训练数据加载完成，共 {len(train_df)} 条记录")
        if test_df is not None:
            print(f"测试数据加载完成，共 {len(test_df)} 条记录")
    else:
        train_df = result
        test_df = None
        print(f"数据加载完成，共 {len(train_df)} 条记录")
    
    print("\n2. 数据预处理...")
    preprocessor = TitanicPreprocessor()
    X = preprocessor.fit_transform(train_df)
    y = train_df['Survived'].values
    
    print("\n3. 构建模型...")
    dnn_model = TitanicDNNModel(input_dim=X.shape[1])
    dnn_model.build_model()
    dnn_model.preprocessor = preprocessor
    dnn_model.model.summary()
    
    print("\n4. 训练模型...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    dnn_model.train(X_train, y_train, epochs=100, batch_size=32)
    
    print("\n5. 评估模型...")
    accuracy = dnn_model.evaluate(X_test, y_test)
    
    print("\n6. 保存训练历史图表...")
    dnn_model.plot_training_history()
    
    print("\n7. 保存模型...")
    model_path = os.path.join(MODEL_DIR, 'titanic_dnn_model.keras')
    dnn_model.save(model_path)
    print(f"模型已保存至: {model_path}")
    
    # 保存模型准确率
    accuracy_path = os.path.join(MODEL_DIR, 'model_accuracy.txt')
    with open(accuracy_path, 'w') as f:
        f.write(str(accuracy))
    print(f"模型准确率已保存至: {accuracy_path}")
    
    return dnn_model, accuracy


if __name__ == "__main__":
    model, accuracy = train_and_save_model()
