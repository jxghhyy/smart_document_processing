"""
智能文档处理 - 多种机器学习模型实现
"""
import os
import json
import re
import platform
from pathlib import Path
from typing import List, Dict, Tuple, Any
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置字体 - 支持Windows/Mac/Linux
system_name = platform.system()

if system_name == 'Windows':
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
elif system_name == 'Darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'STHeiti']
else:  # Linux
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']

plt.rcParams['axes.unicode_minus'] = False
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder


class KeywordMatcher:
    """基于关键词的基线方法"""

    def __init__(self):
        self.keywords = {
            'question': ['?', 'what', 'when', 'where', 'who', 'how', 'why', 'which'],
            'answer': ['$', 'yes', 'no', 'date', 'number', 'quantity', 'amount', 'total'],
            'header': ['to:', 'from:', 'date:', 'subject:', 'invoice', 'order', 'doc']
        }

    def predict(self, text: str) -> str:
        text = text.lower()
        scores = {}

        for label, keywords in self.keywords.items():
            score = sum(1 for kw in keywords if kw in text)
            scores[label] = score

        max_score = max(scores.values()) if scores else 0
        if max_score > 0:
            return max(scores.items(), key=lambda x: x[1])[0]
        return 'other'

    def predict_batch(self, texts: List[str]) -> List[str]:
        return [self.predict(text) for text in texts]


class MLClassifier:
    """基于传统机器学习的分类器"""

    def __init__(self, method: str = 'tfidf', classifier: str = 'lr'):
        self.method = method
        self.classifier_type = classifier

        if method == 'count':
            self.vectorizer = CountVectorizer(max_features=5000, ngram_range=(1, 2))
        else:
            self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

        if classifier == 'nb':
            self.classifier = MultinomialNB()
        elif classifier == 'rf':
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            self.classifier = LogisticRegression(max_iter=500, random_state=42)

        self.label_encoder = LabelEncoder()
        self.is_trained = False

    def fit(self, texts: List[str], labels: List[str]):
        X = self.vectorizer.fit_transform(texts)
        y = self.label_encoder.fit_transform(labels)
        self.classifier.fit(X, y)
        self.is_trained = True

    def predict(self, texts: List[str]) -> List[str]:
        if not self.is_trained:
            return ['other'] * len(texts)
        X = self.vectorizer.transform(texts)
        y_pred = self.classifier.predict(X)
        return self.label_encoder.inverse_transform(y_pred)


def load_funsd_data(data_dir: str, split: str = 'training') -> Tuple[List[str], List[str]]:
    """加载FUNSD数据集"""
    if split == 'training':
        ann_dir = os.path.join(data_dir, 'FUNSD', 'training_data', 'annotations')
    else:
        ann_dir = os.path.join(data_dir, 'FUNSD', 'testing_data', 'annotations')

    all_texts = []
    all_labels = []

    ann_files = [f for f in os.listdir(ann_dir) if f.endswith('.json')]
    for ann_file in ann_files:
        ann_path = os.path.join(ann_dir, ann_file)
        with open(ann_path, 'r', encoding='utf-8') as f:
            ann = json.load(f)

        if 'form' in ann:
            for field in ann['form']:
                if 'text' in field and 'label' in field:
                    all_texts.append(field['text'])
                    all_labels.append(field['label'])

    return all_texts, all_labels


def evaluate_model(y_true: List[str], y_pred: List[str], model_name: str) -> Dict[str, Any]:
    """评估模型并返回结果"""
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    return {
        'model': model_name,
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'f1_micro': float(f1_micro),
        'report': report,
        'confusion_matrix': cm.tolist() if isinstance(cm, np.ndarray) else cm
    }


def plot_results(results: List[Dict], output_path: str):
    """可视化结果对比"""
    models = [r['model'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    f1_macros = [r['f1_macro'] for r in results]
    f1_micros = [r['f1_micro'] for r in results]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width, accuracies, width, label='Accuracy', color='#1f77b4')
    rects2 = ax.bar(x, f1_macros, width, label='F1 (Macro)', color='#ff7f0e')
    rects3 = ax.bar(x + width, f1_micros, width, label='F1 (Micro)', color='#2ca02c')

    ax.set_xlabel('Model', fontsize=14)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('Performance Comparison of Different Models', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=12)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords='offset points',
                        ha='center', va='bottom', fontsize=10)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')


def plot_confusion_matrix(y_true: List[str], y_pred: List[str], labels: List[str],
                          output_path: str, title: str = 'Confusion Matrix'):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels,
           yticklabels=labels,
           title=title,
           ylabel='True Label',
           xlabel='Predicted Label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')


def main():
    # 路径设置
    current_dir = Path(__file__).parent
    project_dir = current_dir.parent
    data_dir = project_dir / "data"
    output_dir = project_dir / "reports"
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("智能文档处理 - 模型训练与评估")
    print("=" * 60)

    # 加载数据
    print("\n加载FUNSD数据集...")
    train_texts, train_labels = load_funsd_data(str(data_dir), 'training')
    test_texts, test_labels = load_funsd_data(str(data_dir), 'testing')

    print(f"训练样本: {len(train_texts)}")
    print(f"测试样本: {len(test_texts)}")
    print(f"训练标签分布: {Counter(train_labels)}")
    print(f"测试标签分布: {Counter(test_labels)}")

    # 保存数据
    pd.DataFrame({'text': train_texts, 'label': train_labels, 'split': 'train'}).to_csv(
        output_dir / 'train_data.csv', index=False)
    pd.DataFrame({'text': test_texts, 'label': test_labels, 'split': 'test'}).to_csv(
        output_dir / 'test_data.csv', index=False)

    # 获取标签列表
    unique_labels = sorted(list(set(train_labels + test_labels)))

    # 存储所有结果
    all_results = []

    # 方法1: 关键词匹配
    print("\n" + "-" * 60)
    print("方法1: 关键词匹配")
    print("-" * 60)

    keyword_clf = KeywordMatcher()
    keyword_preds = keyword_clf.predict_batch(test_texts)
    keyword_result = evaluate_model(test_labels, keyword_preds, 'Keyword Matching')
    all_results.append(keyword_result)
    print(f"准确率: {keyword_result['accuracy']:.4f}")
    print(f"F1 (Macro): {keyword_result['f1_macro']:.4f}")

    # 方法2: Naive Bayes + TF-IDF
    print("\n" + "-" * 60)
    print("方法2: 朴素贝叶斯 + TF-IDF")
    print("-" * 60)

    nb_clf = MLClassifier(method='tfidf', classifier='nb')
    nb_clf.fit(train_texts, train_labels)
    nb_preds = nb_clf.predict(test_texts)
    nb_result = evaluate_model(test_labels, nb_preds, 'Naive Bayes + TF-IDF')
    all_results.append(nb_result)
    print(f"准确率: {nb_result['accuracy']:.4f}")
    print(f"F1 (Macro): {nb_result['f1_macro']:.4f}")

    # 方法3: Logistic Regression + TF-IDF
    print("\n" + "-" * 60)
    print("方法3: 逻辑回归 + TF-IDF")
    print("-" * 60)

    lr_clf = MLClassifier(method='tfidf', classifier='lr')
    lr_clf.fit(train_texts, train_labels)
    lr_preds = lr_clf.predict(test_texts)
    lr_result = evaluate_model(test_labels, lr_preds, 'Logistic Regression + TF-IDF')
    all_results.append(lr_result)
    print(f"准确率: {lr_result['accuracy']:.4f}")
    print(f"F1 (Macro): {lr_result['f1_macro']:.4f}")

    # 方法4: Random Forest + TF-IDF
    print("\n" + "-" * 60)
    print("方法4: 随机森林 + TF-IDF")
    print("-" * 60)

    rf_clf = MLClassifier(method='tfidf', classifier='rf')
    rf_clf.fit(train_texts, train_labels)
    rf_preds = rf_clf.predict(test_texts)
    rf_result = evaluate_model(test_labels, rf_preds, 'Random Forest + TF-IDF')
    all_results.append(rf_result)
    print(f"准确率: {rf_result['accuracy']:.4f}")
    print(f"F1 (Macro): {rf_result['f1_macro']:.4f}")

    # 可视化结果对比
    print("\n生成可视化...")
    plot_results(all_results, output_dir / 'model_comparison.png')

    # 绘制最佳模型的混淆矩阵
    best_result = max(all_results, key=lambda x: x['accuracy'])
    print(f"\n最佳模型: {best_result['model']} (准确率: {best_result['accuracy']:.4f})")

    # 确定最佳模型的预测结果
    best_preds = None
    if best_result['model'] == '关键词匹配':
        best_preds = keyword_preds
    elif best_result['model'] == '朴素贝叶斯+TF-IDF':
        best_preds = nb_preds
    elif best_result['model'] == '逻辑回归+TF-IDF':
        best_preds = lr_preds
    else:
        best_preds = rf_preds

    plot_confusion_matrix(test_labels, best_preds, unique_labels,
                          output_dir / 'best_confusion_matrix.png',
                          f'Confusion Matrix - {best_result["model"]}')

    # 保存结果
    results_summary = {
        'models': all_results,
        'dataset_stats': {
            'train_count': len(train_texts),
            'test_count': len(test_texts),
            'train_labels': dict(Counter(train_labels)),
            'test_labels': dict(Counter(test_labels))
        },
        'best_model': best_result
    }

    with open(output_dir / 'results.json', 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存至: {output_dir}")
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
