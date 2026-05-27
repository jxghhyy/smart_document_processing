"""
智能文档处理 - 多种模型实现
"""
import os
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Any
from collections import defaultdict, Counter
import numpy as np


class RuleBasedExtractor:
    """
    基于规则的文档信息提取
    使用OCR + 正则表达式匹配
    """

    def __init__(self):
        self.patterns = {
            'total': [
                r'(?i)(?:total|amount|balance)[\s:]*\$?([\d,]+\.?\d*)',
                r'(?i)合计[\s:：]*\$?([\d,]+\.?\d*)'
            ],
            'date': [
                r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
                r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})'
            ],
            'tax': [
                r'(?i)tax[\s:]*\$?([\d,]+\.?\d*)'
            ]
        }

    def extract(self, text: str) -> Dict[str, Any]:
        results = {}

        for field, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text)
                if matches:
                    results[field] = matches[-1]  # 取最后一个匹配
                    break

        return results


class NaiveKeywordMatcher:
    """
    基于关键词的方法 - 基线方法
    """

    def __init__(self):
        self.keywords = {
            'question': ['?', 'what', 'when', 'where', 'who', 'how', 'why'],
            'answer': ['$', 'yes', 'no', 'date', 'number', 'quantity'],
            'header': ['to:', 'from:', 'date:', 'subject:', 'invoice', 'order']
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


class StatisticalClassifier:
    """
    基于统计的分类器
    使用词袋模型 + 简单的分类算法
    """

    def __init__(self):
        self.vocab = {}
        self.label_word_counts = defaultdict(Counter)
        self.label_priors = {}
        self.is_trained = False

    def fit(self, texts: List[str], labels: List[str]):
        # 统计标签分布
        label_counts = Counter(labels)
        total = len(labels)
        for label, count in label_counts.items():
            self.label_priors[label] = count / total

        # 构建词汇表
        all_words = []
        for text in texts:
            words = self._tokenize(text)
            all_words.extend(words)
            label = labels[texts.index(text)]
            self.label_word_counts[label].update(words)

        self.vocab = {word: i for i, word in enumerate(set(all_words))}
        self.is_trained = True

    def predict(self, text: str) -> str:
        if not self.is_trained:
            return 'other'

        words = self._tokenize(text)
        scores = {}

        for label in self.label_priors:
            score = np.log(self.label_priors[label])

            for word in words:
                # 拉普拉斯平滑
                count = self.label_word_counts[label].get(word, 0) + 1
                total = sum(self.label_word_counts[label].values()) + len(self.vocab)
                score += np.log(count / total)

            scores[label] = score

        return max(scores.items(), key=lambda x: x[1])[0]

    def _tokenize(self, text: str) -> List[str]:
        # 简单的分词
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
        return text.split()


class SimpleDocumentProcessor:
    """
    简单的文档处理整合类
    """

    def __init__(self):
        self.keyword_matcher = NaiveKeywordMatcher()
        self.statistical_clf = StatisticalClassifier()
        self.rule_extractor = RuleBasedExtractor()

    def train_statistical(self, texts: List[str], labels: List[str]):
        print(f"训练统计分类器，样本数: {len(texts)}")
        self.statistical_clf.fit(texts, labels)

    def predict(self, text: str, method: str = 'keyword') -> str:
        if method == 'keyword':
            return self.keyword_matcher.predict(text)
        elif method == 'statistical':
            return self.statistical_clf.predict(text)
        else:
            return 'other'


def load_funsd_data(data_dir: str, split: str = 'training') -> Tuple[List[Dict], List[str]]:
    """
    加载FUNSD数据集
    """
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


def evaluate_predictions(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    """
    评估预测结果
    """
    accuracy = sum(yt == yp for yt, yp in zip(y_true, y_pred)) / len(y_true)

    # 各类别准确率
    label_acc = {}
    for label in set(y_true):
        mask = [yt == label for yt in y_true]
        correct = sum(yt == yp for yt, yp, m in zip(y_true, y_pred, mask) if m)
        total = sum(mask)
        label_acc[label] = correct / total if total > 0 else 0

    return {'accuracy': accuracy, 'per_class': label_acc}


def main():
    # 设置路径
    current_dir = Path(__file__).parent
    project_dir = current_dir.parent
    data_dir = project_dir / "data"

    print("=" * 60)
    print("智能文档处理 - 模型训练与评估")
    print("=" * 60)

    # 加载数据
    print("\n加载FUNSD数据集...")
    train_texts, train_labels = load_funsd_data(str(data_dir), 'training')
    test_texts, test_labels = load_funsd_data(str(data_dir), 'testing')

    print(f"训练样本: {len(train_texts)}")
    print(f"测试样本: {len(test_texts)}")
    print(f"标签分布: {Counter(train_labels)}")

    # 创建处理器
    processor = SimpleDocumentProcessor()

    # 方法1: 关键词匹配（不需要训练）
    print("\n" + "-" * 60)
    print("方法1: 关键词匹配")
    print("-" * 60)

    keyword_preds = [processor.predict(text, 'keyword') for text in test_texts]
    keyword_metrics = evaluate_predictions(test_labels, keyword_preds)
    print(f"准确率: {keyword_metrics['accuracy']:.4f}")
    print(f"各类别: {keyword_metrics['per_class']}")

    # 方法2: 统计分类器
    print("\n" + "-" * 60)
    print("方法2: 统计分类器")
    print("-" * 60)

    processor.train_statistical(train_texts, train_labels)
    statistical_preds = [processor.predict(text, 'statistical') for text in test_texts]
    statistical_metrics = evaluate_predictions(test_labels, statistical_preds)
    print(f"准确率: {statistical_metrics['accuracy']:.4f}")
    print(f"各类别: {statistical_metrics['per_class']}")

    # 保存结果
    results = {
        'keyword': keyword_metrics,
        'statistical': statistical_metrics,
        'dataset_stats': {
            'train_count': len(train_texts),
            'test_count': len(test_texts),
            'train_labels': Counter(train_labels),
            'test_labels': Counter(test_labels)
        }
    }

    output_dir = project_dir / "reports"
    output_dir.mkdir(exist_ok=True)
    results_path = output_dir / "model_results.json"

    with open(results_path, 'w', encoding='utf-8') as f:
        # 转换Counter为dict以支持JSON序列化
        import copy
        results_save = copy.deepcopy(results)
        results_save['dataset_stats']['train_labels'] = dict(results_save['dataset_stats']['train_labels'])
        results_save['dataset_stats']['test_labels'] = dict(results_save['dataset_stats']['test_labels'])
        json.dump(results_save, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存至: {results_path}")
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
