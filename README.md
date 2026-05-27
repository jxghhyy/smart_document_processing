# 基于多种机器学习算法的表单文档元素分类研究

华中科技大学研究生《高级机器学习理论》课程报告项目

## 项目概述

本项目针对智能文档处理中的表单元素分类任务，基于FUNSD数据集，对比研究了四种机器学习方法的性能：
- 关键词匹配（基线方法）
- 朴素贝叶斯 + TF-IDF
- 逻辑回归 + TF-IDF
- 随机森林 + TF-IDF

## 目录结构

```
smart_document_processing/
├── data/                    # 数据集目录
│   ├── FUNSD/              # FUNSD数据集
│   │   ├── training_data/
│   │   └── testing_data/
│   └── receipts/           # 收据图像（可选）
├── src/                    # 源代码目录
│   ├── eda.py            # 探索性数据分析
│   ├── models.py         # 模型训练与评估
│   └── simple_explore.py # 简化版探索脚本
├── reports/               # 实验结果与报告
│   ├── funsd_eda.png
│   ├── receipt_samples.png
│   ├── results.json
│   └── ...
├── report.tex            # LaTeX报告源文件
├── requirements.txt       # Python依赖
└── README.md            # 本文档
```

## 环境配置

### 创建Conda环境（推荐）

```bash
# 创建新环境
conda create -n doc_process python=3.9 -y
conda activate doc_process

# 安装依赖
pip install -r requirements.txt
```

### 使用现有环境

确保环境包含以下包：
- numpy >= 1.21.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- Pillow >= 8.3.0
- scikit-learn >= 1.0.0

## 快速开始

### 1. 探索性数据分析

```bash
cd src
python eda.py
```

生成数据统计和可视化图表，结果保存到 `reports/` 目录。

### 2. 训练模型并评估

```bash
cd src
python models.py
```

训练四个模型并进行对比评估，输出结果和可视化。

## 实验结果

| 模型 | 准确率 | 宏F1 |
|-----|-------|-----|
| 关键词匹配 | 14.24% | 9.52% |
| 朴素贝叶斯 + TF-IDF | 62.91% | 40.52% |
| 逻辑回归 + TF-IDF | 69.30% | 49.92% |
| 随机森林 + TF-IDF | 68.70% | 52.61% |

主要发现：
- 机器学习方法相比规则方法有巨大提升
- 逻辑回归在整体准确率上最优
- 随机森林在宏F1（类别平衡）上最优
- question和answer类别相对容易识别，header和other类别较难

## 报告编译

如果需要编译LaTeX报告：

```bash
# 确保安装了texlive或miktex
xelatex report.tex
# 可能需要编译两次以获得正确的引用
xelatex report.tex
```

## 数据集说明

### FUNSD数据集

FUNSD(Form Understanding in Noisy Scanned Documents)是表单理解领域的基准数据集，包含：
- 199份扫描表单文档（训练集149份，测试集50份）
- 9743个标注文本块
- 4个语义类别：question（问题）、answer（答案）、header（标题）、other（其他）

官方网站：https://guillaumejaume.github.io/FUNSD/

### 收据数据集

作为补充，还提供了收据图像数据集，但本报告主要使用FUNSD。

## 主要发现

1. **规则方法局限性大**：关键词匹配仅14.24%准确率
2. **统计学习提升显著**：朴素贝叶斯即达62.91%
3. **线性模型表现良好**：逻辑回归准确率最高(69.30%)
4. **集成模型类别平衡好**：随机森林宏F1最高(52.61%)
5. **类别不平衡是挑战**：header和other类别识别较难

## 引用

如果使用本项目，请引用：

```
@inproceedings{jaume2019funsd,
  title={FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents},
  author={Jaume, Guillaume and Ekenel, Hazim Kemal and Thiran, Jean-Philippe},
  booktitle={2019 International Conference on Document Analysis and Recognition Workshops (ICDARW)},
  volume={1},
  pages={1--6},
  year={2019},
  organization={IEEE}
}
```

## 作者

学号：\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

姓名：\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

专业：\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

指导教师：\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

院系：\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

## 许可证

MIT License

## 致谢

感谢课程老师和助教的指导！
