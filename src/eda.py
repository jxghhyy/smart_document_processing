"""
智能文档处理 - 完整的探索性数据分析
"""
import os
import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

# 设置字体 - 支持Windows/Mac/Linux
import platform
system_name = platform.system()

if system_name == 'Windows':
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
elif system_name == 'Darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'STHeiti']
else:  # Linux
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']

plt.rcParams['axes.unicode_minus'] = False


def explore_funsd_dataset(data_dir: str, output_dir: str):
    """探索FUNSD数据集"""
    print("=" * 60)
    print("FUNSD数据集探索")
    print("=" * 60)

    train_img_dir = os.path.join(data_dir, "FUNSD", "training_data", "images")
    train_ann_dir = os.path.join(data_dir, "FUNSD", "training_data", "annotations")
    test_img_dir = os.path.join(data_dir, "FUNSD", "testing_data", "images")
    test_ann_dir = os.path.join(data_dir, "FUNSD", "testing_data", "annotations")

    # 统计文件数量
    train_images = sorted([f for f in os.listdir(train_img_dir) if f.endswith('.png')])
    train_annotations = sorted([f for f in os.listdir(train_ann_dir) if f.endswith('.json')])
    test_images = sorted([f for f in os.listdir(test_img_dir) if f.endswith('.png')])
    test_annotations = sorted([f for f in os.listdir(test_ann_dir) if f.endswith('.json')])

    print(f"\n训练集图像数: {len(train_images)}")
    print(f"训练集标注数: {len(train_annotations)}")
    print(f"测试集图像数: {len(test_images)}")
    print(f"测试集标注数: {len(test_annotations)}")

    # 分析标注结构
    all_data = []
    all_labels = []
    all_words = []
    bbox_sizes = []
    image_sizes = []

    # 处理训练集
    for ann_file in train_annotations:
        ann_path = os.path.join(train_ann_dir, ann_file)
        with open(ann_path, 'r', encoding='utf-8') as f:
            ann = json.load(f)

        if 'form' in ann:
            for field in ann['form']:
                if 'text' in field and 'label' in field:
                    all_data.append({
                        'text': field['text'],
                        'label': field['label'],
                        'split': 'train'
                    })
                    all_labels.append(field['label'])

                if 'words' in field:
                    for word in field['words']:
                        if 'text' in word:
                            all_words.append(word['text'])
                        if 'box' in word:
                            bbox = word['box']
                            width = bbox[2] - bbox[0]
                            height = bbox[3] - bbox[1]
                            bbox_sizes.append((width, height))

        # 获取图像尺寸
        img_file = ann_file.replace('.json', '.png')
        img_path = os.path.join(train_img_dir, img_file)
        if os.path.exists(img_path):
            try:
                with Image.open(img_path) as img:
                    image_sizes.append(img.size)
            except:
                pass

    # 处理测试集
    for ann_file in test_annotations:
        ann_path = os.path.join(test_ann_dir, ann_file)
        with open(ann_path, 'r', encoding='utf-8') as f:
            ann = json.load(f)

        if 'form' in ann:
            for field in ann['form']:
                if 'text' in field and 'label' in field:
                    all_data.append({
                        'text': field['text'],
                        'label': field['label'],
                        'split': 'test'
                    })
                    all_labels.append(field['label'])

    print(f"\n总标注数: {len(all_labels)}")
    print(f"标签分布: {Counter(all_labels)}")
    print(f"总词数: {len(all_words)}")

    # 创建DataFrame
    df = pd.DataFrame(all_data)
    print(f"\n数据集预览:")
    print(df.head(10))

    # 保存统计
    df.to_csv(os.path.join(output_dir, 'funsd_data.csv'), index=False)

    # 可视化
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. 标签分布
    ax1 = fig.add_subplot(gs[0, 0])
    label_counts = df['label'].value_counts()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    ax1.bar(label_counts.index, label_counts.values, color=colors)
    ax1.set_title('Label Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Label', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    for i, v in enumerate(label_counts.values):
        ax1.text(i, v + 50, str(v), ha='center', fontsize=11)

    # 2. 文本长度分布
    ax2 = fig.add_subplot(gs[0, 1])
    text_lengths = df['text'].str.len()
    ax2.hist(text_lengths, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax2.set_title('Text Length Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Character Count', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)

    # 3. 词云图（用简单的频率图代替）
    ax3 = fig.add_subplot(gs[0, 2])
    word_counts = Counter(all_words).most_common(20)
    words, counts = zip(*word_counts)
    y_pos = np.arange(len(words))
    ax3.barh(y_pos, counts, color='coral', alpha=0.8)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(words, fontsize=9)
    ax3.set_title('Top 20 Frequent Words', fontsize=14, fontweight='bold')
    ax3.invert_yaxis()

    # 4. 边界框尺寸分布
    ax4 = fig.add_subplot(gs[1, 0])
    if bbox_sizes:
        widths, heights = zip(*bbox_sizes)
        ax4.scatter(widths, heights, alpha=0.3, s=20, color='purple')
        ax4.set_title('Text Bounding Box Size Distribution', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Width', fontsize=12)
        ax4.set_ylabel('Height', fontsize=12)
        ax4.grid(True, alpha=0.3)

    # 5. 图像尺寸分布
    ax5 = fig.add_subplot(gs[1, 1])
    if image_sizes:
        img_widths, img_heights = zip(*image_sizes)
        ax5.scatter(img_widths, img_heights, alpha=0.6, s=50, color='seagreen')
        ax5.set_title('Document Image Size Distribution', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Width', fontsize=12)
        ax5.set_ylabel('Height', fontsize=12)
        ax5.grid(True, alpha=0.3)

    # 6. 示例文档可视化
    ax6 = fig.add_subplot(gs[1, 2])
    if train_images:
        sample_img = os.path.join(train_img_dir, train_images[0])
        sample_ann = os.path.join(train_ann_dir, train_annotations[0])

        if os.path.exists(sample_img) and os.path.exists(sample_ann):
            with Image.open(sample_img) as img:
                ax6.imshow(img, cmap='gray')

                with open(sample_ann, 'r', encoding='utf-8') as f:
                    ann_data = json.load(f)

                color_map = {
                    'question': '#1f77b4',
                    'answer': '#2ca02c',
                    'header': '#d62728',
                    'other': '#ff7f0e'
                }

                for field in ann_data['form']:
                    label = field.get('label', 'other')
                    color = color_map.get(label, 'gray')

                    if 'words' in field:
                        for word in field['words']:
                            if 'box' in word:
                                box = word['box']
                                rect = patches.Rectangle(
                                    (box[0], box[1]),
                                    box[2] - box[0],
                                    box[3] - box[1],
                                    linewidth=1.5,
                                    edgecolor=color,
                                    facecolor='none',
                                    alpha=0.8
                                )
                                ax6.add_patch(rect)

            ax6.set_title(f'Example Document: {train_images[0]}', fontsize=14, fontweight='bold')
            ax6.axis('off')

    # 7. 更多示例
    for idx, img_idx in enumerate([1, 2]):
        if img_idx < len(train_images):
            ax = fig.add_subplot(gs[2, idx])
            sample_img = os.path.join(train_img_dir, train_images[img_idx])
            if os.path.exists(sample_img):
                with Image.open(sample_img) as img:
                    ax.imshow(img, cmap='gray')
                    ax.set_title(f'Document Example {img_idx + 1}', fontsize=12)
                    ax.axis('off')

    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#1f77b4', lw=2, label='question'),
        Line2D([0], [0], color='#2ca02c', lw=2, label='answer'),
        Line2D([0], [0], color='#d62728', lw=2, label='header'),
        Line2D([0], [0], color='#ff7f0e', lw=2, label='other')
    ]
    ax6.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.suptitle('FUNSD Dataset Exploratory Analysis', fontsize=18, fontweight='bold', y=0.995)
    plt.savefig(os.path.join(output_dir, 'funsd_eda.png'), dpi=150, bbox_inches='tight')
    print(f"\nEDA图表已保存至: {output_dir}/funsd_eda.png")


def explore_receipts_dataset(data_dir: str, output_dir: str):
    """探索收据数据集"""
    print("\n" + "=" * 60)
    print("收据数据集探索")
    print("=" * 60)

    receipt_dir = os.path.join(data_dir, "receipts")
    if os.path.exists(receipt_dir):
        receipt_files = sorted([f for f in os.listdir(receipt_dir)
                                if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        print(f"\n收据图像数: {len(receipt_files)}")

        # 统计图像尺寸
        image_sizes = []
        for img_file in receipt_files[:100]:
            img_path = os.path.join(receipt_dir, img_file)
            try:
                with Image.open(img_path) as img:
                    image_sizes.append(img.size)
            except:
                pass

        if image_sizes:
            widths, heights = zip(*image_sizes)
            print(f"\n图像尺寸统计:")
            print(f"  宽度: {np.mean(widths):.1f} ± {np.std(widths):.1f} (min: {min(widths)}, max: {max(widths)})")
            print(f"  高度: {np.mean(heights):.1f} ± {np.std(heights):.1f} (min: {min(heights)}, max: {max(heights)})")

        # 可视化收据样本
        if len(receipt_files) >= 12:
            fig, axes = plt.subplots(3, 4, figsize=(16, 14))
            axes = axes.ravel()

            for i, img_file in enumerate(receipt_files[:12]):
                img_path = os.path.join(receipt_dir, img_file)
                try:
                    with Image.open(img_path) as img:
                        axes[i].imshow(img, cmap='gray')
                        axes[i].set_title(f'{img_file}', fontsize=9)
                        axes[i].axis('off')
                except Exception as e:
                    pass

            plt.suptitle('Receipt Sample Examples', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'receipt_samples.png'), dpi=150, bbox_inches='tight')
            print(f"\n收据样本图已保存至: {output_dir}/receipt_samples.png")


def main():
    # 路径设置
    current_dir = Path(__file__).parent
    project_dir = current_dir.parent
    data_dir = project_dir / "data"
    output_dir = project_dir / "reports"
    output_dir.mkdir(exist_ok=True)

    print("智能文档处理数据集探索")
    print(f"项目目录: {project_dir}")

    # 探索FUNSD
    explore_funsd_dataset(str(data_dir), str(output_dir))

    # 探索收据数据集
    explore_receipts_dataset(str(data_dir), str(output_dir))

    print("\n" + "=" * 60)
    print("探索完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
