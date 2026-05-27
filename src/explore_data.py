"""
智能文档处理 - 探索性数据分析
"""
import os
import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from collections import defaultdict, Counter

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def explore_funsd_dataset(data_dir: str):
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
    if train_annotations:
        sample_ann_path = os.path.join(train_ann_dir, train_annotations[0])
        with open(sample_ann_path, 'r', encoding='utf-8') as f:
            sample_ann = json.load(f)

        print(f"\n示例标注文件: {train_annotations[0]}")
        print(f"标注结构keys: {list(sample_ann.keys())}")

        if 'form' in sample_ann:
            print(f"\n表单字段数量: {len(sample_ann['form'])}")
            if sample_ann['form']:
                print(f"第一个字段: {json.dumps(sample_ann['form'][0], indent=2, ensure_ascii=False)}")

    # 统计所有标注信息
    all_labels = []
    all_words = []
    bbox_sizes = []
    image_sizes = []

    # 合并训练和测试集进行统计
    all_ann_dirs = [train_ann_dir, test_ann_dir]
    all_img_dirs = [train_img_dir, test_img_dir]

    for ann_dir, img_dir in zip(all_ann_dirs, all_img_dirs):
        ann_files = [f for f in os.listdir(ann_dir) if f.endswith('.json')]
        for ann_file in ann_files:
            ann_path = os.path.join(ann_dir, ann_file)
            with open(ann_path, 'r', encoding='utf-8') as f:
                ann = json.load(f)

            if 'form' in ann:
                for field in ann['form']:
                    if 'label' in field:
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
            img_path = os.path.join(img_dir, img_file)
            if os.path.exists(img_path):
                try:
                    with Image.open(img_path) as img:
                        image_sizes.append(img.size)
                except:
                    pass

    print(f"\n标注统计:")
    print(f"总标签数: {len(all_labels)}")
    print(f"标签分布: {Counter(all_labels)}")
    print(f"总词数: {len(all_words)}")

    # 可视化标签分布
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 标签分布
    label_counts = Counter(all_labels)
    labels = list(label_counts.keys())
    counts = list(label_counts.values())
    axes[0, 0].bar(labels, counts, color='steelblue')
    axes[0, 0].set_title('FUNSD标签分布', fontsize=14)
    axes[0, 0].set_xlabel('标签', fontsize=12)
    axes[0, 0].set_ylabel('数量', fontsize=12)
    axes[0, 0].tick_params(axis='x', rotation=45)

    # 边界框尺寸分布
    if bbox_sizes:
        widths, heights = zip(*bbox_sizes)
        axes[0, 1].hist(widths, bins=50, alpha=0.7, color='coral', label='宽度')
        axes[0, 1].hist(heights, bins=50, alpha=0.7, color='seagreen', label='高度')
        axes[0, 1].set_title('文字边界框尺寸分布', fontsize=14)
        axes[0, 1].legend()
        axes[0, 1].set_xlabel('像素', fontsize=12)

    # 图像尺寸分布
    if image_sizes:
        img_widths, img_heights = zip(*image_sizes)
        axes[1, 0].scatter(img_widths, img_heights, alpha=0.6, color='purple', s=50)
        axes[1, 0].set_title('文档图像尺寸分布', fontsize=14)
        axes[1, 0].set_xlabel('宽度', fontsize=12)
        axes[1, 0].set_ylabel('高度', fontsize=12)
        axes[1, 0].grid(True, alpha=0.3)

    # 可视化一张示例文档
    if train_images and train_annotations:
        sample_img = os.path.join(train_img_dir, train_images[0])
        sample_ann = os.path.join(train_ann_dir, train_annotations[0])

        if os.path.exists(sample_img) and os.path.exists(sample_ann):
            with Image.open(sample_img) as img:
                axes[1, 1].imshow(img, cmap='gray')

                with open(sample_ann, 'r', encoding='utf-8') as f:
                    ann_data = json.load(f)

                # 绘制标注框
                if 'form' in ann_data:
                    color_map = {
                        'question': 'blue',
                        'answer': 'green',
                        'header': 'red',
                        'other': 'orange'
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
                                        linewidth=1,
                                        edgecolor=color,
                                        facecolor='none',
                                        alpha=0.7
                                    )
                                    axes[1, 1].add_patch(rect)

                axes[1, 1].set_title(f'示例文档: {train_images[0]}', fontsize=14)
                axes[1, 1].axis('off')

    plt.tight_layout()
    output_dir = os.path.join(os.path.dirname(os.path.dirname(data_dir)), 'reports')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'funsd_data_analysis.png'), dpi=150, bbox_inches='tight')
    print(f"\n分析图表已保存至: {output_dir}/funsd_data_analysis.png")


def explore_receipts_dataset(data_dir: str):
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
        for img_file in receipt_files[:100]:  # 采样前100张
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
        if len(receipt_files) >= 9:
            fig, axes = plt.subplots(3, 3, figsize=(15, 15))
            axes = axes.ravel()

            for i, img_file in enumerate(receipt_files[:9]):
                img_path = os.path.join(receipt_dir, img_file)
                try:
                    with Image.open(img_path) as img:
                        axes[i].imshow(img, cmap='gray')
                        axes[i].set_title(f'{img_file}', fontsize=10)
                        axes[i].axis('off')
                except Exception as e:
                    print(f"无法打开 {img_file}: {e}")

            plt.suptitle('收据样本示例', fontsize=16)
            plt.tight_layout()

            output_dir = os.path.join(os.path.dirname(os.path.dirname(data_dir)), 'reports')
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, 'receipt_samples.png'), dpi=150, bbox_inches='tight')
            print(f"\n收据样本图已保存至: {output_dir}/receipt_samples.png")


def main():
    # 数据目录
    current_dir = Path(__file__).parent
    project_dir = current_dir.parent
    data_dir = project_dir / "data"

    print("智能文档处理数据集探索")
    print(f"项目目录: {project_dir}")

    # 探索FUNSD
    explore_funsd_dataset(str(data_dir))

    # 探索收据数据集
    explore_receipts_dataset(str(data_dir))

    print("\n" + "=" * 60)
    print("探索完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
