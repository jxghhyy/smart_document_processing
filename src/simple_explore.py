"""
智能文档处理 - 简化的探索性数据分析
"""
import os
import json
from pathlib import Path
from collections import defaultdict, Counter


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

    # 合并训练和测试集进行统计
    all_ann_dirs = [train_ann_dir, test_ann_dir]

    for ann_dir in all_ann_dirs:
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

    print(f"\n标注统计:")
    print(f"总标签数: {len(all_labels)}")
    print(f"标签分布: {Counter(all_labels)}")
    print(f"总词数: {len(all_words)}")

    # 保存统计结果
    output_dir = os.path.join(os.path.dirname(os.path.dirname(data_dir)), 'reports')
    os.makedirs(output_dir, exist_ok=True)

    stats = {
        'train_images': len(train_images),
        'test_images': len(test_images),
        'label_distribution': dict(Counter(all_labels)),
        'total_words': len(all_words)
    }

    stats_path = os.path.join(output_dir, 'funsd_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n统计数据已保存至: {stats_path}")


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
        print(f"\n前10个文件: {receipt_files[:10]}")


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
