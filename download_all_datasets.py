#!/usr/bin/env python3
"""
预下载所有评估所需的数据集,请先运行 HF_ENDPOINT="https://hf-mirror.com"

背景：
- 新版datasets库（v3.0+）不再支持legacy loading scripts
- 旧路径如 'piqa' 会报错: "Dataset scripts are no longer supported"
- 需要使用新的数据集仓库路径，如 'ybisk/piqa'

解决方案：
1. 使用本脚本预下载（使用正确路径）
2. 或清理损坏缓存让系统重新下载

用法：
    # 预下载所有数据集
    HF_ENDPOINT="https://hf-mirror.com"
    python download_all_datasets.py

    # 如果之前下载失败，先清理缓存
    python evaluation/clean_dataset_cache.py --all
    python download_all_datasets.py

注意：
- lm-eval会自动下载数据集，本脚本是可选的
- 适合网络不稳定时预下载，避免评估中断
"""

from datasets import load_dataset

def download_dataset(name, path, config=None, split='test'):
    """下载单个数据集"""
    try:
        print(f'\n{"="*60}')
        print(f'下载 {name}...')
        print(f'{"="*60}')

        if config:
            dataset = load_dataset(path, config, split=split, trust_remote_code=True)
        else:
            dataset = load_dataset(path, split=split, trust_remote_code=True)

        print(f'✓ {name} 下载完成')
        print(f'  样本数: {len(dataset):,}')
        return True

    except Exception as e:
        print(f'✗ {name} 下载失败: {e}')
        return False


def main():
    print("开始下载评估数据集...,如果没有配置镜像，请先配置镜像‘HF_ENDPOINT='https://hf-mirror.com'’")
    print("提示：数据集会缓存到 ~/.cache/huggingface/datasets/")

    success = []
    failed = []

    # PPL数据集
    print("\n" + "="*60)
    print("PPL 评估数据集")
    print("="*60)

    # WikiText-2
    if download_dataset('WikiText-2', 'wikitext', 'wikitext-2-raw-v1', 'test'):
        success.append('WikiText-2')
    else:
        failed.append('WikiText-2')

    # C4 (可选，较大)
    print('\n提示：C4数据集较大，跳过。如需下载，请手动运行：')
    print('  load_dataset("allenai/c4", "en", split="validation", streaming=False)')

    # Zero-shot数据集
    print("\n" + "="*60)
    print("Zero-shot 评估数据集")
    print("="*60)

    # HellaSwag
    if download_dataset('HellaSwag', 'Rowan/hellaswag', split='validation'):
        success.append('HellaSwag')
    else:
        failed.append('HellaSwag')

    # PIQA - 使用新路径
    if download_dataset('PIQA', 'ybisk/piqa', split='validation'):
        success.append('PIQA')
    else:
        failed.append('PIQA')

    # WinoGrande
    if download_dataset('WinoGrande', 'winogrande', 'winogrande_xl', 'validation'):
        success.append('WinoGrande')
    else:
        failed.append('WinoGrande')

    # ARC-Easy
    if download_dataset('ARC-Easy', 'allenai/ai2_arc', 'ARC-Easy', 'test'):
        success.append('ARC-Easy')
    else:
        failed.append('ARC-Easy')

    # BoolQ
    if download_dataset('BoolQ', 'google/boolq', split='validation'):
        success.append('BoolQ')
    else:
        failed.append('BoolQ')

    # 总结
    print("\n" + "="*60)
    print("下载完成！")
    print("="*60)
    print(f"✓ 成功: {len(success)} 个")
    for name in success:
        print(f"  - {name}")

    if failed:
        print(f"\n✗ 失败: {len(failed)} 个")
        for name in failed:
            print(f"  - {name}")
        print("\n建议：失败的数据集可以跳过，使用成功下载的数据集进行评估")
    else:
        print("\n所有数据集下载成功！")

    print("\n缓存位置: ~/.cache/huggingface/datasets/")
    print("可以使用以下命令查看占用空间：")
    print("  du -sh ~/.cache/huggingface/datasets/")


if __name__ == '__main__':
    main()
