#!/usr/bin/env python3
"""
ARC (AI2 Reasoning Challenge) 数据集预加载脚本

从 AI2 官方源下载 ARC 数据集并保存到本地缓存
解决 HuggingFace 上数据集下载问题

用法:
    python evaluation/preload_arc.py
"""

import os
import json
import zipfile
import requests
from datasets import Dataset, DatasetDict

# AI2 官方数据下载链接
ARC_DOWNLOAD_URL = "https://s3-us-west-2.amazonaws.com/ai2-website/data/ARC-V1-Feb2018.zip"


def download_file(url, local_path):
    """
    下载文件到本地（支持代理，带进度显示）

    Args:
        url: 远程文件 URL
        local_path: 本地保存路径
    """
    print(f"  下载: {url.split('/')[-1]}")
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0

    with open(local_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                print(f"\r  进度: {percent:.1f}% ({downloaded // 1024 // 1024}MB / {total_size // 1024 // 1024}MB)", end='')

    print()  # 换行
    return local_path


def load_arc_jsonl(file_path):
    """
    加载 ARC JSONL 文件

    Args:
        file_path: JSONL 文件路径

    Returns:
        list: 问题列表
    """
    questions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


def convert_to_hf_format(questions):
    """
    将 ARC 格式转换为 HuggingFace 格式

    Args:
        questions: ARC 问题列表

    Returns:
        dict: HF 格式的数据字典
    """
    data = {
        'id': [],
        'question': [],
        'choices': [],
        'answerKey': []
    }

    for q in questions:
        data['id'].append(q.get('id', ''))
        data['question'].append(q.get('question', {}).get('stem', ''))

        # 处理选项
        choices = q.get('question', {}).get('choices', [])
        choice_dict = {
            'text': [c.get('text', '') for c in choices],
            'label': [c.get('label', '') for c in choices]
        }
        data['choices'].append(choice_dict)

        data['answerKey'].append(q.get('answerKey', ''))

    return data


def preload_arc():
    """
    从 AI2 官方源下载 ARC 数据集并保存到本地
    """
    print("=" * 60)
    print("ARC 数据集预加载 (从 AI2 官方源)")
    print("=" * 60)

    try:
        # 创建本地下载目录
        local_data_dir = os.path.expanduser("~/.cache/huggingface/datasets/arc_local")
        os.makedirs(local_data_dir, exist_ok=True)

        # 1. 下载 ARC 压缩包
        zip_path = os.path.join(local_data_dir, "ARC-V1-Feb2018.zip")
        extract_dir = os.path.join(local_data_dir, "ARC-V1-Feb2018-2")

        if not os.path.exists(extract_dir):
            print("\n[1/4] 下载 ARC 数据集...")
            download_file(ARC_DOWNLOAD_URL, zip_path)
            print("✓ 下载完成")

            # 解压
            print("\n[2/4] 解压数据集...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(local_data_dir)
            print("✓ 解压完成")
        else:
            print("\n[1/4] 使用已下载的数据...")
            print("[2/4] 跳过解压...")

        # 3. 加载并转换数据
        print("\n[3/4] 加载并转换数据...")

        # ARC-Easy
        arc_easy_dir = os.path.join(extract_dir, "ARC-Easy")
        arc_easy_train = load_arc_jsonl(os.path.join(arc_easy_dir, "ARC-Easy-Train.jsonl"))
        arc_easy_dev = load_arc_jsonl(os.path.join(arc_easy_dir, "ARC-Easy-Dev.jsonl"))
        arc_easy_test = load_arc_jsonl(os.path.join(arc_easy_dir, "ARC-Easy-Test.jsonl"))

        print(f"  ARC-Easy: train={len(arc_easy_train)}, dev={len(arc_easy_dev)}, test={len(arc_easy_test)}")

        # ARC-Challenge
        arc_challenge_dir = os.path.join(extract_dir, "ARC-Challenge")
        arc_challenge_train = load_arc_jsonl(os.path.join(arc_challenge_dir, "ARC-Challenge-Train.jsonl"))
        arc_challenge_dev = load_arc_jsonl(os.path.join(arc_challenge_dir, "ARC-Challenge-Dev.jsonl"))
        arc_challenge_test = load_arc_jsonl(os.path.join(arc_challenge_dir, "ARC-Challenge-Test.jsonl"))

        print(f"  ARC-Challenge: train={len(arc_challenge_train)}, dev={len(arc_challenge_dev)}, test={len(arc_challenge_test)}")

        # 转换为 HF 格式
        arc_easy_dataset = DatasetDict({
            'train': Dataset.from_dict(convert_to_hf_format(arc_easy_train)),
            'validation': Dataset.from_dict(convert_to_hf_format(arc_easy_dev)),
            'test': Dataset.from_dict(convert_to_hf_format(arc_easy_test))
        })

        arc_challenge_dataset = DatasetDict({
            'train': Dataset.from_dict(convert_to_hf_format(arc_challenge_train)),
            'validation': Dataset.from_dict(convert_to_hf_format(arc_challenge_dev)),
            'test': Dataset.from_dict(convert_to_hf_format(arc_challenge_test))
        })

        print("✓ 数据转换完成")

        # 4. 保存数据集
        print("\n[4/4] 保存数据集...")

        # 保存 ARC-Easy
        arc_easy_save_dir = os.path.join(local_data_dir, "arc_easy")
        arc_easy_dataset.save_to_disk(arc_easy_save_dir)
        print(f"✓ ARC-Easy 已保存: {arc_easy_save_dir}")

        # 保存 ARC-Challenge
        arc_challenge_save_dir = os.path.join(local_data_dir, "arc_challenge")
        arc_challenge_dataset.save_to_disk(arc_challenge_save_dir)
        print(f"✓ ARC-Challenge 已保存: {arc_challenge_save_dir}")

        # 验证样本
        print(f"\n样本示例 (ARC-Challenge):")
        sample = arc_challenge_dataset['test'][0]
        print(f"  id: {sample['id']}")
        print(f"  question: {sample['question'][:80]}...")
        print(f"  choices: {sample['choices']['text']}")
        print(f"  answerKey: {sample['answerKey']}")

        print("\n" + "=" * 60)
        print("✓ ARC 预加载完成!")
        print("=" * 60)
        print("\nARC 数据集现在可用于 zero-shot 评估")
        print("注意: lm-eval 可能仍需要在线下载，这只是备份方案")

        return True

    except requests.exceptions.RequestException as e:
        print(f"\n✗ 网络下载失败: {e}")
        print("请检查网络连接或使用代理")
        return False

    except Exception as e:
        print(f"\n✗ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = preload_arc()
    exit(0 if success else 1)
