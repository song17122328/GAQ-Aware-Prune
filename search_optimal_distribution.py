#!/usr/bin/env python3
"""
自动搜索最优 Attention:MLP 剪枝分布比例

两阶段搜索策略:
1. 粗粒度搜索: 0:10, 1:9, 2:8, ..., 10:0 (步长=1, 最多11次)
2. 细粒度搜索: 在PPL最小的两个相邻比例之间细化搜索 (步长=0.1, 最多10次)

目标: 找到使PPL最小的Attention:MLP比例
"""

import subprocess
import argparse
import os
import re
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional

class PPLSearcher:
    """PPL 优化搜索器"""

    def __init__(self,
                 base_model: str,
                 pruning_ratio: float = 0.25,
                 save_ckpt_log_name: str = None,
                 extra_args: List[str] = None):
        """
        初始化搜索器

        Args:
            base_model: 基础模型路径
            pruning_ratio: 总剪枝率
            save_ckpt_log_name: 实验日志名称
            extra_args: 其他额外参数（如 --freeze_top_n_layers 等）
        """
        self.base_model = base_model
        self.pruning_ratio = pruning_ratio
        self.save_ckpt_log_name = save_ckpt_log_name or f"ppl_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.extra_args = extra_args or []

        # 存储结果
        self.results: Dict[str, float] = {}  # {ratio_str: ppl_value}

        # 创建结果目录
        self.result_dir = os.path.join("prune_log", self.save_ckpt_log_name)
        os.makedirs(self.result_dir, exist_ok=True)

        self.result_file = os.path.join(self.result_dir, "search_results.json")
        self.log_file = os.path.join(self.result_dir, "search.log")

    def log(self, message: str):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)

        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_msg + "\n")

    def run_pruning(self, attn_ratio: float, mlp_ratio: float) -> Optional[float]:
        """
        运行单次剪枝实验并获取PPL

        Args:
            attn_ratio: Attention剪枝比例
            mlp_ratio: MLP剪枝比例

        Returns:
            PPL值，如果失败返回None
        """
        ratio_str = f"{attn_ratio:.1f}:{mlp_ratio:.1f}"
        self.log(f"\n{'='*60}")
        self.log(f"开始实验: Attention:MLP = {ratio_str}")
        self.log(f"{'='*60}")

        # 构建命令
        cmd = [
            "python", "llama3_unbalanced_pruning_gqa_aware.py",
            "--base_model", self.base_model,
            "--pruning_ratio", str(self.pruning_ratio),
            "--pruning_distribution", ratio_str,
            "--save_ckpt_log_name", f"{self.save_ckpt_log_name}_ratio_{ratio_str.replace(':', '_')}",
            "--test_after_prune",
        ] + self.extra_args

        self.log(f"执行命令: {' '.join(cmd)}")

        try:
            # 运行剪枝脚本
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1小时超时
            )

            # 从输出中提取PPL
            ppl = self._extract_ppl_from_output(result.stdout)

            if ppl is not None:
                self.log(f"✅ 实验完成: {ratio_str} -> PPL = {ppl:.2f}")
                self.results[ratio_str] = ppl
                self._save_results()
                return ppl
            else:
                self.log(f"❌ 无法从输出中提取PPL")
                return None

        except subprocess.TimeoutExpired:
            self.log(f"❌ 实验超时（>1小时）")
            return None
        except Exception as e:
            self.log(f"❌ 实验失败: {e}")
            return None

    def _extract_ppl_from_output(self, output: str) -> Optional[float]:
        """从脚本输出中提取PPL值"""
        # 查找 "剪枝后 PPL:" 或类似的模式
        patterns = [
            r"剪枝后\s+PPL:\s*\{[^}]*'wikitext2[^']*':\s*([\d.]+)",
            r"wikitext2[^:]*:\s*([\d.]+)",
            r"PPL.*?:\s*([\d.]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, output)
            if match:
                try:
                    ppl = float(match.group(1))
                    return ppl
                except:
                    continue

        return None

    def _save_results(self):
        """保存当前所有结果到JSON"""
        with open(self.result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'base_model': self.base_model,
                'pruning_ratio': self.pruning_ratio,
                'results': self.results,
                'best_ratio': self.get_best_ratio(),
                'best_ppl': self.get_best_ppl()
            }, f, indent=2, ensure_ascii=False)

    def get_best_ratio(self) -> Optional[str]:
        """获取当前最佳比例"""
        if not self.results:
            return None
        return min(self.results, key=self.results.get)

    def get_best_ppl(self) -> Optional[float]:
        """获取当前最佳PPL"""
        if not self.results:
            return None
        return min(self.results.values())

    def coarse_search(self) -> Tuple[Optional[str], Optional[str]]:
        """
        粗粒度搜索: 0:10 到 10:0，步长=1

        Returns:
            (best_ratio, second_best_ratio) 最佳和次佳比例
        """
        self.log("\n" + "="*60)
        self.log("阶段1: 粗粒度搜索 (步长=1)")
        self.log("="*60)

        for attn in range(11):  # 0 到 10
            mlp = 10 - attn
            self.run_pruning(float(attn), float(mlp))

        # 找出PPL最小的两个相邻比例
        if len(self.results) < 2:
            self.log("❌ 粗粒度搜索结果不足，无法进行细粒度搜索")
            return None, None

        # 按PPL排序
        sorted_results = sorted(self.results.items(), key=lambda x: x[1])

        self.log("\n粗粒度搜索结果（按PPL升序）:")
        for i, (ratio, ppl) in enumerate(sorted_results[:5], 1):
            self.log(f"  {i}. {ratio} -> PPL = {ppl:.2f}")

        # 找到最佳比例
        best_ratio = sorted_results[0][0]
        best_ppl = sorted_results[0][1]

        self.log(f"\n✅ 粗粒度搜索最佳: {best_ratio} (PPL = {best_ppl:.2f})")

        # 找到最佳比例的相邻比例中PPL次优的
        best_attn = float(best_ratio.split(':')[0])
        best_mlp = float(best_ratio.split(':')[1])

        # 检查左右邻居
        neighbors = []
        for attn_offset in [-1, 1]:
            neighbor_attn = best_attn + attn_offset
            neighbor_mlp = best_mlp - attn_offset
            if 0 <= neighbor_attn <= 10 and 0 <= neighbor_mlp <= 10:
                neighbor_ratio = f"{neighbor_attn:.1f}:{neighbor_mlp:.1f}"
                if neighbor_ratio in self.results:
                    neighbors.append((neighbor_ratio, self.results[neighbor_ratio]))

        if neighbors:
            # 选择PPL较小的邻居
            second_best_ratio = min(neighbors, key=lambda x: x[1])[0]
            self.log(f"选择相邻比例: {second_best_ratio}")
        else:
            # 如果没有邻居，选择第二小的PPL
            if len(sorted_results) > 1:
                second_best_ratio = sorted_results[1][0]
                self.log(f"选择次优比例: {second_best_ratio}")
            else:
                self.log("❌ 无法找到第二个比例")
                return best_ratio, None

        return best_ratio, second_best_ratio

    def fine_search(self, ratio1: str, ratio2: str) -> str:
        """
        细粒度搜索: 在两个比例之间，步长=0.1

        Args:
            ratio1: 第一个比例（格式: "2.0:8.0"）
            ratio2: 第二个比例（格式: "1.0:9.0"）

        Returns:
            最佳比例
        """
        self.log("\n" + "="*60)
        self.log(f"阶段2: 细粒度搜索 (步长=0.1)")
        self.log(f"搜索区间: {ratio1} 到 {ratio2}")
        self.log("="*60)

        # 解析比例
        attn1 = float(ratio1.split(':')[0])
        attn2 = float(ratio2.split(':')[0])

        # 确保attn1 < attn2
        if attn1 > attn2:
            attn1, attn2 = attn2, attn1

        # 在区间内搜索（不包括端点，因为已经测试过了）
        attn = attn1 + 0.1
        while attn < attn2 - 0.05:  # 0.05是为了避免浮点数精度问题
            mlp = 10.0 - attn
            self.run_pruning(attn, mlp)
            attn += 0.1

        # 找出所有结果中的最佳
        if not self.results:
            self.log("❌ 没有任何有效结果")
            return None

        best_ratio = self.get_best_ratio()
        best_ppl = self.get_best_ppl()

        self.log("\n" + "="*60)
        self.log("细粒度搜索完成")
        self.log("="*60)
        self.log(f"✅ 全局最优: {best_ratio} (PPL = {best_ppl:.2f})")

        return best_ratio

    def search(self) -> Tuple[Optional[str], Optional[float]]:
        """
        执行完整的两阶段搜索

        Returns:
            (best_ratio, best_ppl)
        """
        start_time = datetime.now()
        self.log("\n" + "="*60)
        self.log("开始PPL优化搜索")
        self.log("="*60)
        self.log(f"基础模型: {self.base_model}")
        self.log(f"总剪枝率: {self.pruning_ratio:.2%}")
        self.log(f"额外参数: {' '.join(self.extra_args)}")

        # 阶段1: 粗粒度搜索
        ratio1, ratio2 = self.coarse_search()

        if ratio1 is None:
            self.log("❌ 搜索失败")
            return None, None

        # 阶段2: 细粒度搜索（如果有第二个比例）
        if ratio2 is not None:
            best_ratio = self.fine_search(ratio1, ratio2)
        else:
            best_ratio = ratio1

        best_ppl = self.get_best_ppl()

        # 总结
        elapsed = datetime.now() - start_time
        self.log("\n" + "="*60)
        self.log("搜索完成")
        self.log("="*60)
        self.log(f"总耗时: {elapsed}")
        self.log(f"测试次数: {len(self.results)}")
        self.log(f"最优比例: {best_ratio}")
        self.log(f"最优PPL: {best_ppl:.2f}")
        self.log(f"结果已保存到: {self.result_file}")

        # 显示所有结果（按PPL排序）
        self.log("\n所有测试结果（按PPL升序）:")
        sorted_results = sorted(self.results.items(), key=lambda x: x[1])
        for i, (ratio, ppl) in enumerate(sorted_results, 1):
            marker = "🏆" if ratio == best_ratio else "  "
            self.log(f"  {marker} {i:2d}. {ratio:>8} -> PPL = {ppl:7.2f}")

        return best_ratio, best_ppl


def main():
    parser = argparse.ArgumentParser(
        description="自动搜索最优 Attention:MLP 剪枝分布比例",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

1. 基本搜索（默认参数）:
   python search_optimal_distribution.py \\
       --base_model /path/to/model

2. 指定剪枝率:
   python search_optimal_distribution.py \\
       --base_model /path/to/model \\
       --pruning_ratio 0.30

3. 启用层冻结:
   python search_optimal_distribution.py \\
       --base_model /path/to/model \\
       --freeze_top_n_layers 3

4. 完整示例:
   python search_optimal_distribution.py \\
       --base_model /newdata/LLMs/Llama-3-8B-Instruct \\
       --pruning_ratio 0.25 \\
       --save_ckpt_log_name my_search \\
       --freeze_top_n_layers 3 \\
       --layer_importance_method removal

注意: 完整搜索可能需要数小时甚至更长时间（取决于模型大小和硬件）
        """
    )

    # 必需参数
    parser.add_argument('--base_model', type=str, required=True,
                       help='基础模型路径')

    # 可选参数
    parser.add_argument('--pruning_ratio', type=float, default=0.25,
                       help='总剪枝率（默认: 0.25）')
    parser.add_argument('--save_ckpt_log_name', type=str, default=None,
                       help='实验日志名称（默认: ppl_search_<timestamp>）')

    # 传递给剪枝脚本的额外参数
    parser.add_argument('--freeze_top_n_layers', type=int, default=None,
                       help='冻结重要度最高的n层')
    parser.add_argument('--layer_importance_method', type=str, default=None,
                       choices=['removal', 'activation'],
                       help='层重要度计算方法')
    parser.add_argument('--pruning_strategy', type=str, default=None,
                       choices=['inverse', 'proportional', 'uniform'],
                       help='剪枝策略')
    parser.add_argument('--prune_mlp', action='store_true',
                       help='是否剪枝MLP（默认只剪Attention）')

    args = parser.parse_args()

    # 构建额外参数列表
    extra_args = []
    if args.freeze_top_n_layers is not None:
        extra_args.extend(['--freeze_top_n_layers', str(args.freeze_top_n_layers)])
    if args.layer_importance_method is not None:
        extra_args.extend(['--layer_importance_method', args.layer_importance_method])
    if args.pruning_strategy is not None:
        extra_args.extend(['--pruning_strategy', args.pruning_strategy])
    if args.prune_mlp:
        extra_args.append('--prune_mlp')

    # 创建搜索器
    searcher = PPLSearcher(
        base_model=args.base_model,
        pruning_ratio=args.pruning_ratio,
        save_ckpt_log_name=args.save_ckpt_log_name,
        extra_args=extra_args
    )

    # 执行搜索
    best_ratio, best_ppl = searcher.search()

    if best_ratio:
        print(f"\n🎉 搜索成功！")
        print(f"最优 Attention:MLP 比例: {best_ratio}")
        print(f"对应的 PPL: {best_ppl:.2f}")
        print(f"\n可使用以下命令进行最优比例剪枝:")
        print(f"python llama3_unbalanced_pruning_gqa_aware.py \\")
        print(f"    --base_model {args.base_model} \\")
        print(f"    --pruning_distribution {best_ratio} \\")
        print(f"    --pruning_ratio {args.pruning_ratio} \\")
        if extra_args:
            print(f"    {' '.join(extra_args)} \\")
        print(f"    --save_model --test_after_prune")
    else:
        print("\n❌ 搜索失败，请检查日志")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
