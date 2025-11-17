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
                 extra_args: List[str] = None,
                 search_freeze_layers: bool = False,
                 freeze_range: List[int] = None,
                 coarse_start_ratio: Tuple[int, int] = (2, 8)):
        """
        初始化搜索器

        Args:
            base_model: 基础模型路径
            pruning_ratio: 总剪枝率
            save_ckpt_log_name: 实验日志名称
            extra_args: 其他额外参数（不包括 --freeze_top_n_layers）
            search_freeze_layers: 是否搜索最优冻结层数
            freeze_range: 冻结层数搜索范围（默认[0,1,2,3,4,5,6,8]）
            coarse_start_ratio: 粗粒度搜索起点（默认2:8，基于LLaMA-3实际参数比例）
        """
        self.base_model = base_model
        self.pruning_ratio = pruning_ratio
        self.save_ckpt_log_name = save_ckpt_log_name or f"ppl_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.extra_args = extra_args or []
        self.search_freeze_layers = search_freeze_layers
        self.freeze_range = freeze_range or [0, 1, 2, 3, 4, 5, 6, 8]
        self.coarse_start_ratio = coarse_start_ratio

        # 存储结果
        self.results: Dict[str, float] = {}  # {ratio_str: ppl_value} 或 {ratio_str_freeze_N: ppl_value}

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

    def run_pruning(self, attn_ratio: float, mlp_ratio: float, freeze_layers: int = 0) -> Optional[float]:
        """
        运行单次剪枝实验并获取PPL

        Args:
            attn_ratio: Attention剪枝比例
            mlp_ratio: MLP剪枝比例
            freeze_layers: 冻结层数

        Returns:
            PPL值，如果失败返回None
        """
        ratio_str = f"{attn_ratio:.1f}:{mlp_ratio:.1f}"
        if freeze_layers > 0:
            result_key = f"{ratio_str}_freeze_{freeze_layers}"
            self.log(f"\n{'='*60}")
            self.log(f"开始实验: Attention:MLP = {ratio_str}, 冻结层数 = {freeze_layers}")
            self.log(f"{'='*60}")
        else:
            result_key = ratio_str
            self.log(f"\n{'='*60}")
            self.log(f"开始实验: Attention:MLP = {ratio_str}")
            self.log(f"{'='*60}")

        # 构建命令
        cmd = [
            "python", "llama3_unbalanced_pruning_gqa_aware.py",
            "--base_model", self.base_model,
            "--pruning_ratio", str(self.pruning_ratio),
            "--pruning_distribution", ratio_str,
            "--save_ckpt_log_name", f"{self.save_ckpt_log_name}_ratio_{ratio_str.replace(':', '_')}_freeze_{freeze_layers}",
            "--test_after_prune",
        ]

        # 添加冻结层数参数
        if freeze_layers > 0:
            cmd.extend(["--freeze_top_n_layers", str(freeze_layers)])

        # 添加其他额外参数
        cmd.extend(self.extra_args)

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
                if freeze_layers > 0:
                    self.log(f"✅ 实验完成: {ratio_str} (freeze={freeze_layers}) -> PPL = {ppl:.2f}")
                else:
                    self.log(f"✅ 实验完成: {ratio_str} -> PPL = {ppl:.2f}")
                self.results[result_key] = ppl
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

    def _should_early_stop(self, ppl_history: List[float], min_points: int = 2) -> bool:
        """
        判断是否应该早停

        条件：连续min_points次PPL都在增大，且增速加快（二阶导数为正）

        Args:
            ppl_history: PPL历史记录（最新的在最后）
            min_points: 至少需要多少个点来判断趋势

        Returns:
            True表示应该早停
        """
        if len(ppl_history) < min_points + 1:
            return False

        # 检查最近的min_points+1个点
        recent = ppl_history[-(min_points+1):]

        # 检查是否连续增大
        is_increasing = all(recent[i] > recent[i-1] for i in range(1, len(recent)))

        if not is_increasing:
            return False

        # 检查增速是否加快（二阶导数为正）
        if len(recent) >= 3:
            # 计算一阶导数（增量）
            deltas = [recent[i] - recent[i-1] for i in range(1, len(recent))]
            # 检查增量是否递增（增速加快）
            is_accelerating = all(deltas[i] > deltas[i-1] for i in range(1, len(deltas)))
            return is_accelerating

        return False

    def coarse_search(self) -> Tuple[Optional[str], Optional[str]]:
        """
        智能粗粒度搜索: 从中间开始，向两边搜索，自动早停

        策略:
        1. 从 5:5 开始
        2. 向左搜索 (4:6, 3:7, ..., 0:10)
        3. 向右搜索 (6:4, 7:3, ..., 10:0)
        4. 检测到PPL持续增大且加速时提前停止

        Returns:
            (best_ratio, second_best_ratio) 最佳和次佳比例
        """
        self.log("\n" + "="*60)
        self.log("阶段1: 智能粗粒度搜索 (步长=1, 带早停)")
        self.log("="*60)

        # 从智能起点开始（基于模型实际参数比例）
        center = self.coarse_start_ratio[0]
        center_mlp = self.coarse_start_ratio[1]
        self.log(f"\n从智能起点开始: {center}:{center_mlp} (基于模型实际Attention:MLP参数比例)")
        center_ppl = self.run_pruning(float(center), float(center_mlp))

        if center_ppl is None:
            self.log("❌ 中心点测试失败")
            return None, None

        # 向左搜索 (Attention减少，MLP增加)
        self.log(f"\n向左搜索 (减少Attention比例):")
        left_ppls = [center_ppl]
        left_ratios = [(center, center_mlp)]

        for attn in range(center - 1, -1, -1):
            mlp = 10 - attn
            self.log(f"  测试 {attn}:{mlp}")
            ppl = self.run_pruning(float(attn), float(mlp))

            if ppl is not None:
                left_ppls.append(ppl)
                left_ratios.append((attn, mlp))

                # 早停检测
                if self._should_early_stop(left_ppls, min_points=2):
                    self.log(f"  ⚠️  检测到PPL持续增大且加速，停止向左搜索")
                    self.log(f"     最近3次PPL: {left_ppls[-3:]}")
                    break
            else:
                self.log(f"  ⚠️  测试失败，跳过")

        # 向右搜索 (Attention增加，MLP减少)
        self.log(f"\n向右搜索 (增加Attention比例):")
        right_ppls = [center_ppl]
        right_ratios = [(center, center_mlp)]

        for attn in range(center + 1, 11):
            mlp = 10 - attn
            self.log(f"  测试 {attn}:{mlp}")
            ppl = self.run_pruning(float(attn), float(mlp))

            if ppl is not None:
                right_ppls.append(ppl)
                right_ratios.append((attn, mlp))

                # 早停检测
                if self._should_early_stop(right_ppls, min_points=2):
                    self.log(f"  ⚠️  检测到PPL持续增大且加速，停止向右搜索")
                    self.log(f"     最近3次PPL: {right_ppls[-3:]}")
                    break
            else:
                self.log(f"  ⚠️  测试失败，跳过")

        # 统计搜索效率
        total_possible = 11
        total_tested = len(self.results)
        saved_tests = total_possible - total_tested
        self.log(f"\n搜索效率统计:")
        self.log(f"  可能测试数: {total_possible}")
        self.log(f"  实际测试数: {total_tested}")
        self.log(f"  节省测试数: {saved_tests} ({saved_tests/total_possible*100:.1f}%)")

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

    def freeze_layers_search(self, best_ratio: str) -> int:
        """
        搜索最优冻结层数（阶段2：在最优分布下搜索）

        策略:
        1. 使用最优的剪枝分布
        2. 测试不同的冻结层数
        3. 使用早停机制检测PPL趋势

        Args:
            best_ratio: 最优剪枝分布（格式: "0.3:9.7"）

        Returns:
            最优冻结层数
        """
        self.log("\n" + "="*60)
        self.log("阶段3: 搜索最优冻结层数")
        self.log("="*60)
        self.log(f"使用最优剪枝分布: {best_ratio}")
        self.log(f"冻结层数搜索范围: {self.freeze_range}")

        # 解析最优比例
        best_attn = float(best_ratio.split(':')[0])
        best_mlp = float(best_ratio.split(':')[1])

        # 存储冻结层数搜索结果
        freeze_ppls = []
        freeze_results = {}

        for freeze_n in self.freeze_range:
            self.log(f"\n测试冻结层数 = {freeze_n}")
            ppl = self.run_pruning(best_attn, best_mlp, freeze_layers=freeze_n)

            if ppl is not None:
                freeze_ppls.append(ppl)
                freeze_results[freeze_n] = ppl

                # 早停检测（如果连续增大且加速）
                if self._should_early_stop(freeze_ppls, min_points=2):
                    self.log(f"  ⚠️  检测到PPL持续增大且加速，停止搜索")
                    self.log(f"     最近3次PPL: {freeze_ppls[-3:]}")
                    self.log(f"     提前终止，跳过剩余冻结层数测试")
                    break
            else:
                self.log(f"  ⚠️  测试失败，跳过")

        # 找出最优冻结层数
        if not freeze_results:
            self.log("❌ 没有任何有效的冻结层数结果")
            return 0

        best_freeze = min(freeze_results, key=freeze_results.get)
        best_freeze_ppl = freeze_results[best_freeze]

        self.log(f"\n" + "="*60)
        self.log("冻结层数搜索完成")
        self.log("="*60)
        self.log(f"✅ 最优冻结层数: {best_freeze} (PPL = {best_freeze_ppl:.2f})")

        # 显示所有冻结层数结果
        self.log(f"\n所有冻结层数结果（按PPL升序）:")
        sorted_freeze = sorted(freeze_results.items(), key=lambda x: x[1])
        for i, (freeze_n, ppl) in enumerate(sorted_freeze, 1):
            marker = "🏆" if freeze_n == best_freeze else "  "
            self.log(f"  {marker} {i}. freeze={freeze_n:2d} -> PPL = {ppl:7.2f}")

        # 统计搜索效率
        self.log(f"\n冻结层数搜索效率统计:")
        self.log(f"  搜索范围大小: {len(self.freeze_range)}")
        self.log(f"  实际测试数: {len(freeze_results)}")
        self.log(f"  节省测试数: {len(self.freeze_range) - len(freeze_results)}")

        return best_freeze

    def fine_search(self, center_ratio: str) -> str:
        """
        智能细粒度搜索: 从最优点向两边扩展，带早停

        策略:
        1. 从粗粒度的最优点开始
        2. 向左搜索（减少Attention）
        3. 向右搜索（增加Attention）
        4. 检测到PPL持续增大且加速时提前停止

        Args:
            center_ratio: 中心比例（粗粒度搜索的最优点）

        Returns:
            最佳比例
        """
        self.log("\n" + "="*60)
        self.log(f"阶段2: 智能细粒度搜索 (步长=0.1, 带早停)")
        self.log(f"从最优点开始: {center_ratio}")
        self.log("="*60)

        # 解析中心比例
        center_attn = float(center_ratio.split(':')[0])
        center_mlp = float(center_ratio.split(':')[1])
        center_ppl = self.results[center_ratio]

        # 向左搜索（减少Attention，步长0.1）
        self.log(f"\n向左精细搜索 (减少Attention):")
        left_ppls = [center_ppl]
        attn = center_attn - 0.1

        while attn >= 0:
            mlp = 10.0 - attn
            ratio_str = f"{attn:.1f}:{mlp:.1f}"
            self.log(f"  测试 {ratio_str}")

            ppl = self.run_pruning(attn, mlp)
            if ppl is not None:
                left_ppls.append(ppl)

                # 早停检测
                if self._should_early_stop(left_ppls, min_points=2):
                    self.log(f"  ⚠️  检测到PPL持续增大且加速，停止向左搜索")
                    self.log(f"     最近3次PPL: {left_ppls[-3:]}")
                    break
            else:
                self.log(f"  ⚠️  测试失败，跳过")

            attn -= 0.1
            attn = round(attn, 1)  # 避免浮点数精度问题

        # 向右搜索（增加Attention，步长0.1）
        self.log(f"\n向右精细搜索 (增加Attention):")
        right_ppls = [center_ppl]
        attn = center_attn + 0.1

        while attn <= 10.0:
            mlp = 10.0 - attn
            ratio_str = f"{attn:.1f}:{mlp:.1f}"
            self.log(f"  测试 {ratio_str}")

            ppl = self.run_pruning(attn, mlp)
            if ppl is not None:
                right_ppls.append(ppl)

                # 早停检测
                if self._should_early_stop(right_ppls, min_points=2):
                    self.log(f"  ⚠️  检测到PPL持续增大且加速，停止向右搜索")
                    self.log(f"     最近3次PPL: {right_ppls[-3:]}")
                    break
            else:
                self.log(f"  ⚠️  测试失败，跳过")

            attn += 0.1
            attn = round(attn, 1)  # 避免浮点数精度问题

        # 统计搜索效率（细粒度理论上最多10个点）
        theoretical_max = min(int((10.0 - 0) / 0.1) + 1, 101)  # 理论上最多101个点
        total_tested = len([k for k in self.results.keys() if '.' in k])  # 统计带小数的比例
        self.log(f"\n细粒度搜索效率统计:")
        self.log(f"  细粒度测试数: {total_tested}")

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

    def search(self) -> Tuple[Optional[str], Optional[float], Optional[int]]:
        """
        执行完整的两阶段（或三阶段）搜索

        阶段1: 粗粒度分布搜索（步长=1，智能双向+早停）
        阶段2: 细粒度分布搜索（步长=0.1，智能双向+早停）
        阶段3: 冻结层数搜索（可选，在最优分布下贪心搜索）

        Returns:
            (best_ratio, best_ppl, best_freeze)
        """
        start_time = datetime.now()
        self.log("\n" + "="*60)
        self.log("开始PPL优化搜索")
        self.log("="*60)
        self.log(f"基础模型: {self.base_model}")
        self.log(f"总剪枝率: {self.pruning_ratio:.2%}")
        self.log(f"搜索策略: {'三阶段贪心搜索（分布+冻结层）' if self.search_freeze_layers else '两阶段搜索（仅分布）'}")
        self.log(f"额外参数: {' '.join(self.extra_args)}")

        # 阶段1: 粗粒度搜索
        best_coarse_ratio, _ = self.coarse_search()

        if best_coarse_ratio is None:
            self.log("❌ 搜索失败")
            return None, None, None

        # 阶段2: 细粒度搜索（从粗粒度最优点开始）
        best_ratio = self.fine_search(best_coarse_ratio)

        best_ppl = self.get_best_ppl()

        # 阶段3: 冻结层数搜索（可选）
        best_freeze = 0
        if self.search_freeze_layers:
            best_freeze = self.freeze_layers_search(best_ratio)
            # 更新最优PPL（如果冻结层搜索找到了更好的）
            best_ppl = self.get_best_ppl()

        # 总结
        elapsed = datetime.now() - start_time
        self.log("\n" + "="*60)
        self.log("搜索完成")
        self.log("="*60)
        self.log(f"总耗时: {elapsed}")
        self.log(f"测试次数: {len(self.results)}")
        self.log(f"最优比例: {best_ratio}")
        if self.search_freeze_layers:
            self.log(f"最优冻结层数: {best_freeze}")
        self.log(f"最优PPL: {best_ppl:.2f}")
        self.log(f"结果已保存到: {self.result_file}")

        # 显示所有结果（按PPL排序）
        self.log("\n所有测试结果（按PPL升序）:")
        sorted_results = sorted(self.results.items(), key=lambda x: x[1])
        for i, (ratio, ppl) in enumerate(sorted_results[:10], 1):  # 只显示前10个
            marker = "🏆" if ppl == best_ppl else "  "
            self.log(f"  {marker} {i:2d}. {ratio:>20} -> PPL = {ppl:7.2f}")

        if len(sorted_results) > 10:
            self.log(f"  ... (共 {len(sorted_results)} 个结果)")

        return best_ratio, best_ppl, best_freeze


def main():
    parser = argparse.ArgumentParser(
        description="自动搜索最优 Attention:MLP 剪枝分布比例",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

1. 基本搜索（两阶段：分布优化）:
   python search_optimal_distribution.py \\
       --base_model /path/to/model

2. 指定剪枝率:
   python search_optimal_distribution.py \\
       --base_model /path/to/model \\
       --pruning_ratio 0.30

3. 三阶段搜索（分布+冻结层优化）:
   python search_optimal_distribution.py \\
       --base_model /path/to/model \\
       --search_freeze_layers \\
       --freeze_range 0,1,2,3,4,5,6,8

4. 自定义搜索起点（基于模型架构）:
   python search_optimal_distribution.py \\
       --base_model /path/to/model \\
       --coarse_start_ratio 3:7

5. 完整示例（三阶段+自定义配置）:
   python search_optimal_distribution.py \\
       --base_model /newdata/LLMs/Llama-3-8B-Instruct \\
       --pruning_ratio 0.25 \\
       --save_ckpt_log_name my_search \\
       --search_freeze_layers \\
       --freeze_range 0,1,2,3,4,5,6,8 \\
       --coarse_start_ratio 2:8 \\
       --layer_importance_method removal \\
       --prune_mlp

6. 固定冻结层数（非搜索模式）:
   python search_optimal_distribution.py \\
       --base_model /path/to/model \\
       --freeze_top_n_layers 3

搜索策略说明:
- 阶段1: 粗粒度分布搜索（步长=1，智能双向+早停）
- 阶段2: 细粒度分布搜索（步长=0.1，智能双向+早停）
- 阶段3: 冻结层数搜索（在最优分布下贪心搜索，需启用 --search_freeze_layers）

注意: 完整搜索可能需要数小时甚至更长时间（取决于模型大小和硬件）
      启用早停机制后，实际测试次数通常为总可能数的60-70%
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

    # 搜索策略参数
    parser.add_argument('--search_freeze_layers', action='store_true',
                       help='是否搜索最优冻结层数（阶段3，在最优分布下贪心搜索）')
    parser.add_argument('--freeze_range', type=str, default='0,1,2,3,4,5,6,8',
                       help='冻结层数搜索范围（逗号分隔，默认: 0,1,2,3,4,5,6,8）')
    parser.add_argument('--coarse_start_ratio', type=str, default='2:8',
                       help='粗粒度搜索起点（默认: 2:8，基于LLaMA-3实际Attention:MLP参数比例）')

    # 传递给剪枝脚本的额外参数
    parser.add_argument('--freeze_top_n_layers', type=int, default=None,
                       help='冻结重要度最高的n层（用于非搜索模式下的固定冻结）')
    parser.add_argument('--layer_importance_method', type=str, default=None,
                       choices=['removal', 'activation'],
                       help='层重要度计算方法')
    parser.add_argument('--pruning_strategy', type=str, default=None,
                       choices=['inverse', 'proportional', 'uniform'],
                       help='剪枝策略')
    parser.add_argument('--prune_mlp', action='store_true',
                       help='是否剪枝MLP（默认只剪Attention）')

    args = parser.parse_args()

    # 解析冻结层数范围
    freeze_range = [int(x.strip()) for x in args.freeze_range.split(',')]

    # 解析粗粒度搜索起点
    coarse_start_parts = args.coarse_start_ratio.split(':')
    if len(coarse_start_parts) != 2:
        print(f"❌ 错误: --coarse_start_ratio 格式应为 'X:Y' (例如 '2:8')")
        return 1
    coarse_start_ratio = (int(coarse_start_parts[0]), int(coarse_start_parts[1]))

    # 构建额外参数列表（不包括freeze_top_n_layers，因为会在搜索中动态设置）
    extra_args = []
    # 注意：如果启用了freeze layer search，则不使用固定的freeze_top_n_layers
    if not args.search_freeze_layers and args.freeze_top_n_layers is not None:
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
        extra_args=extra_args,
        search_freeze_layers=args.search_freeze_layers,
        freeze_range=freeze_range,
        coarse_start_ratio=coarse_start_ratio
    )

    # 执行搜索
    best_ratio, best_ppl, best_freeze = searcher.search()

    if best_ratio:
        print(f"\n🎉 搜索成功！")
        print(f"最优 Attention:MLP 比例: {best_ratio}")
        if args.search_freeze_layers:
            print(f"最优冻结层数: {best_freeze}")
        print(f"对应的 PPL: {best_ppl:.2f}")
        print(f"\n可使用以下命令进行最优配置剪枝:")
        print(f"python llama3_unbalanced_pruning_gqa_aware.py \\")
        print(f"    --base_model {args.base_model} \\")
        print(f"    --pruning_distribution {best_ratio} \\")
        print(f"    --pruning_ratio {args.pruning_ratio} \\")
        if args.search_freeze_layers and best_freeze > 0:
            print(f"    --freeze_top_n_layers {best_freeze} \\")
        if extra_args:
            print(f"    {' '.join(extra_args)} \\")
        print(f"    --save_model --test_after_prune")
    else:
        print("\n❌ 搜索失败，请检查日志")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
