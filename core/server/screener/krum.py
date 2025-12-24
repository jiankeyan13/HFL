import torch
import numpy as np
from typing import List, Dict, Any

from core.utils.registry import SCREENER_REGISTRY
from .base_screener import BaseScreener

@SCREENER_REGISTRY.register("krum")
class KrumScreener(BaseScreener):
    def __init__(self, f: int = 0, m: int = 1, **kwargs):
        """
        Args:
            f: 假设的攻击者数量 (Byzantine Tolerance)。
            m: Multi-Krum 最终保留的客户端数量。
               如果 m=1, 就是标准的 Krum。
        """
        super().__init__()
        self.f = f
        self.m = m

    def screen(self, updates: List[Dict[str, Any]], global_model=None) -> List[Dict[str, Any]]:
        """
        执行 Multi-Krum 筛选。
        updates: [{'client_id':..., 'weights':..., 'num_samples':...}, ...]
        """
        n = len(updates)
        
        # 边界检查：如果客户端太少，Krum 跑不起来
        # Krum 要求 n >= 2f + 3 (理论上)
        if n <= 2 * self.f + 2:
            # 可以在这里打印 warning 或者动态调整 f
            pass

        # 1. 展平参数 (Flatten Weights)
        # 为了算欧氏距离，必须把 state_dict 变成一个大向量
        vectors = []
        for up in updates:
            # 将所有 tensor 拼成一个一维向量
            flat = torch.cat([p.view(-1) for p in up['weights'].values()])
            vectors.append(flat)
        
        # 堆叠成矩阵 [n, d]
        vec_stack = torch.stack(vectors)
        
        # 2. 计算两两距离矩阵 (Pairwise Distance)
        # dists[i, j] = ||v_i - v_j||^2
        dists = torch.cdist(vec_stack, vec_stack, p=2)
        
        # 3. 计算 Krum Score
        # 对于每个 i，选最近的 n - f - 2 个邻居求和
        k = n - self.f - 2
        if k <= 0: k = 1 # 容错
        
        scores = []
        for i in range(n):
            # 拿到第 i 行的所有距离，从小到大排序
            # [1:] 是为了排除自己 (距离为0)
            sorted_dists, _ = torch.sort(dists[i])
            # 取最近的 k 个求和
            score = torch.sum(sorted_dists[1 : k+1])
            scores.append(score.item())
            
        # 4. 选择 Score 最小的 m 个索引
        # argsort 返回从小到大的索引
        sorted_indices = np.argsort(scores)
        selected_indices = sorted_indices[:self.m]
        
        # 5. 返回筛选后的 updates
        clean_updates = [updates[i] for i in selected_indices]
        
        return clean_updates