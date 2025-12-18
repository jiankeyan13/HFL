import os
import torch
import numpy as np
import random
import copy
from functools import partial
from typing import Dict, Any, List, Optional

from core.utils.logger import Logger
from core.utils.scheduler import build_scheduler
import core.utils.metrics
import models
import core.server.aggregator
import core.server.screener
import core.server.updater
import algorithms
from core.utils.registry import (
    MODEL_REGISTRY, 
    AGGREGATOR_REGISTRY, 
    SCREENER_REGISTRY, 
    UPDATER_REGISTRY,
    ALGORITHM_REGISTRY,
    METRIC_REGISTRY
)

from data.task_generator import TaskGenerator
from core.client.base_client import BaseClient
from core.server.base_server import BaseServer

import data.datasets

class FederatedRunner:
    """
    联邦学习主控循环 (Simulator)。
    职责：
    1. 初始化 (数据、模型、服务器、日志)。
    2. 执行 Round 循环 (选人 -> 训练 -> 聚合 -> 评估)。
    3. 管理全局状态 (轮次、学习率、Checkpoints)。
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.logger = Logger(
            project_name=config.get('project', 'FL_Project'),
            experiment_name=config.get('name', 'experiment'),
            config=config,
            use_wandb=config.get('use_wandb', False)
        )
        self.logger.info(f"Runner is configured to use device: {self.device}")
        self._set_seed(config.get('seed', 42))
        
        self._setup()

    def _set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _setup(self):
        self.logger.info(">>> Initializing components...")

        # 准备数据
        # 假设 config['data'] 包含了 root, dataset_name, partitioner 等参数
        from data.partitioner import DirichletPartitioner, IIDPartitioner # 临时引入
        global_seed = self.config.get('seed', 42)
        part_conf = self.config['data']['partitioner']
        if part_conf['name'] == 'dirichlet':
            partitioner = DirichletPartitioner(alpha=part_conf.get('alpha', 0.5), seed = global_seed)
        else:
            partitioner = IIDPartitioner(seed = global_seed)

        self.task_generator = TaskGenerator(
            dataset_name=self.config['data']['dataset'],
            root=self.config['data']['root'],
            partitioner=partitioner,
            num_clients=self.config['data']['num_clients'],
            val_ratio=self.config['data'].get('val_ratio', 0.1),
            seed = global_seed
        )
        # 生成任务和数据源
        self.task_set, self.dataset_stores = self.task_generator.generate()
        self.logger.info(f"Data setup complete. Clients: {self.config['data']['num_clients']}")

        # 准备模型构建函数 (Model Fn)
        model_conf = self.config['model']
        model_cls = MODEL_REGISTRY.get(model_conf['name'])
        self.model_fn = partial(model_cls, **model_conf.get('params', {}))
        self.global_model = self.model_fn().to(self.device)

        server_test_task = self.task_set.get_task("server", "test_global")
        algo_conf = self.config['algorithm'] # e.g. {'name': 'fedavg', 'params': {...}}
        self.server, self.client_class = ALGORITHM_REGISTRY.build(
            algo_conf['name'],
            # --- 传递给 build_xxx_algorithm 的参数 ---
            model=self.global_model,
            device=self.device,
            dataset_store=self.dataset_stores, # 传下去，方便 Server 构建测试集
            config=self.config,                # 传下去，方便解析 server/updater 配置
            server_test_task=server_test_task, # 传下去
            **algo_conf.get('params', {})
        )

        self.logger.info(f"Algorithm Loaded: {algo_conf['name']}")
        self.logger.info(f"Server Type: {type(self.server).__name__}")
        self.logger.info(f"Client Type: {self.client_class.__name__}")

        # 学习率调度器
        self.lr_scheduler = build_scheduler(self.config['training'])

    def run(self):
        total_rounds = self.config['training']['rounds']
        eval_interval = self.config['training'].get('eval_interval', 5)
        
        self.logger.info(">>> Start Training...")
        
        best_acc = 0.0

        # 从 TaskSet 获取所有客户端 ID，供后续采样
        all_client_ids = [cid for cid in self.task_set._tasks.keys() if cid != 'server']

        for round_idx in range(total_rounds):
            self.logger.info(f"--- Round {round_idx} ---")

            # 计算全局 LR
            current_lr = self.lr_scheduler.get_lr(round_idx)
            round_config = copy.deepcopy(self.config['client'])
            round_config['lr'] = current_lr
            round_config['current_round'] = round_idx

            # 控制客户端选择
            num_select = self.config['training']['clients_per_round']
            
            # 获取攻击者配置
            attack_conf = self.config.get('attack', {})
            attacker_ids = attack_conf.get('clients', [])
            num_attackers_per_round = attack_conf.get('num_per_round', 0)
            
            # 选人
            selected_attackers = random.sample(attacker_ids, k=min(num_attackers_per_round, len(attacker_ids)))
            benign_ids = [cid for cid in all_client_ids if cid not in attacker_ids]
            num_benign_to_select = num_select - len(selected_attackers)
            selected_benign = random.sample(benign_ids, k=min(num_benign_to_select, len(benign_ids)))
            selected_ids = selected_attackers + selected_benign
            random.shuffle(selected_ids)

            self.logger.info(f"Selected clients ({len(selected_ids)}): {selected_ids}")
            
            # Broadcast
            client_models = self.server.broadcast(selected_ids)

            # Local Training
            updates = self._run_local_training(selected_ids, client_models, round_config)

            # Server Step
            self.server.step(updates)
            train_metrics = self.server.aggregate_metrics(updates)
            self.logger.log_metrics(train_metrics, step=round_idx)

            # 全局评估
            global_metric_confs = self.config.get('evaluation', {}).get('global', [{'name': 'acc'}])
            metric_objs = self._build_metrics(global_metric_confs)
            test_metrics = self.server.eval_global(metrics=metric_objs)
        
            self.logger.log_metrics(test_metrics, step=round_idx)
            
            # 优化日志打印
            log_msg = "Global Eval: "
            for k, v in test_metrics.items():
                log_msg += f"{k}={v:.4f} "
            self.logger.info(log_msg)

            # 本地抽样评估 (Local Eval)
            if round_idx % eval_interval == 0:
                local_metric_confs = self.config.get('evaluation', {}).get('local', [{'name': 'acc'}])
                local_metrics = self._build_metrics(local_metric_confs)
                self._run_local_evaluation(round_idx, round_config, local_metrics)

            # 保存 Checkpoint
            if test_metrics.get('acc', 0) > best_acc:
                best_acc = test_metrics['acc']
                self._save_checkpoint(round_idx, is_best=True)

        self.logger.info(f"Training Finished. Best Acc: {best_acc:.4f}")
        self.logger.close()
    def _run_local_training(self, client_ids, client_models, config):
        """
        执行本地训练循环。
        """
        updates = []
        
        for cid in client_ids:
            attack_profile = self._get_attack_profile(cid, config['current_round'])
            
            client = self.client_class(cid, self.device, self.model_fn)
            task = self.task_set.get_task(cid, "train")
            store = self.dataset_stores[task.dataset_tag]
            
            payload = client.execute(
                global_state_dict=client_models[cid],
                task=task,
                dataset_store=store,
                config=config,
                attack_profile=attack_profile
            )
            
            updates.append(payload)
            
            # 5. 显式销毁 (Python GC 会处理，但显式删除更保险)
            del client
            
        return updates

    def _run_local_evaluation(self, round_idx, config, metrics):
        """
        抽样评估客户端本地性能。
        Args:
            metrics: 由 _build_metrics 构建好的 Metric 对象列表
        """
        
        all_clients = list(self.task_set._tasks.keys())
        
        client_candidates = [c for c in all_clients if c != 'server']
        
        eval_frac = self.config.get('evaluation', {}).get('local', {}).get('sample_frac', 0.2)
        eval_ids = random.sample(client_candidates, k=max(1, int(len(client_candidates) * eval_frac)))

        results_collector = {m.name: [] for m in metrics}
        
        for cid in eval_ids:
            # 使用动态 Client 类
            client = self.client_class(cid, self.device, self.model_fn)
            task = self.task_set.get_task(cid, "test") 
            store = self.dataset_stores[task.dataset_tag]
            
            # 获取模型
            model_dict = self.server.broadcast([cid])[cid]
            
            # 执行评估
            # 这里的 metrics 是外面传进来的通用列表
            res = client.evaluate(
                global_state_dict=model_dict,
                task=task,
                dataset_store=store,
                config=config,
                metrics=metrics
            )
            
            # 收集结果
            for key, val in res.items():
                if key in results_collector:
                    results_collector[key].append(val)
            
            del client
            
        log_dict = {}
        for metric_name, values in results_collector.items():
            if len(values) > 0:
                log_dict[f"local/avg_{metric_name}"] = np.mean(values)
                log_dict[f"local/std_{metric_name}"] = np.std(values)
                
        self.logger.log_metrics(log_dict, step=round_idx)

    def _get_attack_profile(self, client_id: str, round_idx: int):
        """
        获取攻击配置。
        目前返回 None (无攻击)。
        未来在这里接入 AttackManager。
        """
        # 示例逻辑：
        # if client_id in self.attacker_list:
        #     return ATTACK_REGISTRY.build("badnets", target_label=0)
        return None

    def _save_checkpoint(self, round_idx, is_best=False):

        state = {
            'round': round_idx,
            'model_state_dict': self.global_model.state_dict(),
            'config': self.config
        }
        filename = "checkpoint_best.pth" if is_best else f"checkpoint_{round_idx}.pth"
        path = os.path.join(self.logger.run_dir, filename)
        torch.save(state, path)
    def _build_metrics(self, metric_configs: List[Dict]) -> List[Any]:
        """
        根据配置列表构建 Metric 对象
        """
        metrics = []
        for conf in metric_configs:
            name = conf['name']
            params = conf.get('params', {})
            # 从注册表构建
            metrics.append(METRIC_REGISTRY.build(name, **params))
        return metrics