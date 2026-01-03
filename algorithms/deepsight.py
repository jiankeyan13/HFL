from typing import Tuple, Type
from torch.utils.data import Subset, DataLoader

from core.server.base_server import BaseServer
from core.client.base_client import BaseClient
from core.server.screener.deepsight_screener import DeepSightScreener
from core.server.aggregator.flame_aggregator import FlameAggregator
from core.server.updater.base_updater import BaseUpdater
from core.utils.registry import ALGORITHM_REGISTRY

"""
12/27/2025: 22ndss-DeepSight 在V.C部分采用了后处理方式(
    簇内独立聚合:在训练的最后一轮, DeepSight 不再分发一个通用的全局模型给所有人。\
        它会根据第一阶段聚类出来的簇，为每个簇分别生成一个聚合模型。)
    但由于对特定时间点的需求难以控制, 且后续实验未做消融实验, 因此未采用此方法。(值得一提的是,大部分非官方实现都未做后处理)
    参考:
    1. https://github.com/hassanalikhatim/AGSD
"""

@ALGORITHM_REGISTRY.register("deepsight")
def build_deepsight_algorithm(model, device, dataset_store, config, **kwargs) -> Tuple[BaseServer, Type[BaseClient]]:
    """
    DeepSight 组装：使用 DeepSightScreener + FlameAggregator。
    返回 server 实例和 client 类（不在此创建 client 对象）。
    """
    server_conf = config.get("server", {})
    seed = kwargs.get("seed", config.get("seed", 42))

    screener = DeepSightScreener(**server_conf.get("screener", {})) \
        if server_conf.get("screener", {}) is not None else DeepSightScreener()
    aggregator = FlameAggregator(device=device, **server_conf.get("aggregator", {}))
    updater = BaseUpdater(config=server_conf.get("updater", {}))

    # 从 dataset_store 中提取 server 测试集
    server_test_task = kwargs.get("server_test_task")

    test_loader = None
    if server_test_task:
        test_ds_store = dataset_store[server_test_task.dataset_tag]
        server_dataset = Subset(test_ds_store, server_test_task.indices)
        batch_size = config.get("client", {}).get("batch_size", 64)
        test_loader = DataLoader(server_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=0)

    server = BaseServer(
        model=model,
        aggregator=aggregator,
        screener=screener,
        updater=updater,
        device=device,
        test_loader=test_loader,
        seed=seed,
    )

    return server, BaseClient

