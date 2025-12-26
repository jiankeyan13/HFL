from typing import Tuple, Type
from torch.utils.data import Subset, DataLoader

from core.server.base_server import BaseServer
from core.client.base_client import BaseClient
from core.server.screener.hdbscan import HdbscanScreener
from core.server.aggregator.flame_aggregator import FlameAggregator
from core.server.updater.flame_noise_updater import FlameNoiseUpdater
from core.utils.registry import ALGORITHM_REGISTRY

# 客户端不进行梯度裁剪效果更好 
# https://wandb.ai/jiankeyan13-lab/FL-Test?nw=nwuserjiankeyan13xyz

@ALGORITHM_REGISTRY.register("flame")
def build_flame_algorithm(model, device, dataset_store, config, **kwargs) -> Tuple[BaseServer, Type[BaseClient]]:
    """
    构建 FLAME 算法：HDBSCAN 筛选 + 范数裁剪聚合 + 噪声添加。
    
    Returns:
        (server_instance, client_class)
    """
    server_conf = config.get('server', {})
    
    # 初始化 FLAME 三组件
    screener_conf = server_conf.get('screener', {})
    screener = HdbscanScreener(**screener_conf.get('params', {}))
    
    aggregator_conf = server_conf.get('aggregator', {})
    aggregator = FlameAggregator(**aggregator_conf.get('params', {}))
    
    updater_conf = server_conf.get('updater', {})
    updater = FlameNoiseUpdater(**updater_conf.get('params', {}))
    
    # 构建测试集 DataLoader
    server_test_task = kwargs.get('server_test_task')
    
    test_loader = None
    if server_test_task:
        test_ds_store = dataset_store[server_test_task.dataset_tag]
        server_dataset = Subset(test_ds_store, server_test_task.indices)
        batch_size = config.get('client', {}).get('batch_size', 64)
        test_loader = DataLoader(server_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    server = BaseServer(
        model=model,
        aggregator=aggregator,
        screener=screener,
        updater=updater,
        device=device,
        test_loader=test_loader
    )

    return server, BaseClient

