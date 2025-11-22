import numpy as np
from abc import ABC, abstractmethod
from task import Task, TaskSet
from dataset_store import DatasetStore
from typing import List, Dict

class Partitioner(ABC):
    """
    基类，用于将数据集进行划分
    """
    @abstractmethod
    def partition(self, store: DatasetStore, num_clients: int, split: str="train")->TaskSet:
        pass

class IIDPartitioner(Partitioner):
    def __init__(self, seed: int=42):
        self.seed = seed
    def partition(self, store: DatasetStore, num_clients: int, split: str="train")->TaskSet:
        #创建Task集合
        taskset = TaskSet()

        #获取索引列表，打乱
        n = len(store)
        indices = np.arange(n)
        rng = np.random.default_rng(self.seed)
        rng.shuffle(indices)
        
        #获取索引,封装成Task，添加到TaskSet中
        splits = np.array_split(indices, num_clients)
        for i,  client_indice in enumerate(splits):
            task = Task(
                owner_id=f"client_{i}",
                dataset_tag=store.name,
                split=split,
                indices=client_indice.tolist() # 转成纯 Python list 方便序列化
            )
            taskset.add_task(task)
        return taskset

if __name__ == '__main__':
    class TestData:
        def __init__(self):
            self.name = "test"
        def __len__(self):
            return 100
    testdata = TestData()
    iid_partitioner = IIDPartitioner()
    taskset = iid_partitioner.partition(testdata, 3)

    t0 = taskset.get_task("client_0", "train")
    print(f"Client_0样本数:{len(t0.indices)}")
    print(f"Client_0样本前5索引:{t0.indices[:5]}")



    