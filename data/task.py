from dataclasses import dataclass

@dataclass
class Task:
    owner_id:str
    dataset_tag:str
    split:str
    indices:list[int]

class TaskSet:
    def __init__(self):
        self._tasks = {}
    def add_task(self, task:Task):
        id = task.owner_id
        split = task.split
        # 是否存在客户端
        if id not in self._tasks:
            self._tasks[id] = {}
        self._tasks[id][split] = task
    
    def get_task(self, id:str, split:str)->Task:
        #不存在时keyerror，符合预期
        return self._tasks[id][split]
    
    def __str__(self):
        return str(self._tasks)

if __name__ == "__main__":
    ts = TaskSet()
    ts.add_task(Task("client_1", "cifar10", "train", [1,2,3]))
    ts.add_task(Task("client_1", "cifar10", "test", [1,2,3]))
    print(ts)
    print(ts.get_task("client_1", "train"))