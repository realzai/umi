from umi import Umi
from umi.dataset import TrainData
from umi.config import base_config

umi = Umi(config=base_config)

dataset = TrainData(config=base_config)

umi.train(dataset=dataset)

# umi.evaluate(dataset=dataset)
