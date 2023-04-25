from src.model import CAE
from src.config import config_CAE as config
from src.trainer import Trainer

import torch
import torch.nn as nn
from torch.optim import Adam

model = CAE(config)

criterion = nn.MSELoss()

optimizer = Adam(model.parameters())

trainer = Trainer(model, criterion, optimizer, config)

trainer.train()

with open("CAE_re.pt", "wb") as f:
    torch.save(trainer.model.state_dict(), f)

trainer.plot()
