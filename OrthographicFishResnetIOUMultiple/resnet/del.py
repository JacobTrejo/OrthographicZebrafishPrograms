import torch.nn as nn
import torch


on = torch.ones(18, 2, 24)
on2 = torch.ones(18, 2, 24) * 3

c = nn.MSELoss(reduction='none'  )
#c = nn.MSE()
#print(c(on, on2))
print(c(on,on2).shape)

