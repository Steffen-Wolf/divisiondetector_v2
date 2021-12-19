from divisiondetector.models import Unet4D
import torch

unet = Unet4D(1, 1, 16, depth=2).cuda()
input_data = torch.rand(1, 1, 7, 16, 132, 132).cuda()

output = unet(input_data)
print(output)
print(output.shape)