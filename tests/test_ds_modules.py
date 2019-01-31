from dst.modules import *
from dst.utils import *

# m = DSConvTranspose2d(
#     in_channels=3,
#     out_channels=16,
#     kernel_size=3,
#     stride=2,
#     padding=1
# )

m = SpatialMask(sparse_mask_2d(16), shuffle=True)
x = torch.rand(1, 1, 16, 16)

print(m(x))
print(m(x))

# TODO: Tests for DSLinear

# TODO: Tests for DSConv2d

# TODO: Tests for DSConvTranspose2d
