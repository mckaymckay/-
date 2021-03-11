# 此为U-net中的双线性插值

import torch
import numpy as np

src = torch.Tensor(np.asarray([[[[10, 20], [30, 40]]]]))
print('src:')
print(src)

upsample = torch.nn.Upsample(size=None,
                             scale_factor=2,
                             mode='bilinear',
                             align_corners=False)
print('upsample:')
print(upsample(src))