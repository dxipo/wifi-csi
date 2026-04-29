# SDP To Image-Like Translator

This folder contains a lightweight PyTorch model that translates the offline SDP
feature produced by preprocessing into an RGB-sized image-like tensor.

Default shape:

```text
input : (B, 30, 270, 3) or (B, 3, 30, 270)
output: (B, 3, 360, 640)
```

The design follows the same high-level idea as DensePose From WiFi's modality
translation module: encode CSI-domain information into a compact latent map and
decode it into an image-domain representation. Because the current SDP tensor is
already image-like, this model uses a small convolutional encoder instead of a
large fully connected encoder over all input elements.

Minimal usage:

```python
import torch
from dataset_model import SDPToImageLikeTranslator

model = SDPToImageLikeTranslator(output_activation="none")
sdp = torch.randn(2, 30, 270, 3)
image_like = model(sdp)
print(image_like.shape)  # torch.Size([2, 3, 360, 640])
```

Use `output_activation="sigmoid"` if training directly against RGB images scaled
to `[0, 1]`, or keep `"none"` if the tensor is used as a learned feature map for
another model.
