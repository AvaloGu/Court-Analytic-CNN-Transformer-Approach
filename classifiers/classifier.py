import torch.nn as nn
from torch.nn import functional as F
import inspect
import torch
from abc import ABC, abstractmethod
from loaders.vocab import CLASSCOUNT
from models.transformer import GPT
from models.convnext import ConvNeXt


class ShotPredictor(nn.Module):
    def __init__(self, config_conv, config_transformer):
        super().__init__()
        self.convnet = ConvNeXt(config_conv)
        self.gpt = GPT(config_transformer)

    def forward(self, x, y=None, h=None):
        # x: (N, 3, 224, 224)
        # y: (T,)
        x = self.convnet(x)  # (N, 768)
        logit = self.gpt(x, y[:-1])  # (T-1, vocab_size)
        loss = None
        if y is not None:
            loss = F.cross_entropy(logit, y[1:], label_smoothing=0.1)
        return logit, loss  # (N, vocab_size), loss

    def configure_optimizers(
        self,
        weight_decay=0.2,
        learning_rate=4e-3,
        device_type="cpu",
        master_process=True,
    ):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            fused=use_fused,
        )
        return optimizer
