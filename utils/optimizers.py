import torch.optim as optim
from pytorch_ranger import Ranger


OPTIMIZERS = {
    "adamw": optim.AdamW,
    "adam": optim.Adam,
    "rmsprop": optim.RMSprop,
    "sgd": optim.SGD,
    "ranger": Ranger,
}


def get_optimizer(optimizer_name, model, **kwargs):
    optimizer_name = optimizer_name.lower()

    if optimizer_name not in OPTIMIZERS:
        raise ValueError(
            f"Unknown optimizer: {optimizer_name}. "
            f"Choose from {list(OPTIMIZERS.keys())}"
        )

    optimizer_class = OPTIMIZERS[optimizer_name]

    return optimizer_class(
        (param for param in model.parameters() if param.requires_grad),
        **kwargs,
    )