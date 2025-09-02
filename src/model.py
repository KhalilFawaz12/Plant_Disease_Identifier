import torch
from torchvision import models

def create_efficientnetb1_model(num_classes: int = 15, fine_tune: bool = False):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    weights = models.EfficientNet_B1_Weights.DEFAULT
    auto_transforms = weights.transforms()
    model = models.efficientnet_b1(weights=weights)

    if not fine_tune:
        for param in model.parameters():
            param.requires_grad = False

    in_Linear_features = model.classifier[1].in_features           # model.classifier[1].in_features is the number of inputs to the final linear layer
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model.classifier=torch.nn.Sequential(
            torch.nn.Dropout(p=0.3,inplace=False),
            torch.nn.Linear(in_features=in_Linear_features,out_features=num_classes,bias=True)
        )
    return model, auto_transforms
