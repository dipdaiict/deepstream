# aerial_image_segmentation/models/model.py
import torch
import segmentation_models_pytorch as smp
from aerial_image_segmentation.entity.config_entity import ExternalModelConfig

def get_model(config: ExternalModelConfig):
    model = getattr(smp, config.model_name)(
        config.encoder_name,
        encoder_weights=config.encoder_weights,
        classes=config.classes,
        activation=config.activation,
        encoder_depth=config.encoder_depth,
        decoder_channels=config.decoder_channels
    )
    device = torch.device(config.device)
    model = model.to(device)
    return model