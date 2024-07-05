import segmentation_models_pytorch as smp
from aerial_image_segmentation.entity.config_entity import ExternalModelConfig

model_evaluation_config = ExternalModelConfig()

model = getattr(smp, model_evaluation_config.model_name)(
        model_evaluation_config.encoder_name,
        encoder_weights=model_evaluation_config.encoder_weights,
        classes=model_evaluation_config.classes,
        activation=model_evaluation_config.activation,
        encoder_depth=model_evaluation_config.encoder_depth,
        decoder_channels=model_evaluation_config.decoder_channels)

model = model.to(model_evaluation_config.device)
