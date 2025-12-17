# ======================================================================================================================
# PREPROCESS_IMAGES.PY
# ======================================================================================================================
# Nome do projeto no Google Earth Engine.
GEE_PROJECT = 'gee-airstrip'
# Raio da área de interesse (ROI) em metros ao redor de cada ponto.
BUFFER_METERS = 2000
# Lista de imagens por coordenadas (longitude, latitude) para download.
COORDINATES = [
    (-49.923611, -0.999722),
    (-48.143889, -4.799444),
    (-49.976667, -7.090278),
    (-47.526667, -2.042222),
    (-51.179167, -6.744167),
    (-53.775000, -6.088889),
    (-55.837500, -0.326667),
    (-51.192778, -6.762222),
    (-55.165556, -0.220833)
]

# ======================================================================================================================
# CONVERT_TIF_TO_PNG.PY
# ======================================================================================================================

# ======================================================================================================================
# CONVERT_JSON_TO_MASK.PY
# ======================================================================================================================

# ======================================================================================================================
# GENERATE_OVERLAY.PY
# ======================================================================================================================
# Intensidade da máscara de overlay (default 0.5).
ALPHA = 0.5

# ======================================================================================================================
# TRAIN_UNET.PY
# ======================================================================================================================
# Dispositivo de treinamento ('cuda' ou 'cpu').
DEVICE = "cuda"
# Número de épocas de treinamento.
EPOCHS = 2000
# Tamanho do lote (batch size).
BATCH_SIZE = 6
# Taxa de aprendizado (learning rate).
LEARNING_RATE = 1e-4
# Decaimento de peso para o otimizador Adam.
WEIGHT_DECAY = 1e-4
# Paciência para o early stopping.
EARLY_STOPPING_PATIENCE = 100
# Dimensões para redimensionar as imagens de entrada.
IMAGE_HEIGHT = 400
IMAGE_WIDTH = 400
# ======================================================================================================================
# Data Augmentation:
# Probabilidade de aplicar HorizontalFlip.
P_HORIZONTAL_FLIP = 0.5
# Probabilidade de aplicar VerticalFlip.
P_VERTICAL_FLIP = 0.5
# Limite de rotação em graus.
ROTATE_LIMIT = 45
# Probabilidade de aplicar rotação.
P_ROTATE = 0.8
# Limite de brilho para RandomBrightnessContrast.
BRIGHTNESS_LIMIT = 0.35
# Limite de contraste para RandomBrightnessContrast.
CONTRAST_LIMIT = 0.35
# Probabilidade de aplicar RandomBrightnessContrast.
P_BRIGHTNESS_CONTRAST = 0.8
# Probabilidade de aplicar GaussNoise.
P_GAUSS_NOISE = 0.3
# Probabilidade de aplicar ElasticTransform.
P_ELASTIC_TRANSFORM = 0.3
# Alpha para ElasticTransform.
ELASTIC_ALPHA = 120
# Sigma para ElasticTransform.
ELASTIC_SIGMA = 120 * 0.07
# Probabilidade de aplicar GridDistortion.
P_GRID_DISTORTION = 0.3


# ======================================================================================================================
# TRAIN_UNETPLUSPLUS.PY
# ======================================================================================================================
# Número de épocas de treinamento.
PLUSPLUS_EPOCHS = 1500
# Tamanho do lote (batch size).
PLUSPLUS_BATCH_SIZE = 4
# Taxa de aprendizado (learning rate).
PLUSPLUS_LEARNING_RATE = 1e-4
# Decaimento de peso para o otimizador Adam.
PLUSPLUS_WEIGHT_DECAY = 1e-5
# Paciência para o early stopping.
PLUSPLUS_EARLY_STOPPING_PATIENCE = 100
# Dimensões para redimensionar as imagens de entrada (divisível por 32).
PLUSPLUS_IMAGE_HEIGHT = 416
PLUSPLUS_IMAGE_WIDTH = 416
# Peso para a Focal Loss na função de perda combinada.
PLUSPLUS_FOCAL_LOSS_WEIGHT = 0.75
# Peso para a Dice Loss na função de perda combinada.
PLUSPLUS_DICE_LOSS_WEIGHT = 0.25
# Fator de redução para o scheduler de learning rate.
PLUSPLUS_SCHEDULER_FACTOR = 0.2
# Paciência para o scheduler de learning rate.
PLUSPLUS_SCHEDULER_PATIENCE = 25
# ======================================================================================================================
# Data Augmentation:
# Probabilidade de aplicar HorizontalFlip.
PLUSPLUS_P_HORIZONTAL_FLIP = 0.5
# Probabilidade de aplicar VerticalFlip.
PLUSPLUS_P_VERTICAL_FLIP = 0.5
# Limite de rotação em graus.
PLUSPLUS_ROTATE_LIMIT = 20
# Probabilidade de aplicar rotação.
PLUSPLUS_P_ROTATE = 0.7
# Probabilidade de aplicar RandomBrightnessContrast.
PLUSPLUS_P_BRIGHTNESS_CONTRAST = 0.7
# Probabilidade de aplicar GaussNoise.
PLUSPLUS_P_GAUSS_NOISE = 0.2
