# ======================================================================================================================
# PREPROCESS_IMAGES.PY
# ======================================================================================================================
# Coleção de imagens a ser utilizada (default "COPERNICUS/S2_SR_HARMONIZED").
IMAGE_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
# Nome da propriedade que indica a porcentagem de nuvens.
CLOUD_PROPERTY = "CLOUDY_PIXEL_PERCENTAGE"
# Bandas a serem selecionadas para a imagem de saída (ex: ["B4", "B3", "B2"] para RGB).
BANDS = ["B4", "B3", "B2"]
# Sistema de Coordenadas de Referência.
CRS = "EPSG:4326"
# Período de busca das imagens.
START_DATE = "2023-01-01"
END_DATE = "2023-12-31"
# Percentual máximo de cobertura de nuvens permitido na imagem (default 10).
CLOUD_THRESHOLD = 10
# Resolução espacial da imagem exportada em metros/pixel (default 10).
SCALE = 10
# Diretório de saída para as imagens processadas.
TIF_DIRECTORY = "data/images_tif"
# Formato do nome do arquivo de saída. Use {idx} para o índice do ponto.
OUTPUT_FILENAME_FORMAT = "sentinel2_airstrip_{idx}.tif"
# Se True, exporta cada banda da imagem em um arquivo separado.
# Se False, exporta um único GeoTIFF com todas as bandas (default).
FILE_PER_BAND = False


# ======================================================================================================================
# CONVERT_TIF_TO_PNG.PY
# ======================================================================================================================
# Diretório de saída para as imagens PNG convertidas.
PNG_DIRECTORY = "data/images_png"


# ======================================================================================================================
# CONVERT_JSON_TO_MASK.PY
# ======================================================================================================================
# Diretório de entrada contendo os arquivos .json do LabelMe.
JSON_MASKS_DIRECTORY = "data/masks_json"
# Diretório de saída para as máscaras de segmentação em formato .png.
PNG_MASKS_DIRECTORY = "data/masks_png"


# ======================================================================================================================
# GENERATE_OVERLAY.PY
# ======================================================================================================================
# Diretório de saída para as imagens em overlay (imagem + máscara).
OVERLAY_DIRECTORY = "data/overlay_png"


# ======================================================================================================================
# TRAIN_UNET.PY
# ======================================================================================================================
# Diretórios de dados para treinamento e validação.
TRAIN_IMAGES_DIR = "data/train/images"
TRAIN_MASKS_DIR = "data/train/masks"
VAL_IMAGES_DIR = "data/val/images"
VAL_MASKS_DIR = "data/val/masks"
# Caminho para salvar o melhor modelo treinado.
MODEL_PATH = "results/unet_best_model.pth"
# Arquitetura do encoder da U-Net.
ENCODER_NAME = "resnet34"
# Pesos pré-treinados para o encoder.
ENCODER_WEIGHTS = "imagenet"
# Número de canais de entrada da imagem (3 para RGB).
IN_CHANNELS = 3
# Número de classes de saída (1 para segmentação binária).
NUM_CLASSES = 1
# Limiar para binarizar a máscara (0-255).
MASK_THRESHOLD = 127
# Número de workers para o DataLoader.
NUM_WORKERS = 4
# Modo da Dice Loss.
DICE_LOSS_MODE = 'binary'
# Média para a normalização das imagens.
NORM_MEAN = [0.0, 0.0, 0.0]
# Desvio padrão para a normalização das imagens.
NORM_STD = [1.0, 1.0, 1.0]


# ======================================================================================================================
# TRAIN_UNETPLUSPLUS.PY
# ======================================================================================================================
# Caminho para salvar o melhor modelo treinado.
PLUSPLUS_MODEL_PATH = "results/unetplusplus_best_model.pth"
# Arquitetura do encoder da U-Net++.
PLUSPLUS_ENCODER_NAME = "efficientnet-b7"
# Tipo de atenção para o decoder.
PLUSPLUS_DECODER_ATTENTION = 'scse'
# Modo da Focal Loss.
PLUSPLUS_FOCAL_LOSS_MODE = 'binary'


# ======================================================================================================================
# PLOTTING
# ======================================================================================================================
# Título para o gráfico de convergência.
PLOT_TITLE = 'Curva de Convergência do Treinamento'
# Rótulo do eixo X.
PLOT_X_LABEL = 'Epochs'
# Rótulo do eixo Y.
PLOT_Y_LABEL = 'Loss'
# Rótulo para a linha de loss de treinamento.
PLOT_TRAIN_LABEL = 'Training Loss'
# Rótulo para a linha de loss de validação.
PLOT_VAL_LABEL = 'Validation Loss'
# Formato do texto para a linha de melhor loss.
PLOT_BEST_LOSS_LABEL = 'Best Val Loss: {best_loss_value:.4f} (Epoch {best_loss_epoch})'


# ======================================================================================================================
# INFERENCE
# ======================================================================================================================
# Diretório de imagens de teste para inferência.
TEST_IMAGES_DIR = "data/test/images"
# Diretório de saída para as máscaras previstas pelo modelo U-Net.
PREDICTED_MASKS_UNET_DIR = "results/predicted_masks_unet"
# Diretório de saída para as máscaras previstas pelo modelo U-Net++.
PREDICTED_MASKS_UNETPLUSPLUS_DIR = "results/predicted_masks_unetplusplus"
# Diretório de saída para os overlays das predições do U-Net.
PREDICTED_OVERLAYS_UNET_DIR = "results/predicted_overlays_unet"
# Diretório de saída para os overlays das predições do U-Net++.
PREDICTED_OVERLAYS_UNETPLUSPLUS_DIR = "results/predicted_overlays_unetplusplus"
