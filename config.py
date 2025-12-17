# ======================================================================================================================
# PREPROCESS_IMAGES.PY
# ======================================================================================================================
# Nome do projeto no Google Earth Engine.
GEE_PROJECT = 'gee-airstrip'
# Coleção de imagens a ser utilizada (default "COPERNICUS/S2_SR_HARMONIZED").
IMAGE_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
# Nome da propriedade que indica a porcentagem de nuvens.
CLOUD_PROPERTY = "CLOUDY_PIXEL_PERCENTAGE"
# Bandas a serem selecionadas para a imagem de saída (ex: ["B4", "B3", "B2"] para RGB).
BANDS = ["B4", "B3", "B2"]
# Sistema de Coordenadas de Referência.
CRS = "EPSG:4326"
# Raio da área de interesse (ROI) em metros ao redor de cada ponto.
BUFFER_METERS = 2000
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
# Diretório de saída para as imagens PNG convertidas.
PNG_DIRECTORY = "data/images_png"
