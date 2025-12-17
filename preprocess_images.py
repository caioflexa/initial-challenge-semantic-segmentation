"""
Este script busca e pré-processa imagens de satélite do Google Earth Engine.
Ele exporta as imagens como arquivos GeoTIFF com base nas configurações definidas.
"""
import ee
import geemap
import os

import definitions
import config


def main():
    """
    Função principal para processar e exportar imagens de satélite.
    """
    # Cria o diretório de saída, se ele não existir.
    os.makedirs(definitions.TIF_DIRECTORY, exist_ok=True)

    # Inicializa a API do Google Earth Engine.
    ee.Initialize(project=config.GEE_PROJECT)

    # Itera sobre cada coordenada definida no arquivo de configuração.
    for idx, (lon, lat) in enumerate(config.COORDINATES):
        print(f"\nProcessando ponto {idx + 1} de {len(config.COORDINATES)}...")

        # Define a Região de Interesse (ROI) como um buffer ao redor do ponto.
        roi = ee.Geometry.Point(lon, lat).buffer(config.BUFFER_METERS)

        # Busca, filtra e ordena a coleção de imagens para obter a melhor imagem.
        collection = (
            ee.ImageCollection(definitions.IMAGE_COLLECTION)
            .filterBounds(roi)
            .filterDate(definitions.START_DATE, definitions.END_DATE)
            .filter(ee.Filter.lt(definitions.CLOUD_PROPERTY, definitions.CLOUD_THRESHOLD))
        )
        image = collection.sort(definitions.CLOUD_PROPERTY).first()

        # Seleciona as bandas de interesse da imagem.
        image_selection = image.select(definitions.BANDS)

        # Define o nome do arquivo e o caminho completo de saída.
        filename = definitions.OUTPUT_FILENAME_FORMAT.format(idx=idx)
        tif_path = os.path.join(definitions.TIF_DIRECTORY, filename)

        # Exporta a imagem processada como um arquivo GeoTIFF.
        geemap.ee_export_image(
            image_selection,
            filename=tif_path,
            region=roi,
            scale=definitions.SCALE,
            crs=definitions.CRS,
            file_per_band=definitions.FILE_PER_BAND
        )
        print(f"GeoTIFF salvo: {tif_path}")

    print("\nTodas as imagens foram processadas com sucesso.")


if __name__ == "__main__":
    main()
