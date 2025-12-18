# Desafio de Programação: Classificação de Pistas de Pouso em Imagens de Satélite

Bem-vindo(a) ao desafio de programação! O objetivo é construir uma aplicação capaz de identificar e classificar pistas de pouso em uma imagem de satélite Sentinel-2, com foco na região do sudoeste paraense. Você tem a liberdade de escolher a abordagem tecnológica, seja utilizando o Google Earth Engine (GEE) para processamento em nuvem ou baixando a imagem para processamento local.

---

## O Desafio

Seu projeto deve abordar os seguintes pontos:

1.  **Acesso à Imagem Sentinel-2:** Obtenha uma imagem de satélite **Sentinel-2** cobrindo a região do **sudoeste paraense**. Você pode definir as coordenadas exatas ou usar um polígono que represente a área de interesse. A coleta dos dados pode ser feita através da API do Google Earth Engine (`ee`), da biblioteca `xee` (uma extensão do GEE para `xarray`) ou por meio de download direto.

2.  **Pré-processamento (Opcional, mas recomendado):** Se achar necessário, aplique técnicas de pré-processamento na imagem para melhorar a qualidade e facilitar a classificação. Isso pode incluir a remoção de nuvens, correção atmosférica (se não estiver pré-aplicada) ou a criação de índices de vegetação.

3.  **Classificação de Pistas de Pouso:** Desenvolva um algoritmo ou modelo para classificar as áreas que correspondem a pistas de pouso na imagem. A escolha da metodologia é sua desde que seja um segmentador semantico como as redes em forma de U ou até mesmo Transformers como o SwinTransformer.

4.  **Visualização e/ou Exportação:** Apresente os resultados da sua classificação de forma clara. Isso pode ser uma visualização interativa (no GEE ou em uma biblioteca como `folium`), ou a exportação dos resultados para um formato geoespacial padrão, como **GeoJSON** ou **GeoTIFF**, com as áreas classificadas.

---

## Requisitos Técnicos

Você pode usar as seguintes ferramentas, mas sinta-se à vontade para explorar outras:

* **Linguagem de Programação:** **Python** é altamente recomendado devido à vasta gama de bibliotecas geoespaciais e de aprendizado de máquina.
* **Acesso a Dados:**
    * **Google Earth Engine (GEE):** Para processamento em nuvem. Use a biblioteca `earthengine-api` ou `xee` para uma interface mais intuitiva.
* **Bibliotecas Sugeridas:**
    * `earthengine-api`
    * `xee` (para integrar o GEE com `xarray`)
    * `rasterio` e `fiona` (para manipulação de dados geoespaciais)
    * `scikit-learn` (para modelos de Machine Learning)
    * `matplotlib` ou `folium` (para visualização)
    * `geopandas` (para análise de dados vetoriais)

---

## Entrega do Projeto

Seu projeto deve ser entregue com os seguintes componentes:

* **Código-fonte:** Todo o código utilizado (scripts Python, notebooks Jupyter, etc.).
* **Instruções de Execução:** Um guia detalhado explicando como configurar o ambiente e executar seu código.
* **Resultados:** Uma amostra dos resultados gerados (imagens, mapas, arquivos). Se a visualização for interativa, inclua uma captura de tela ou um GIF.
* **Documentação:** Um texto breve (pode ser no próprio README) descrevendo a sua abordagem, a metodologia de classificação escolhida, os desafios encontrados e como você os superou.

---

## Critérios de Avaliação

O projeto será avaliado com base em:

* **Funcionalidade:** A aplicação executa as tarefas propostas de forma correta e completa.
* **Qualidade do Código:** Organização, clareza, modularidade e uso de boas práticas de programação.
* **Escolha da Abordagem:** A justificativa para as ferramentas e métodos utilizados é clara e bem fundamentada.
* **Documentação:** As instruções e explicações fornecidas são fáceis de entender.

---

## Como Começar

1. **Faça um fork deste repositório.**

2. **Clone o repositório** para sua máquina local.

3. **Crie um ambiente virtual** (recomendado) e instale as dependências necessárias.

4. **Configure o acesso ao Google Earth Engine** (se for utilizá-lo). Siga as instruções oficiais do GEE para autenticação.

5. **Comece a codificar!**

Boa sorte e divirta-se com o desafio!

---

## Guia de Execução do Projeto de Segmentação Semântica

### 1. Configuração do Ambiente

#### 1.1. Criar e Ativar o Ambiente Virtual

É altamente recomendável usar um ambiente virtual para isolar as dependências do projeto.

```sh
# Crie o ambiente virtual
python3 -m venv venv

# Ative o ambiente virtual
source venv/bin/activate
```

#### 1.2. Instalar Dependências Python

Com o ambiente virtual ativado, instale todas as bibliotecas Python necessárias usando o arquivo `requirements.txt`.

```sh
pip install -r requirements.txt
```

#### 1.3. Configurar o Google Earth Engine (GEE)

O script `preprocess_images.py` utiliza a API do Google Earth Engine. Você precisa autenticar sua conta:

```sh
earthengine authenticate
```

Siga as instruções no terminal para autenticar sua conta Google.

### 2. Fluxo de Execução (Passo a Passo)

Siga a ordem abaixo para preparar seus dados e treinar os modelos. **Certifique-se de que seu ambiente virtual esteja ativado (`source venv/bin/activate`) antes de executar qualquer script.**

#### 2.1. Pré-processamento de Imagens de Satélite (GEE)

Este script baixa imagens de satélite do Google Earth Engine em formato TIFF. Os diretórios de saída serão criados automaticamente.

```sh
python preprocess_images.py
```

*   **Entrada:** Coordenadas e parâmetros de busca definidos em `config.py` e `definitions.py`.
*   **Saída:** Arquivos `.tif` salvos em `data/images_tif/`.

**Observação.**
No projeto, as configurações são divididas em dois arquivos:
- `definitions.py`: Contém constantes e "regras de negócio" do projeto (ex: caminhos de diretórios, nomes de encoders, modos de loss, etc.). São valores que raramente mudam.
- `config.py`: Contém hiperparâmetros e configurações que podem ser ajustadas com frequência para experimentação (ex: learning rate, batch size, parâmetros de augmentation, etc.).

#### 2.2. Conversão de TIFF para PNG

Converte as imagens TIFF baixadas para o formato PNG, que é mais comum para treinamento de modelos. Os diretórios de saída serão criados automaticamente.

```sh
python convert_tif_to_png.py
```

*   **Entrada:** Imagens `.tif` de `data/images_tif/`.
*   **Saída:** Imagens `.png` salvas em `data/images_png/`.

#### 2.3. Anotação de Máscaras (LabelMe)

Neste ponto, você precisará usar uma ferramenta de anotação, como o **LabelMe**, para criar as máscaras de segmentação das imagens.

1.  **Use o LabelMe:** Abra o LabelMe e carregue as imagens PNG que você gerou (`data/images_png/`).
2.  **Crie Polígonos:** Desenhe os polígonos correspondentes às pistas de pouso em cada imagem.
3.  **Salve como JSON:** Salve as anotações no formato JSON no diretório `data/masks_json/`.

#### 2.4. Conversão de Máscaras JSON para PNG

Este script converte as anotações JSON geradas pelo LabelMe para máscaras PNG binárias. Os diretórios de saída serão criados automaticamente.

```sh
python convert_json_to_mask.py
```

*   **Entrada:** Arquivos `.json` de `data/masks_json/`.
*   **Saída:** Máscaras `.png` salvas em `data/masks_png/`.

#### 2.5. Geração de Overlays Visuais (Opcional)

Este script combina as imagens PNG com suas máscaras PNG para criar visualizações de overlay. Os diretórios de saída serão criados automaticamente.

```sh
python generate_overlay.py
```

*   **Entrada:** Imagens `.png` de `data/images_png/` e máscaras `.png` de `data/masks_png/`.
*   **Saída:** Imagens de overlay salvas em `data/overlay_png/`.

#### 2.6. Preparação dos Dados para Treinamento

Após os passos acima, você precisará **organizar manualmente** seus dados nos diretórios `data/train/images`, `data/train/masks`, `data/val/images`, `data/val/masks` e `data/test/images`.

*   Copie as imagens PNG para `data/train/images`, `data/val/images` e `data/test/images`.
*   Copie as máscaras PNG correspondentes para `data/train/masks` e `data/val/masks`.

#### 2.7. Treinamento do Modelo U-Net

Este script treina o modelo U-Net. Os diretórios de saída serão criados automaticamente.

```sh
python train_unet.py
```

*   **Entrada:** Imagens e máscaras de `data/train/` e `data/val/`.
*   **Saída:**
    *   Modelo treinado (`unet_best_model.pth`) salvo em `results/`.
    *   Gráfico da curva de treinamento (`unet_training_curve.png`) salvo em `results/`.

#### 2.8. Treinamento do Modelo U-Net++

Este script treina o modelo U-Net++. Os diretórios de saída serão criados automaticamente.

```sh
python train_unetplusplus.py
```

*   **Entrada:** Imagens e máscaras de `data/train/` e `data/val/`.
*   **Saída:**
    *   Modelo treinado (`unetplusplus_best_model.pth`) salvo em `results/`.
    *   Gráfico da curva de treinamento (`unetplusplus_training_curve.png`) salvo em `results/`.

#### 2.9. Inferência com o Modelo U-Net

Este script usa o modelo U-Net treinado para prever máscaras em imagens de teste e gerar overlays. Os diretórios de saída serão criados automaticamente.

```sh
python predict_unet.py
```

*   **Entrada:** Imagens de teste de `data/test/images/` e o modelo `unet_best_model.pth` de `results/`.
*   **Saída:**
    *   Máscaras previstas (`.png`) salvas em `results/predicted_masks_unet/`.
    *   Overlays (`.png`) salvos em `results/predicted_overlays_unet/`.

#### 2.10. Inferência com o Modelo U-Net++

Este script usa o modelo U-Net++ treinado para prever máscaras em imagens de teste e gerar overlays. Os diretórios de saída serão criados automaticamente.

```sh
python predict_unetplusplus.py
```

*   **Entrada:** Imagens de teste de `data/test/images/` e o modelo `unetplusplus_best_model.pth` de `results/`.
*   **Saída:**
    *   Máscaras previstas (`.png`) salvas em `results/predicted_masks_unetplusplus/`.
    *   Overlays (`.png`) salvos em `results/predicted_overlays_unetplusplus/`.
