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
    * **Google Earth Engine (GEE)::** Para processamento em nuvem. Use a biblioteca `earthengine-api` ou `xee` para uma interface mais intuitiva.
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

## Guia de Execução do Projeto: Segmentação Semântica de Pistas de Pouso

Este guia detalha o processo de configuração, aquisição de dados, pré-processamento, treinamento e inferência para o projeto.

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

**Observação sobre Configurações:**
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

---

## Documentação do Projeto: Abordagem e Metodologia

### 1. Abordagem Geral

O projeto adota uma abordagem modular e baseada em scripts para o problema de segmentação semântica de pistas de pouso em imagens de satélite Sentinel-2. O fluxo de trabalho é dividido em etapas claras:

#### 1.1. Aquisição de Dados
Utilização do Google Earth Engine (GEE) para baixar imagens de satélite Sentinel-2. As coordenadas de 9 pistas de pouso foram obtidas através de buscas no Google Earth. Durante este processo, as imagens são filtradas para remover nuvens (com um limiar configurável) e são baixadas com a maior resolução espacial disponível gratuitamente (10x10 metros por pixel). O conjunto de dados foi dividido em: 6 imagens para treino, 2 para validação e 1 para teste (Vila Canopus - Altamira, Região Sudoeste do Pará).
#### 1.2. Pré-processamento
Masking e conversão de formatos de imagem (TIFF para PNG) e de anotações (JSON para PNG) para preparar os dados para o treinamento.
#### 1.3. Treinamento de Modelos
Implementação e treinamento de duas arquiteturas de redes neurais convolucionais (U-Net e U-Net++) para segmentação semântica.
#### 1.4. Inferência e Visualização
Geração de máscaras de predição e overlays visuais para avaliar o desempenho dos modelos em novas imagens.

### 2. Metodologia de Classificação

Duas arquiteturas de redes neurais foram ostensivamente exploradas:

#### 2.1. U-Net (Baseline)

*   **Arquitetura:** U-Net clássica.
*   **Encoder:** `resnet34` pré-treinado no ImageNet. O ResNet34 é um encoder robusto e de bom desempenho, servindo como uma base sólida.
*   **Função de Perda:** Combinação de `Binary Cross-Entropy (BCE)` e `Dice Loss`. A BCE é comum para classificação binária de pixels, enquanto a Dice Loss é eficaz para lidar com o desbalanceamento de classes e otimizar a métrica IoU (Intersection over Union).
*   **Otimizador:** Adam.
*   **Scheduler de Learning Rate:** Fixo.

#### 2.2. U-Net++ (Estratégia Avançada)

*   **Arquitetura:** U-Net++.
*   **Encoder:** `efficientnet-b7` pré-treinado no ImageNet. O EfficientNet-B7 é um encoder mais moderno e eficiente que o ResNet34, oferecendo um melhor equilíbrio entre precisão e custo computacional.
*   **Mecanismo de Atenção:** Utiliza `scse` (Spatial and Channel Squeeze & Excitation) no decoder, permitindo que o modelo foque nas características mais relevantes espacialmente e entre os canais.
*   **Função de Perda:** Combinação ponderada de `Focal Loss` (0.75) e `Dice Loss` (0.25). A Focal Loss é particularmente útil para problemas com grande desbalanceamento de classes, dando mais peso aos exemplos difíceis de classificar.
*   **Otimizador:** Adam com `weight_decay` ajustado para `5e-5` para regularização.
*   **Scheduler de Learning Rate:** `CosineAnnealingLR`. Este scheduler varia a taxa de aprendizado de forma cíclica, permitindo que o modelo explore o espaço de parâmetros de forma mais eficaz e se estabeleça em mínimos mais profundos e estáveis.

### 3. Desafios Encontrados e Soluções

#### 3.1. Quantidade Limitada de Dados
*   **Desafio:** O treinamento foi realizado com um dataset pequeno, contendo apenas 6 imagens para treino e 2 para validação. Isso aumenta significativamente o risco de overfitting e dificulta a generalização do modelo.
*   **Solução:** Para mitigar este problema, uma estratégia agressiva de **Data Augmentation** foi implementada. Transformações como rotações, flips, variações de brilho/contraste, e distorções elásticas (`ElasticTransform`, `GridDistortion`) foram aplicadas em tempo real para criar novas variantes das imagens de treino a cada época, expandindo artificialmente o dataset e forçando o modelo a aprender características mais robustas e invariantes.

#### 3.2. Gerenciamento de Configurações
*   **Desafio:** Manter múltiplos parâmetros (caminhos, hiperparâmetros, configurações de modelo, regras de negócio, etc.) organizados e fáceis de modificar.
*   **Solução:** Implementação de um sistema de configuração modular com `config.py` (para hiperparâmetros de experimentação) e `definitions.py` (para constantes e regras de negócio). Isso centraliza as configurações e melhora a legibilidade e a manutenibilidade do código.

#### 3.3. Convergência e Estabilidade do Treinamento
*   **Desafio:** O modelo U-Net++ inicial apresentava estagnação da perda de validação em um platô, mesmo com um scheduler `ReduceLROnPlateau`.
*   **Solução:**
    *   **Substituição do Scheduler:** Troca do `ReduceLROnPlateau` por `CosineAnnealingLR`, que promove uma descida mais suave e consistente da taxa de aprendizado, ajudando o modelo a explorar melhor o espaço de perda e evitar mínimos locais rasos.
    *   **Ajuste de Regularização:** Aumento do `weight_decay` para `5e-5` para incentivar o modelo a encontrar soluções mais generalizáveis e prevenir o overfitting.
    *   **Otimização da Função de Perda:** Reversão dos pesos da `Focal Loss` e `Dice Loss` para `0.75` e `0.25`, respectivamente, após a constatação de que um peso 50/50 piorava o desempenho, indicando a importância da Focal Loss para o problema de desbalanceamento de classes.

#### 3.4. Otimização de Hiperparâmetros
*   **Desafio:** Encontrar a combinação ideal de parâmetros para maximizar a performance do modelo, como a taxa de aprendizado, o tamanho do batch, a intensidade do data augmentation e o critério de parada (`early stopping`).
*   **Solução:** Realização de um processo iterativo de experimentação. Cada alteração nos hiperparâmetros (`config.py`) foi seguida por um novo ciclo de treinamento e avaliação, permitindo a observação do impacto de cada ajuste na curva de aprendizado e na perda de validação final.
