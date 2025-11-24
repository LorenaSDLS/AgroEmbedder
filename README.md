# MultiModalAE – Autoencoder Multimodal para Municipios Agrícolas

## Descripción

**MultiModalAE** es un autoencoder multimodal diseñado para generar **embeddings de municipios mexicanos**, integrando información de:

* Clima (temperatura, precipitación, etc.)
* Edafología (suelo: MO, arcilla, arena, limos)
* Radiación

El objetivo es capturar similitudes entre municipios y permitir análisis de similitud usando distancias coseno.

El modelo puede manejar datos faltantes, aplicando máscaras para ignorar entradas ausentes durante el entrenamiento.





















