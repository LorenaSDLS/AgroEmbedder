import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from data import *


# ----------------------------------------------------------
# 1. Cargar datos en bruto
# ----------------------------------------------------------

def load_raw_data():
    df_clima = pd.read_parquet("data/climatologia_mensual_wide.parquet")
    df_suelo = pd.read_parquet("data/suelo_listo.parquet")
    df_rad   = pd.read_parquet("data/municipios_GHI_wide.parquet")
    return df_clima, df_suelo, df_rad


# ----------------------------------------------------------
# 2. Indexar por CVEGEO
# ----------------------------------------------------------

def index_by_cvegeo(df_clima, df_suelo, df_rad):
    df_clima = df_clima.set_index("CVEGEO")
    df_suelo = df_suelo.set_index("CVEGEO")
    df_rad   = df_rad.set_index("CVEGEO")
    return df_clima, df_suelo, df_rad


# ----------------------------------------------------------
# 3. Seleccionar columnas numéricas y categóricas de suelo
# ----------------------------------------------------------

def get_soil_column_groups(df_suelo):
    categorical_cols = [
        "CLAVE_WRB", "CALIF_PRIM", "CALIF_SEC", "F_RúDICA", "NOMEN_HTE",
        "HORIZONTE", "PROP_MAT", "COL_CAMPO", "CLAS_TEXT",
        "COL_SECO_L", "COL_HUM_L"
    ]
    categorical_cols = [c for c in categorical_cols if c in df_suelo.columns]

    numeric_cols = [c for c in df_suelo.columns if c not in categorical_cols]
    return numeric_cols, categorical_cols


# ----------------------------------------------------------
# 4. Rellenar NaNs y crear máscaras
# ----------------------------------------------------------

def fill_and_mask(df_clima, df_rad, df_suelo, numeric_cols, categorical_cols):
    
    # Clima
    mask_clima = (~df_clima.isna()).astype(float)
    df_clima_filled = df_clima.fillna(0)

    # Radiación
    mask_rad = (~df_rad.isna()).astype(float)
    df_rad_filled = df_rad.fillna(0)

    # Suelo numérico
    mask_soil_num  = (~df_suelo[numeric_cols].isna()).astype(float)
    df_suelo_num_filled = df_suelo[numeric_cols].fillna(0)

    # Suelo categórico
    df_suelo_cat_filled = df_suelo[categorical_cols].fillna("missing")

    return (
        df_clima_filled, df_rad_filled, df_suelo_num_filled, df_suelo_cat_filled,
        mask_clima, mask_rad, mask_soil_num
    )


# ----------------------------------------------------------
# 5. One-Hot Encoding para las columnas categóricas
# ----------------------------------------------------------

def encode_categorical(df_suelo_cat_filled):
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    encoded = ohe.fit_transform(df_suelo_cat_filled)
    encoded_df = pd.DataFrame(
        encoded,
        index=df_suelo_cat_filled.index,
        columns=ohe.get_feature_names_out(df_suelo_cat_filled.columns)
    )

    mask_cat = np.ones_like(encoded)  # OHE nunca tiene NaNs

    return encoded_df, mask_cat


# ----------------------------------------------------------
# 6. Normalizar las variables numéricas
# ----------------------------------------------------------

def normalize_all(df_clima_filled, df_rad_filled, df_suelo_num_filled):
    scaler_clima = StandardScaler()
    scaler_rad   = StandardScaler()
    scaler_suelo = StandardScaler()

    df_clima_scaled = pd.DataFrame(
        scaler_clima.fit_transform(df_clima_filled),
        index=df_clima_filled.index,
        columns=df_clima_filled.columns
    )

    df_rad_scaled = pd.DataFrame(
        scaler_rad.fit_transform(df_rad_filled),
        index=df_rad_filled.index,
        columns=df_rad_filled.columns
    )

    df_suelo_num_scaled = pd.DataFrame(
        scaler_suelo.fit_transform(df_suelo_num_filled),
        index=df_suelo_num_filled.index,
        columns=df_suelo_num_filled.columns
    )

    return df_clima_scaled, df_rad_scaled, df_suelo_num_scaled


# ----------------------------------------------------------
# 7. Fusionar niveles numéricos y categóricos del suelo
# ----------------------------------------------------------

def merge_soil(df_suelo_num_scaled, df_suelo_cat_encoded, mask_soil_num, mask_cat):
    df_suelo_all = pd.concat([df_suelo_num_scaled, df_suelo_cat_encoded], axis=1)
    mask_soil_all = np.concatenate([mask_soil_num.values, mask_cat], axis=1)
    return df_suelo_all, mask_soil_all


# ----------------------------------------------------------
# 8. Crear vectores alineados por municipio
# ----------------------------------------------------------

def create_vectors(df_clima_scaled, df_suelo_all, df_rad_scaled,
                   mask_clima, mask_rad, mask_soil_all):

    municipios = sorted(
        set(df_clima_scaled.index)
        | set(df_suelo_all.index)
        | set(df_rad_scaled.index)
    )

    Xc, Xe, Xr = [], [], []
    Mc, Me, Mr = [], [], []

    clima_dim = df_clima_scaled.shape[1]
    suelo_dim = df_suelo_all.shape[1]
    rad_dim   = df_rad_scaled.shape[1]

    for muni in municipios:

        # ---- CLIMA ----
        if muni in df_clima_scaled.index:
            Xc.append(df_clima_scaled.loc[muni].values)
            Mc.append(mask_clima.loc[muni].values)
        else:
            Xc.append(np.zeros(clima_dim))
            Mc.append(np.zeros(clima_dim))

        # ---- SUELO ----
        if muni in df_suelo_all.index:
            Xe.append(df_suelo_all.loc[muni].values)
            Me.append(mask_soil_all[df_suelo_all.index.get_loc(muni)])
        else:
            Xe.append(np.zeros(suelo_dim))
            Me.append(np.zeros(suelo_dim))

        # ---- RADIACIÓN ----
        if muni in df_rad_scaled.index:
            Xr.append(df_rad_scaled.loc[muni].values)
            Mr.append(mask_rad.loc[muni].values)
        else:
            Xr.append(np.zeros(rad_dim))
            Mr.append(np.zeros(rad_dim))

    return (
        np.array(Xc),
        np.array(Xe),
        np.array(Xr),
        np.array(Mc),
        np.array(Me),
        np.array(Mr),
        municipios
    )


# ----------------------------------------------------------
# 9. FUNCIÓN FINAL — Lo que usarán train.py y generate_embeddings.py
# ----------------------------------------------------------

def cargar_datos_finales():
    df_clima, df_suelo, df_rad = load_raw_data()
    df_clima, df_suelo, df_rad = index_by_cvegeo(df_clima, df_suelo, df_rad)

    numeric_cols, categorical_cols = get_soil_column_groups(df_suelo)

    (
        df_clima_filled, df_rad_filled, df_suelo_num_filled,
        df_suelo_cat_filled, mask_clima, mask_rad, mask_soil_num
    ) = fill_and_mask(df_clima, df_rad, df_suelo,
                      numeric_cols, categorical_cols)

    df_suelo_cat_encoded, mask_cat = encode_categorical(df_suelo_cat_filled)

    df_clima_scaled, df_rad_scaled, df_suelo_num_scaled = normalize_all(
        df_clima_filled, df_rad_filled, df_suelo_num_filled
    )

    df_suelo_all, mask_soil_all = merge_soil(
        df_suelo_num_scaled, df_suelo_cat_encoded,
        mask_soil_num, mask_cat
    )

    return create_vectors(
        df_clima_scaled, df_suelo_all, df_rad_scaled,
        mask_clima, mask_rad, mask_soil_all
    )
