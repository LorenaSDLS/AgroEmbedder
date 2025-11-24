import pandas as pd
import torch
from torch.nn.functional import cosine_similarity


# ===========================
# 1. Cargar artefactos
# ===========================

print("Cargando matrices...")

sim_df = pd.read_parquet("sim_matrix4.parquet")  # NxN
emb_df = pd.read_parquet("embeddings_cvegeo4.parquet")  # N x D


# ===========================
# 2. Elegir dos municipios
# ===========================

# puedes poner cualquier par que exista en el índice
muni_a = emb_df.index[77]   # ejemplo
muni_b = emb_df.index[77]   # ejemplo

print(f"\nMunicipio A: {muni_a}")
print(f"Municipio B: {muni_b}")


# ===========================
# 3. Similitud desde matriz NxN
# ===========================

sim_from_matrix = sim_df.loc[muni_a, muni_b]
print(f"\nSimilitud (usando matriz NxN) = {sim_from_matrix:.6f}")


# ===========================
# 4. Similitud calculada directo de embeddings
# ===========================

emb_a = torch.tensor(emb_df.loc[muni_a].values, dtype=torch.float32)
emb_b = torch.tensor(emb_df.loc[muni_b].values, dtype=torch.float32)

sim_direct = cosine_similarity(emb_a, emb_b, dim=0).item()

print(f"Similitud (calculada directo desde embeddings) = {sim_direct:.6f}")


# ===========================
# 5. Diferencia
# ===========================

diff = abs(sim_from_matrix - sim_direct)
print(f"\nDiferencia = {diff:.12f}")

if diff < 1e-6:
    print("\n✔️ Coinciden: la matriz está correcta.")
else:
    print("\n⚠️ No coinciden: revisar generación de matriz.")
