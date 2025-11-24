import torch
from torch.utils.data import DataLoader
from models.multimodal_ae import MultiModalEncoder
from utils.dataload import cargar_datos_finales
import pandas as pd



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- 1. Cargar datos ----
Xc, Xe, Xr, Mc, Me, Mr, municipios = cargar_datos_finales()

class MunicipioDataset(torch.utils.data.Dataset):
    def __init__(self, Xc, Xe, Xr, Mc, Me, Mr):
        self.Xc = torch.tensor(Xc, dtype=torch.float32)
        self.Xe = torch.tensor(Xe, dtype=torch.float32)
        self.Xr = torch.tensor(Xr, dtype=torch.float32)
        self.Mc = torch.tensor(Mc, dtype=torch.float32)
        self.Me = torch.tensor(Me, dtype=torch.float32)
        self.Mr = torch.tensor(Mr, dtype=torch.float32)

    def __len__(self):
        return len(self.Xc)

    def __getitem__(self, idx):
        return (
            self.Xc[idx], self.Xe[idx], self.Xr[idx],
            self.Mc[idx], self.Me[idx], self.Mr[idx]
        )

dataset = MunicipioDataset(Xc, Xe, Xr, Mc, Me, Mr)
loader = DataLoader(dataset, batch_size=64, shuffle=False)

# ---- 2. Cargar modelo ----
model = MultiModalEncoder(
    dim_clima=Xc.shape[1],
    dim_suelo=Xe.shape[1],
    dim_rad=Xr.shape[1],
)
model.load_state_dict(torch.load("modelo_multimodal3.pth"))
model.to(device)
model.eval()

# ---- 3. Generar embeddings ----
embeddings_list = []

with torch.no_grad():
    for clima, suelo, rad, mc, ms, mr in loader:
        clima, suelo, rad = clima.to(device), suelo.to(device), rad.to(device)

        emb, _, _, _ = model(clima, suelo, rad)
        embeddings_list.append(emb.cpu())

embeddings = torch.cat(embeddings_list, dim=0)

# ---- 4. Matriz de similitud ----
norm_emb = embeddings / embeddings.norm(dim=1, keepdim=True)
sim_matrix = norm_emb @ norm_emb.T

# ---- 5. Guardar artefactos ----
torch.save(embeddings, "embeddings4.pt")
torch.save(sim_matrix, "sim_matrix4.pt")

print("Embeddings y matriz de similitud generados correctamente.")

# ---- 6. Guardar Parquet ----
emb_df = pd.DataFrame(
    embeddings.numpy(),
    index=municipios
)
emb_df.to_parquet("embeddings_cvegeo4.parquet")

sim_df = pd.DataFrame(
    sim_matrix.numpy(),
    index=municipios,
    columns=municipios
)
sim_df.to_parquet("sim_matrix4.parquet")

print("Parquets generados correctamente.")