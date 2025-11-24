import torch
from models.multimodal_ae import MultiModalEncoder, ModalEncoder, ModalDecoder
from utils.losses import compute_loss, masked_mse_loss
from utils.dataload import cargar_datos_finales


# === Cargar todo ===
Xc, Xe, Xr, Mc, Me, Mr, municipios = cargar_datos_finales()

device = "cuda" if torch.cuda.is_available() else "cpu"

model = MultiModalEncoder(
    dim_clima = Xc.shape[1],
    dim_suelo = Xe.shape[1],
    dim_rad   = Xr.shape[1]
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

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

dataset = MunicipioDataset(Xc, Xe, Xr,
                           Mc, Me, Mr)

loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

history = {"total": [], "clima": [], "suelo": [], "rad": [],"total_ponderada": [] }

for epoch in range(400):
    model.train()
    total_loss = 0

    for clima, suelo, rad, mc, ms, mr in loader:
        clima = clima.to(device)
        suelo = suelo.to(device)
        rad = rad.to(device)
      
        mc    = mc.to(device)
        ms    = ms.to(device)
        mr    = mr.to(device)

        # 1. Calcular pérdida
        loss, (lc, ls, lr) = compute_loss(
            model, clima, suelo, rad, mc, ms, mr
        )

        # 2. Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 3. Acumular
        total_loss += loss.item()

        history["total"].append(loss.item())
        history["clima"].append(lc.item())
        history["suelo"].append(ls.item())
        history["rad"].append(lr.item())
        history["total_ponderada"].append(
            2.0*lc.item() + 2.0*ls.item() + 0.5*lr.item()
        )

    print(f"Epoch {epoch+1}, Loss = {total_loss:.4f}")

    # === Guardar modelo entrenado ===
torch.save(model.state_dict(), "modelo_multimodal3.pth")
print("Modelo guardado en modelo_multimodal3.pth")

torch.save({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
}, "checkpoint_multimodal3.pth")
print("Checkpoint guardado en checkpoint_multimodal3.pth")