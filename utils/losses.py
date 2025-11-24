import torch


def masked_mse_loss(pred,target,mask):
  #mse normal
    mse=(pred - target)**2

  #aplicar mascara: ignorar las posiciones con datos que faltan
    masked=mse*mask
  #evitar división entre 0
    denom=mask.sum()
    if denom==0:
        return torch.tensor(0.0, device=pred.device)
    return masked.sum()/denom


def compute_loss(model, clima, suelo,rad , clima_mask, suelo_mask, rad_mask):
    emb, rec_clima, rec_suelo, rec_rad=model(clima, suelo, rad)

    loss_clima=masked_mse_loss(rec_clima, clima, clima_mask)
    loss_suelo=masked_mse_loss(rec_suelo, suelo, suelo_mask)
    loss_rad=masked_mse_loss(rec_rad, rad, rad_mask)

    # ---- PESOS ----
    w_clima = 2.0   # más importante
    w_suelo = 2.0   # más importante
    w_rad   = 0.5   # menos importante (radiación)

    loss = (w_clima * loss_clima + w_suelo * loss_suelo + w_rad * loss_rad)

    return loss, (loss_clima, loss_suelo, loss_rad)