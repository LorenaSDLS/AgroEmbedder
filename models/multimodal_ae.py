import torch
import torch.nn as nn


class ModalEncoder(nn.Module):
  def __init__(self, input_dim, hidden_dims, output_dim):
    super().__init__()
    layers=[]
    dims=[input_dim]+ hidden_dims + [output_dim]
    for i in range(len(dims)-1):
      layers.append(nn.Linear(dims[i],dims[i+1]))
      if i < len(dims)-2:
        layers.append(nn.ReLU())
    self.encoder=nn.Sequential(*layers)

  def forward(self,x):
    return self.encoder(x)

class ModalDecoder(nn.Module):
  def __init__(self, output_dim, hidden_dims, input_dim):
    super().__init__()
    layers=[]
    dims=[output_dim]+hidden_dims+[input_dim]
    for i in range(len(dims)-1):
      layers.append(nn.Linear(dims[i],dims[i+1]))
      if i < len(dims)-2:
        layers.append(nn.ReLU())

    self.decoder=nn.Sequential(*layers)

  def forward(self,z):
    return self.decoder(z)


class MultiModalEncoder(nn.Module):
  def __init__(self, dim_clima, dim_suelo, dim_rad):
    super().__init__()

    #encoders por modalidad
    self.enc_clima=ModalEncoder(dim_clima, [128], 64)
    self.enc_suelo=ModalEncoder(dim_suelo, [256], 128)
    self.enc_rad=ModalEncoder(dim_rad, [128], 64)

    #fusion --> embedding final

    self.fusion=nn.Sequential(
        nn.Linear(64+128+64, 128),
        nn.ReLU(),
        nn.Linear(128,64)
    )

    #decoders por modalidad
    self.dec_clima=ModalDecoder(64, [128], dim_clima)
    self.dec_suelo=ModalDecoder(64, [256], dim_suelo)
    self.dec_rad=ModalDecoder(64, [128], dim_rad)

  def forward(self, clima, suelo, rad):
    z_clima=self.enc_clima(clima)
    z_suelo=self.enc_suelo(suelo)
    z_rad=self.enc_rad(rad)

    #fusión
    z_comb=torch.cat([z_clima, z_suelo, z_rad], dim=1)
    emb = self.fusion(z_comb)

    #decoders
    reco_clima=self.dec_clima(emb)
    reco_suelo=self.dec_suelo(emb)
    reco_rad=self.dec_rad(emb)

    return emb, reco_clima, reco_suelo, reco_rad
