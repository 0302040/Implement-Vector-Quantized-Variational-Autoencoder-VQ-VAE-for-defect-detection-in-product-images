import torch
import torch.nn as nn

# Vector Quantizer
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)
        self.commitment_cost = commitment_cost

    def forward(self, z):
        # Reshape for distance computation
        z_flattened = z.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)
        
        # Compute distances to embeddings
        distances = (
            torch.sum(z_flattened ** 2, dim=1, keepdim=True)
            + torch.sum(self.embeddings.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embeddings.weight.t())
        )

        # Get nearest embeddings
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=z.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and reshape back
        quantized = torch.matmul(encodings, self.embeddings.weight).view(z.shape)

        # Compute VQ Loss
        loss = torch.mean((quantized.detach() - z) ** 2) + self.commitment_cost * torch.mean((quantized - z.detach()) ** 2)

        # Straight-through estimator
        quantized = z + (quantized - z).detach()

        return quantized, loss

# Encoder
class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 4, 2, 1),  # 128x128
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 4, 2, 1),  # 64x64
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1)   # maintain 64x64
        )

    def forward(self, x):
        return self.encoder(x)

# Decoder
class Decoder(nn.Module):
    def __init__(self, embedding_dim, hidden_channels):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, hidden_channels, 3, 1, 1),  # 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, hidden_channels, 4, 2, 1),  # 128x128
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, 3, 4, 2, 1),  # 256x256
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)

# VQ-VAE Model
class VQVAE(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=128, embedding_dim=64, num_embeddings=512, commitment_cost=0.25):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(in_channels, hidden_channels)
        self.proj = nn.Conv2d(hidden_channels, embedding_dim, 1)
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(embedding_dim, hidden_channels)

    def forward(self, x):
        z = self.encoder(x)
        z = self.proj(z)
        quantized, vq_loss = self.quantizer(z)
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss
