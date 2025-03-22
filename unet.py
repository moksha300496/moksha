import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: (batch_size, 3, height, width)
        self.layers = nn.Sequential(
            # First layer
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Second layer
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Third layer
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Fourth layer
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # Final layer
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

class UNetSteganography(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder (Hiding Network)
        self.enc1 = DoubleConv(4, 64)  # 3 channels for image + 1 for secret per channel
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        # Decoder (Hiding Network)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(128, 64)
        self.final = nn.Conv2d(64, 3, kernel_size=1)

        # Revelation Network
        self.rev_enc1 = DoubleConv(3, 64)
        self.rev_enc2 = DoubleConv(64, 128)
        self.rev_enc3 = DoubleConv(128, 256)
        self.rev_final = nn.Conv2d(256, 3, kernel_size=1)

    def encode(self, image, secret):
        """
        image: (batch_size, 3, height, width)
        secret: (batch_size, 3) - One value per channel
        """
        # Create secret channel mask
        b, _, h, w = image.shape
        # Expand secret to match spatial dimensions and combine with image
        secret_channels = secret.view(b, 3, 1, 1).expand(-1, -1, h, w)
        # Create a combined input with secret data as additional channel
        x = torch.cat([image, secret_channels[:, 0:1, :, :]], dim=1)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))

        # Decoder
        d1 = self.dec1(torch.cat([self.up1(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), e2], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d2), e1], dim=1))

        # Generate output image
        return torch.tanh(self.final(d3))

    def decode(self, stego_image):
        """
        stego_image: (batch_size, 3, height, width)
        returns: (batch_size, 3)
        """
        # Revelation Network
        x = self.rev_enc1(stego_image)
        x = self.rev_enc2(F.max_pool2d(x, 2))
        x = self.rev_enc3(F.max_pool2d(x, 2))
        x = self.rev_final(x)
        # Global average pooling to get the secret
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return x.squeeze(-1).squeeze(-1)  # Return (batch_size, 3)

class DeepSteganography:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = UNetSteganography().to(device)
        self.discriminator = Discriminator().to(device)
        self.model.eval()
        self.discriminator.eval()

    def encode_image(self, image_tensor, secret_data):
        """
        Encode secret data into image using U-Net
        image_tensor: (1, 3, H, W)
        secret_data: (1, 3)
        """
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            secret_data = secret_data.to(self.device)
            # Generate stego image
            stego_image = self.model.encode(image_tensor, secret_data)
            return stego_image

    def decode_image(self, stego_image):
        """
        Decode secret data from stego image using U-Net
        stego_image: (1, 3, H, W)
        returns: (1, 3)
        """
        with torch.no_grad():
            stego_image = stego_image.to(self.device)
            return self.model.decode(stego_image)

    def train_step(self, image, secret):
        """
        Perform one training step with adversarial loss
        """
        # Generate stego image
        stego_image = self.model.encode(image, secret)

        # Train discriminator
        real_score = self.discriminator(image)
        fake_score = self.discriminator(stego_image.detach())

        # Calculate losses
        mse_loss = F.mse_loss(stego_image, image)
        adversarial_loss = -torch.mean(torch.log(1 - fake_score))

        return stego_image, mse_loss + 0.001 * adversarial_loss