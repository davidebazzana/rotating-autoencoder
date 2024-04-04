import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import math
import einops
from einops import rearrange
from sklearn.cluster import KMeans
import numpy as np
import ShapesDataset
from utils import run_object_discovery_evaluation
import sys
from getopt import GetoptError, getopt

transform = transforms.ToTensor()

ROT_DIMS = 6
BATCH_SIZE = 64
N_EPOCHS = 7
DATASET = "4Shapes"

def main():
    try:
        opts, args = getopt(sys.argv[1:], "", ["epochs="])
    except GetoptError:
        print("Wrong arguments: python autoencoder.py <--epochs=NUMBER_OF_EPOCHS>")

    opts = dict(opts)

    epochs = int(opts["--epochs"])

    dataset_train = ShapesDataset.ShapesDataset(partition="train")
    dataset_val = ShapesDataset.ShapesDataset(partition="eval")

    data_loader = torch.utils.data.DataLoader(dataset=dataset_train,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)

    model = AutoEncoder(rotating_dimensions=ROT_DIMS, batch_size=BATCH_SIZE, dim=32)
    model = model.cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    train(data_loader=data_loader, model=model, criterion=criterion, optimizer=optimizer, num_epochs=epochs)

    data_loader = torch.utils.data.DataLoader(dataset=dataset_val,batch_size=BATCH_SIZE,shuffle=False,drop_last=True)
    val(data_loader=data_loader, model=model, criterion=criterion)

class RotatingConv2d(nn.Module):
    def __init__(self, batch_size: int,
                        in_dim: int,
                        in_channels: int,
                        out_channels: int,
                        kernel_size: int,
                        stride: int = 1,
                        padding: int = 0,
                        dilation: int = 1,
                        rotating_dimensions: int = 1):
        super().__init__()
        self.batch_size = batch_size
        self.rotating_dimensions = rotating_dimensions
        self.layer = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation,
                               bias=False)
        self.activation = nn.ReLU()
        self.norm = nn.BatchNorm2d(num_features=out_channels, affine=True)
        self.out_dim = RotatingConv2d.conv_out_dim(h_w_in=in_dim, padding=padding, dilation=dilation, kernel_size=kernel_size, stride=stride)
        self.bias = nn.Parameter(torch.empty((1, rotating_dimensions, out_channels, 1, 1)))
        self.fan_in = (out_channels*self.layer.kernel_size[0]*self.layer.kernel_size[1])
        self.bias = init_rotation_bias(self.fan_in, self.bias)

    def preprocess(self, x: torch.Tensor):
        # b n c h w -> b*n c h w
        x = rearrange(x, "b n c h w -> (b n) c h w")
        return x

    def postprocess(self, x: torch.Tensor):
        # b*n c h w -> b n c h w
        x = rearrange(x, "(b n) c h w -> b n c h w", b=self.batch_size)
        return x

    def forward(self, x):
        z_out = apply_rotating_layer(x, self.layer, self.preprocess, self.postprocess, self.bias, self.norm)
        return z_out
    
    def conv_out_dim(h_w_in: int, padding: int, dilation: int, kernel_size: int, stride: int):
        res = math.floor((h_w_in+(2*padding)-(dilation*(kernel_size-1))-1)/stride + 1)
        return res
    
def init_rotation_bias(fan_in: int, bias: nn.Parameter) -> nn.Parameter:
    bound = 1 / math.sqrt(fan_in)
    return torch.nn.init.uniform_(bias, -bound, bound)

class RotatingConvTranspose2d(nn.Module):
    def __init__(self, batch_size: int,
                        in_dim: int,
                        in_channels: int,
                        out_channels: int,
                        kernel_size: int,
                        stride: int = 1,
                        padding: int = 0,
                        output_padding: int = 0,
                        dilation: int = 1,
                        rotating_dimensions: int = 1):
        super().__init__()
        self.batch_size = batch_size
        self.rotating_dimensions = rotating_dimensions
        self.layer = nn.ConvTranspose2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               output_padding=output_padding,
                               bias=False,
                               dilation=dilation)
        self.activation = nn.ReLU()
        self.norm = nn.BatchNorm2d(num_features=out_channels, affine=True)
        self.out_dim = RotatingConvTranspose2d.conv_transpose_out_dim(h_w_in=in_dim, padding=padding, dilation=dilation, kernel_size=kernel_size, stride=stride, output_padding=output_padding)
        self.bias = nn.Parameter(torch.empty((1, rotating_dimensions, out_channels, 1, 1)))
        self.fan_in = (out_channels*self.layer.kernel_size[0]*self.layer.kernel_size[1])
        self.bias = init_rotation_bias(self.fan_in, self.bias)

    def preprocess(self, x: torch.Tensor):
        # b n c h w -> b*n c h w
        x = x.reshape(x.shape[0]*x.shape[1], x.shape[2], x.shape[3], x.shape[4])
        return x

    def postprocess(self, x: torch.Tensor):
        # b*n c h w -> b n c h w
        x = x.reshape(self.batch_size, self.rotating_dimensions, x.shape[1], x.shape[2], x.shape[3])
        return x

    def forward(self, x):
        z_out = apply_rotating_layer(x, self.layer, self.preprocess, self.postprocess, self.bias, self.norm)
        return z_out
    
    def conv_transpose_out_dim(h_w_in: int, stride: int, padding: int, dilation: int, kernel_size: int, output_padding):
        res = (h_w_in-1)*stride -2*padding + dilation*(kernel_size-1) + output_padding + 1
        return res
    
class RotatingLinear(nn.Module):
    def __init__(self, batch_size: int, rotating_dimensions: int, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.layer = nn.Linear(in_features=in_dim, out_features=out_dim, bias=False)
        self.bias = nn.Parameter(torch.randn(1, rotating_dimensions, out_dim))
        self.norm = nn.LayerNorm(out_dim, elementwise_affine=True)
        self.batch_size = batch_size
        self.activation = nn.ReLU()
        self.rotating_dimensions = rotating_dimensions
        self.in_dim = in_dim
    
    def preprocess(self, x: torch.Tensor):
        # b n c -> b*n c
        x = x.reshape(x.shape[0]*x.shape[1], x.shape[2])
        return x

    def postprocess(self, x: torch.Tensor):
        # b*n c' -> b n c'
        x = x.reshape(self.batch_size, self.rotating_dimensions, -1)
        return x

    def forward(self, x):
        z_out = apply_rotating_layer(x, self.layer, self.preprocess, self.postprocess, self.bias, self.norm)
        return z_out

def apply_rotating_layer(x: torch.Tensor, layer: nn.Module, preprocess, postprocess, bias: nn.Parameter, norm):
    z = preprocess(x)
    z = layer(z)
    z = postprocess(z)
    psi = z + bias.data
    chi = layer(torch.linalg.vector_norm(x, dim=1))
    m_bind = 0.5*torch.linalg.vector_norm(z, dim=1) + 0.5*chi # b c h w
    m_out = norm(m_bind)
    m_out = nn.functional.relu(m_out)
    z_out = torch.nn.functional.normalize(psi, dim=1) * m_out[:, None]
    return z_out

class OutLayer(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(channels))
        self.bias = nn.Parameter(torch.empty(1, channels, 1, 1))
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # b c h w -> (b*h*w) c
        reconstruction = torch.einsum("b c h w, c -> b c h w", x, self.weight) + self.bias
        return self.activation(reconstruction)

class AutoEncoder(nn.Module):
    def __init__(self, rotating_dimensions: int, batch_size: int, dim: int):
        super().__init__()
        self.rotating_dimensions = rotating_dimensions
        
        conv_1 = RotatingConv2d(batch_size, in_dim=dim, in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1, rotating_dimensions=rotating_dimensions)
        conv_2 = RotatingConv2d(batch_size, in_dim=conv_1.out_dim, in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, rotating_dimensions=rotating_dimensions)
        conv_3 = RotatingConv2d(batch_size, in_dim=conv_2.out_dim, in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, rotating_dimensions=rotating_dimensions)
        conv_4 = RotatingConv2d(batch_size, in_dim=conv_3.out_dim, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, rotating_dimensions=rotating_dimensions)
        conv_5 = RotatingConv2d(batch_size, in_dim=conv_4.out_dim, in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, rotating_dimensions=rotating_dimensions)
        self.enc_linear = RotatingLinear(batch_size=batch_size, rotating_dimensions=rotating_dimensions, in_dim=64*(conv_5.out_dim**2), out_dim=64)
        self.dec_linear = RotatingLinear(batch_size=batch_size, rotating_dimensions=rotating_dimensions, in_dim=64, out_dim=64*(conv_5.out_dim**2))
        conv_transpose_1 = RotatingConvTranspose2d(batch_size=batch_size, in_dim=conv_5.out_dim, in_channels=64, out_channels=64, kernel_size=3, stride=2, output_padding=(1 if DATASET=="4Shapes" else 0), padding=1, rotating_dimensions=rotating_dimensions)
        conv_dec_2 = RotatingConv2d(batch_size, in_dim=conv_transpose_1.out_dim, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, rotating_dimensions=rotating_dimensions)
        conv_transpose_3 = RotatingConvTranspose2d(batch_size=batch_size, in_dim=conv_dec_2.out_dim, in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1, rotating_dimensions=rotating_dimensions)
        conv_dec_4 = RotatingConv2d(batch_size, in_dim=conv_transpose_3.out_dim, in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, rotating_dimensions=rotating_dimensions)
        conv_transpose_5 = RotatingConvTranspose2d(batch_size=batch_size, in_dim=conv_dec_4.out_dim, in_channels=32, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1, rotating_dimensions=rotating_dimensions)

        self.encoder = nn.Sequential(
            conv_1, # b, 16, 14, 14
            conv_2, # b, 32, 7, 7
            conv_3, # b, 64, 3, 3
            conv_4,
            conv_5
        )
        self.decoder = nn.Sequential(
            conv_transpose_1, # b, 32, 7, 7
            conv_dec_2, # b, 16, 14, 14 ... output_padding=1 allows to have 14x14 instead of 13x13
            conv_transpose_3, # b, 1, 28, 28
            conv_dec_4,
            conv_transpose_5
        )
        self.output_layer = OutLayer(1)

    def preprocess(self, x: torch.Tensor, rotating_dimensions: int = 1):
        # b c h w -> b n c h w
        extra_dimensions = einops.repeat(torch.zeros_like(x), "b ... -> b n ...", n=rotating_dimensions - 1)
        y = torch.cat((x[:, None], extra_dimensions), dim=1)
        return y
    
    def postprocess(self, x: torch.Tensor):
        # b n c h w -> b c h w
        y = torch.linalg.vector_norm(x, dim=1)
        return y
    
    def evaluation(self, x: torch.Tensor):
        # b n c h w -> b c h w
        magnitude = torch.linalg.vector_norm(x, dim=1) # b n c h w -> b c h w
        norm_magnitude = torch.ones_like(magnitude)
        masking_idx = torch.where(magnitude <= 0.1)
        norm_magnitude[masking_idx] = 0
        y = torch.nn.functional.normalize(x, dim=1) * norm_magnitude[:, None]
        y = rearrange(y.detach().cpu().numpy(), "b n c h w -> b h w (c n)") # b n c h w -> b h w n*c
        pred_labels = np.zeros(
            (64, 32, 32)
        )
        for image_idx in range(y.shape[0]):
            norm_rotating_output_img = y[image_idx]
            norm_rotating_output_img = rearrange(norm_rotating_output_img, "h w c -> (h w) c")
            kmeans = KMeans(n_clusters=5, random_state=0, n_init=10).fit(norm_rotating_output_img)
            cluster_img = rearrange(kmeans.labels_, "(h w) -> h w", h=32, w=32)
            pred_labels[image_idx] = cluster_img

        return x, pred_labels

    def forward(self, x, gt_labels):
        input = self.preprocess(x, self.rotating_dimensions)
        encoded = self.encoder(input)
        encoded = self.enc_linear(rearrange(encoded, "b n c h w -> b n (c h w)"))
        decoded = self.dec_linear(encoded)
        decoded = self.decoder(rearrange(decoded, "b n (c h w) -> b n c h w", c=64, h=4, w=4))
        if not self.training: 
            n_recon, labels = self.evaluation(decoded)
            metrics = run_object_discovery_evaluation(64, True, labels, gt_labels)
        output = self.postprocess(decoded)
        output = self.output_layer(output)
        if self.training:
            return output
        else:
            return output, labels, n_recon, metrics

def train(data_loader, model, criterion, optimizer, num_epochs):
    outputs = []
    for epoch in range(num_epochs):
        n = 0
        loss_CA = 0
        for (img, _) in data_loader:
            img = img.cuda(non_blocking=True)
            recon = model(img, _)
            loss = criterion(recon, img)

            if n == 0: loss_CA = criterion(recon, img).item()
            else: loss_CA = (criterion(recon, img).item() + n*loss_CA)/(n+1)
            n += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch:{epoch+1}, Loss:{loss_CA:.4f}')
        outputs.append((epoch, img, recon))

    plot(outputs, num_epochs)

def val(data_loader, model, criterion):
    outputs = []
    model.eval()
    with torch.no_grad():
        n = 0
        loss_CA = 0
        for (img, labels) in data_loader:
            img = img.cuda(non_blocking=True)
            recon, labels, n_recon, metrics = model(img, labels)

            if n == 0: loss_CA = criterion(recon, img).item()
            else: loss_CA = (criterion(recon, img).item() + n*loss_CA)/(n+1)
            n += 1
        outputs.append((1, recon, labels))
        print(f'Validation loss: {loss_CA:.4f}')
    
    for key, value in metrics.items():
        print(f"{key}: {value:.4f} \t", end="")
    plot_val(outputs, 1)
    plot_n_recon(n_recon)

def plot_n_recon(n_recon):
    plt.figure(figsize=(9,2))
    plt.gray()
    for k in range(ROT_DIMS):
        plt.subplot(1, ROT_DIMS, k+1)
        plt.imshow(n_recon[0][k][0].cpu().detach().numpy())
    plt.show()

def plot_val(image_pairs, num_epochs):
    for k in range(0, num_epochs, 1):
        plt.figure(figsize=(9,2))
        plt.gray()
        imgs = image_pairs[k][1].cpu().detach().numpy()
        recon = image_pairs[k][2]
        for i, item in enumerate(imgs):
            if i >= 9: break
            plt.subplot(2, 9, i+1)
            plt.imshow(item[0])
        
        for i, item in enumerate(recon):
            if i >= 9: break

            plt.subplot(2, 9, 9+i+1)
            plt.imshow(item)
        
        plt.show()

def plot(image_pairs, num_epochs):
    for k in range(0, num_epochs, 1):
        plt.figure(figsize=(9,2))
        plt.gray()
        imgs = image_pairs[k][1].cpu().detach().numpy()
        recon = image_pairs[k][2].cpu().detach().numpy()
        for i, item in enumerate(imgs):
            if i >= 9: break
            plt.subplot(2, 9, i+1)
            plt.imshow(item[0])

        for i, item in enumerate(recon):
            if i >= 9: break
            plt.subplot(2, 9, 9+i+1)
            plt.imshow(item[0])
        plt.show()

if __name__ == "__main__":
    main()