import argparse
import os, math
import tqdm, torchvision
from skimage.color import rgb2lab, lab2rgb
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 0)')
parser.add_argument('--dataset', type=str, default='cifar10', help='the name of dataset [places365, cifar10] (default: cifar10)')
parser.add_argument('--dataset-path', type=str, default='./dataset', help='dataset path (default: ./dataset)')
parser.add_argument('--checkpoints-path', type=str, default='./checkpoints', help='models are saved here (default: ./checkpoints)')
parser.add_argument('--image-dir', type=str, default='./images', help='images saving path (default: ./images)')
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument('--color-space', type=str, default='lab', help='model color space [lab, rgb] (default: lab)')
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--device", type=str, default='cuda', help="whether the algorithm executed on cpu or gpu")
parser.add_argument("--lr", type=float, default=1e-3, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--l1-weight", type=float, default=100.0, help="weight on L1 term for generator gradient (default: 100.0)")
opt = parser.parse_args()
print(opt)

cuda = True if opt.device == "cuda" else False

def mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
mkdir(opt.image_dir)
mkdir(opt.checkpoints_path)
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
# Calculate asymmetric TensorFlow-like 'SAME' padding for a convolution
def get_same_padding(x, kernel_sz, stride, dialation):
    return max((math.ceil(x / stride) - 1) * stride + (kernel_sz - 1) * dialation + 1 - x, 0)

# Dynamically pad input x with 'SAME' padding for conv with specified args
def pad_same(x, kernel_sz, stride, dialation=(1, 1), value=0.):
    ih, iw = x.size()[-2:]
    pad_h, pad_w = get_same_padding(ih, kernel_sz[0], stride[0], dialation[0]), get_same_padding(iw, kernel_sz[1],
                                                                                                 stride[1],
                                                                                                 dialation[1])
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=value)
    return x

class GeneratorEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, batch_norm=True, kernel_size=4):
        super(GeneratorEncoderBlock, self).__init__()

        layer_modules = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        ]
        if batch_norm == True:
            layer_modules.append(nn.BatchNorm2d(out_channels))
        layer_modules.append(nn.LeakyReLU(negative_slope=0.2))
        self.encoder = nn.Sequential(*layer_modules)
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x):
        x = pad_same(x, kernel_sz=(self.kernel_size, self.kernel_size), stride=(self.stride, self.stride))
        return self.encoder(x)

class GeneratorDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, batch_norm=True, kernel_size=4, padding= 1):
        super(GeneratorDecoderBlock, self).__init__()
        layer_modules = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        ]
        if batch_norm == True:
            layer_modules.append(nn.BatchNorm2d(out_channels))
        layer_modules.append(nn.ReLU())
        self.decoder = nn.Sequential(*layer_modules)
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding

    def forward(self, x):
        x = self.decoder(x)
        return x

class Generator(nn.Module):
    def __init__(self, output_channels=3, training=True):
        super(Generator, self).__init__()
        self.output_channels = output_channels
        self.training = training
        self.encoder_block_1 = GeneratorEncoderBlock(1, 64, 1)
        self.encoder_block_2 = GeneratorEncoderBlock(64, 128, 2)
        self.encoder_block_3 = GeneratorEncoderBlock(128, 256, 2)
        self.encoder_block_4 = GeneratorEncoderBlock(256, 512, 2)
        self.encoder_block_5 = GeneratorEncoderBlock(512, 512, 2)

        self.decoder_block_1 = GeneratorDecoderBlock(512, 512, 2)
        self.decoder_block_2 = GeneratorDecoderBlock(512, 256, 2)
        self.decoder_block_3 = GeneratorDecoderBlock(256, 128, 2)
        self.decoder_block_4 = GeneratorDecoderBlock(128, 64, 2)
        self.last_layer = nn.Conv2d(64, self.output_channels, kernel_size=1, stride=1)

    def forward(self, inputs):
        layers = []
        output = self.encoder_block_1(inputs)
        layers.append(output)
        output = self.encoder_block_2(output)
        layers.append(output)
        output = self.encoder_block_3(output)
        layers.append(output)
        output = self.encoder_block_4(output)
        layers.append(output)
        output = self.encoder_block_5(output)
        layers.append(output)

        output = self.decoder_block_1(output)
        output = nn.Dropout(p=0.5)(output)
        output = output + layers[len(layers) - 0 - 2]

        output = self.decoder_block_2(output)
        output = nn.Dropout(p=0.5)(output)
        output = output + layers[len(layers) - 1 - 2]

        output = self.decoder_block_3(output)
        output = output + layers[len(layers) - 2 - 2]

        output = self.decoder_block_4(output)
        output = output + layers[len(layers) - 3 - 2]

        output = torch.tanh(self.last_layer(output))
        return output

class Discriminator(nn.Module):
    def __init__(self, training=True):
        super(Discriminator, self).__init__()
        self.training = training
        self.discriminator_block_1 = GeneratorEncoderBlock(3, 64, 2, False)
        self.discriminator_block_2 = GeneratorEncoderBlock(64, 128, 2)
        self.discriminator_block_3 = GeneratorEncoderBlock(128, 256, 2)
        self.discriminator_block_4 = GeneratorEncoderBlock(256, 512, 1)
        self.last_layer = nn.Conv2d(512, 1, kernel_size=4, stride=1)

    def forward(self, inputs):
        output = self.discriminator_block_1(inputs)
        output = self.discriminator_block_2(output)
        output = self.discriminator_block_3(output)
        output = self.discriminator_block_4(output)
        output = self.last_layer(output)
        return output

def convert(input_, type_):
    return {
        'float': input_.float(),
        'double': input_.double(),
    }.get(type_, input_)

def generic_transform_sk_4d(transform, in_type='', out_type=''):
    def apply_transform(input_):
        to_squeeze = (input_.dim() == 3)
        device = input_.device
        input_ = input_.cpu()
        input_ = convert(input_, in_type)

        if to_squeeze:
            input_ = input_.unsqueeze(0)

        input_ = input_.permute(0, 2, 3, 1).numpy()
        transformed = transform(input_)
        output = torch.from_numpy(transformed).float().permute(0, 3, 1, 2)
        if to_squeeze:
            output = output.squeeze(0)
        output = convert(output, out_type)
        return output.to(device)
    return apply_transform

rgb_to_lab = generic_transform_sk_4d(rgb2lab)
lab_to_rgb = generic_transform_sk_4d(lab2rgb)
# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()
# Loss function
adversarial_loss = torch.nn.BCEWithLogitsLoss()
if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure dataset loader
loading_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

gray_scale_transform = transforms.Compose(
        [transforms.ToPILImage(), transforms.Grayscale(num_output_channels=1), transforms.ToTensor(),
         transforms.Normalize(mean=[0.5], std=[0.5])])

dataset_dir = os.path.join(opt.dataset_path, opt.dataset)
batch_size = opt.batch_size
num_workers = 1
trainset_color = torchvision.datasets.CIFAR10(root=dataset_dir, train=True,
                                        download=False, transform=loading_transform)
trainloader_color = torch.utils.data.DataLoader(trainset_color, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers, drop_last=True)

testset = torchvision.datasets.CIFAR10(root=dataset_dir, train=False,
                                       download=False, transform=loading_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=num_workers, drop_last=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# ----------
#  Training
# ----------
global_step = 0
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

for epoch in range(opt.n_epochs):
    for i, (real_color_imgs,_) in tqdm.tqdm(enumerate(trainloader_color)):
        # Adversarial ground truths
        valid = Tensor(real_color_imgs.shape[0], 1).fill_(0.9).unsqueeze(0).unsqueeze(0).permute(2, 0, 1, 3) #1.0
        fake = Tensor(real_color_imgs.shape[0], 1).fill_(0.0).unsqueeze(0).unsqueeze(0).permute(2, 0, 1, 3)

        gray_scale_imgs = torch.stack([gray_scale_transform(color_img) for color_img in real_color_imgs])
        real_color_imgs = rgb_to_lab(real_color_imgs) # convert imgs from RGB to L*A*B color space
        # real_color_imgs = Variable(real_color_imgs.type(Tensor))
        if cuda:
            gray_scale_imgs = gray_scale_imgs.cuda()
            real_color_imgs = real_color_imgs.cuda()

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()
        # Generate a batch of images
        colored_imgs_by_generator = generator(gray_scale_imgs)

        # Loss measures generator's ability to fool the discriminator
        # g_loss = - F.logsigmoid(discriminator(colored_imgs_by_generator)).mean() \


        g_loss = adversarial_loss(discriminator(colored_imgs_by_generator), valid) + torch.abs(real_color_imgs - colored_imgs_by_generator).mean() * opt.l1_weight

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        # d_loss = - F.logsigmoid(discriminator(real_color_imgs)).mean() - F.logsigmoid(0.9 - discriminator(colored_imgs_by_generator.detach())).mean()
        real_loss = adversarial_loss(discriminator(real_color_imgs), valid)
        fake_loss = adversarial_loss(discriminator(colored_imgs_by_generator.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2.0

        d_loss.backward()
        optimizer_D.step()
        print("G loss: %.2f\tD loss: %.2f" % (g_loss, d_loss))

        if i and i % opt.sample_interval == 0:
            stacked_imgs_tensor = torch.cat((gray_scale_imgs.repeat(1, 3, 1, 1), lab_to_rgb(colored_imgs_by_generator.detach()), lab_to_rgb(real_color_imgs)), 0)
            grid_img = torchvision.utils.make_grid(stacked_imgs_tensor, nrow=batch_size)
            save_image(grid_img, "%s/Epoch%d_%d.png" % (opt.image_dir, epoch, i + 1))
    with torch.no_grad():
        torch.save((generator, discriminator), '%s/model_%04d.pt' % (opt.checkpoints_path, epoch))
