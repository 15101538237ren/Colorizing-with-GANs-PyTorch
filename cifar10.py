import argparse
import os, math
import tqdm, torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 0)')
parser.add_argument('--dataset', type=str, default='cifar10', help='the name of dataset [places365, cifar10] (default: cifar10)')
parser.add_argument('--dataset-path', type=str, default='dataset', help='dataset path (default: ./dataset)')
parser.add_argument('--checkpoints-dir', type=str, default='checkpoints', help='models are saved here (default: ./checkpoints)')
parser.add_argument('--image-dir', type=str, default='images', help='images saving path (default: ./images)')
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--plot_batch_size", type=int, default=12, help="size of the batches in plotting")
parser.add_argument("--device", type=str, default='cuda', help="whether the algorithm executed on cpu or gpu")
parser.add_argument("--lr", type=float, default=1e-3, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--sample_interval", type=int, default=350, help="interval between image sampling")
parser.add_argument("--l1-weight", type=float, default=100.0, help="weight on L1 term for generator gradient (default: 100.0)")
parser.add_argument("--training", type=bool, default=True, help="training or testing")
opt = parser.parse_args()
print(opt)

def mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

image_dir = os.path.join(opt.image_dir, opt.dataset)
training_img_dir = os.path.join(image_dir, "training")
testing_img_dir = os.path.join(image_dir, "testing")
dataset_dir = os.path.join(opt.dataset_path, opt.dataset)
checkpoints_dir = os.path.join(opt.checkpoints_dir, opt.dataset)

mkdir(dataset_dir)
mkdir(image_dir)
mkdir(training_img_dir)
mkdir(testing_img_dir)
mkdir(checkpoints_dir)

cuda = True #if opt.device == "cuda" else False

training = opt.training
batch_size = opt.batch_size
plot_batch_size = opt.plot_batch_size
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
    def __init__(self, name, output_channels=3, training=True):
        super(Generator, self).__init__()
        self.name = name
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
    def __init__(self, name, training=True):
        super(Discriminator, self).__init__()
        self.name = name
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

# Configure dataset loader
loading_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

gray_scale_transform = transforms.Compose(
    [transforms.ToPILImage(), transforms.Grayscale(num_output_channels=1), transforms.ToTensor(),
     transforms.Normalize(mean=[0.5], std=[0.5])])

# Initialize generator and discriminator
generator = Generator('G', training=training)
discriminator = Discriminator('D', training=training)

if cuda:
    generator.cuda()
    discriminator.cuda()
if training:
    # Loss function
    bce_loss_with_logits = torch.nn.BCEWithLogitsLoss()
    if cuda:
        bce_loss_with_logits.cuda()

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)


    trainset_color = torchvision.datasets.CIFAR10(root=dataset_dir, train=True,
                                            download=False, transform=loading_transform)
    trainloader_color = torch.utils.data.DataLoader(trainset_color, batch_size=batch_size,
                                              shuffle=True, num_workers=1, drop_last=True)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # ----------
    #  Training
    # ----------
    global_step = 0
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    for epoch in range(opt.n_epochs):
        g_losses = []
        d_losses = []
        for i, (real_color_imgs,_) in tqdm.tqdm(enumerate(trainloader_color)):
            # Adversarial ground truths
            valid = Tensor(real_color_imgs.shape[0], 1).fill_(0.9).unsqueeze(0).unsqueeze(0).permute(2, 0, 1, 3) #1.0
            fake = Tensor(real_color_imgs.shape[0], 1).fill_(0.0).unsqueeze(0).unsqueeze(0).permute(2, 0, 1, 3)

            gray_scale_imgs = torch.stack([gray_scale_transform(color_img) for color_img in real_color_imgs])
            # real_color_imgs = rgb_to_lab(real_color_imgs) # convert imgs from RGB to L*A*B color space
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
            # g_loss = - F.logsigmoid(discriminator(colored_imgs_by_generator)).sum() + torch.abs(real_color_imgs - colored_imgs_by_generator).mean() * opt.l1_weight
            g_loss = bce_loss_with_logits(discriminator(colored_imgs_by_generator), valid) + torch.abs(real_color_imgs - colored_imgs_by_generator).mean() * opt.l1_weight
            g_losses.append(g_loss)
            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            # d_loss = - F.logsigmoid(discriminator(real_color_imgs)).sum() - F.logsigmoid(0.9 - discriminator(colored_imgs_by_generator.detach())).sum()
            d_loss = (bce_loss_with_logits(discriminator(real_color_imgs), valid) + bce_loss_with_logits(discriminator(colored_imgs_by_generator.detach()), fake)) / 2.0
            d_losses.append(d_loss)
            d_loss.backward()
            optimizer_D.step()
            if i and (i + 1) % 50 == 0:
                print("Epoch: %d, Iter: %d,\tG loss: %.2f,\tD loss: %.2f" % (epoch, i + 1, g_loss.item(), d_loss.item()))
            if i and (i + 1) % opt.sample_interval == 0:
                gray_scale_imgs = gray_scale_imgs.to('cpu')[0:plot_batch_size, :, :, :].repeat(1, 3, 1, 1)
                colored_imgs_by_generator = colored_imgs_by_generator.detach().to('cpu')[0:plot_batch_size, :, :, :]
                real_color_imgs = real_color_imgs.to('cpu')[0:plot_batch_size, :, :, :]
                # stacked_imgs_tensor = torch.cat((gray_scale_imgs.repeat(1, 3, 1, 1), lab_to_rgb(colored_imgs_by_generator.detach()), lab_to_rgb(real_color_imgs)), 0)
                stacked_imgs_tensor = torch.cat((gray_scale_imgs, colored_imgs_by_generator, real_color_imgs), 0)
                grid_img = torchvision.utils.make_grid(stacked_imgs_tensor, nrow=plot_batch_size)
                image_path = "%s/Epoch%d_%d.png" % (training_img_dir, epoch, i + 1)
                save_image(grid_img, image_path)
                print("Validation image %s saved" % image_path)
        loss_dict = {
            'epoch': epoch,
            'd_losses': d_losses,
            'g_losses': g_losses,
        }
        with torch.no_grad():
            loss_path = '%s/loss_%04d.pt' % (checkpoints_dir, epoch)
            print("%s saved" % loss_path)
            torch.save(loss_dict, loss_path)

        if epoch and (epoch + 1) % 50 == 0:
            model_dict = {
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
            }
            with torch.no_grad():
                model_path = '%s/model_%04d.pt' % (checkpoints_dir, epoch)
                print("%s saved" % model_path)
                torch.save(model_dict, model_path)
else:
    model_path = '%s/model_%04d.pt' % (checkpoints_dir, opt.n_epochs - 1)
    checkpoint = torch.load(model_path)

    testset = torchvision.datasets.CIFAR10(root=dataset_dir, train=False,
                                           download=False, transform=loading_transform)
    testloader = DataLoader(testset, batch_size=plot_batch_size,
                                         shuffle=False, num_workers=1, drop_last=True)
    if cuda:
        generator.cuda()
        discriminator.cuda()
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()

    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    discriminator.eval()

    for i, (real_color_imgs, _) in enumerate(testloader):
        if i and (i + 1) % 10 == 0:
            gray_scale_imgs = torch.stack([gray_scale_transform(color_img) for color_img in real_color_imgs])

            if cuda:
                real_color_imgs = real_color_imgs.cuda()
                gray_scale_imgs = gray_scale_imgs.cuda()

            colored_imgs_by_generator = generator(gray_scale_imgs)

            # stacked_imgs_tensor = torch.cat((gray_scale_imgs.repeat(1, 3, 1, 1), lab_to_rgb(colored_imgs_by_generator.detach()), lab_to_rgb(real_color_imgs)), 0)
            stacked_imgs_tensor = torch.cat((gray_scale_imgs.repeat(1, 3, 1, 1), colored_imgs_by_generator.detach(), real_color_imgs), 0)
            grid_img = torchvision.utils.make_grid(stacked_imgs_tensor, nrow=plot_batch_size)
            save_image(grid_img, "%s/%d.png" % (testing_img_dir, i + 1))
            print("tested %d batch" % (i + 1))