import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
from skimage.metrics import peak_signal_noise_ratio as psnr

def evaluate_models(G, D_G, D_L, dataloader, criterion_rec, criterion_parsing, criterion_ssim, face_parsing_model, device):
    G = G.eval()
    D_G = D_G.eval()
    D_L = D_L.eval()

    total_gen_loss = 0
    total_disc_global_loss = 0
    total_disc_local_loss = 0
    total_samples = 0

    with torch.no_grad():
        for i, (images, masks, masked_images) in enumerate(dataloader):
            if i == 0:
                print(i)
            print(images.shape)
            print(masks.shape)
            images = images.to(device)
            masks = masks.to(device)
            masked_images = masked_images.to(device) 
            
            # generator loss
            first_out, second_out  = G(images, masks)
            first_out_wholeimg = images * (1 - masks) + first_out * masks     
            second_out_wholeimg = images * (1 - masks) + second_out * masks
            loss_l1_1 = criterion_rec(first_out_wholeimg, images)
            loss_l1_2 = criterion_rec(second_out_wholeimg, images)
            loss_ssim_1 = criterion_ssim(first_out_wholeimg, images)
            loss_ssim_2 = criterion_ssim(second_out_wholeimg, images)
            loss_rec_1 = 0.5 * loss_l1_1 + 0.5 * (1 - loss_ssim_1)
            loss_rec_2 = 0.5 * loss_l1_2 + 0.5 * (1 - loss_ssim_2)

            lambda_G = 1.0
            lambda_rec_1 = 100.0
            lambda_rec_2 = 100.0
            lambda_per = 10.0

            loss_P = criterion_parsing(face_parsing_model(second_out_wholeimg), face_parsing_model(images))
            loss_generator = lambda_G * loss_G + lambda_rec_1 * loss_rec_1 + lambda_rec_2 * loss_rec_2 + lambda_per * loss_P
            
            local_real_D = D_L(images)
            local_fake_D = D_L(second_out_wholeimg.detach())

            global_real_D = D_G(images)
            global_fake_D = D_G(second_out_wholeimg.detach())

            loss_local_fake_D = criterion_adv(local_fake_D, target_is_real=False)
            loss_local_real_D = criterion_adv(local_real_D, target_is_real=True)

            loss_global_fake_D = criterion_adv(global_fake_D, target_is_real=False)
            loss_global_real_D = criterion_adv(global_real_D, target_is_real=True)

            #gp = gradient_penalty(discriminator_global, images, completed_images)
            loss_d = (loss_local_real_D + loss_local_fake_D + loss_global_fake_D + loss_global_real_D) * 0.25 #+ lambda_gp * gp


            
            total_gen_loss += loss_generator.item()
            total_disc_global_loss += (loss_global_real_D + loss_global_fake_D).item()
            total_disc_local_loss += (loss_local_real_D + loss_local_fake_D).item()
            total_samples += images.size(0)

    mean_gen_loss = total_gen_loss / total_samples
    mean_disc_global_loss = total_disc_global_loss / total_samples
    mean_disc_local_loss = total_disc_local_loss / total_samples

    return mean_gen_loss, mean_disc_global_loss, mean_disc_local_loss

#Source: https://github.com/Po-Hsun-Su/pytorch-ssim.git
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

def evaluate_model_external(model, dataloader, device, criterion_ssim):
    model.eval()
    total_psnr = 0
    total_ssim = 0
    total_samples = 0

    with torch.no_grad():
        for i, (images, masks, masked_images) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.to(device) 
            
            first_out, second_out  = model(images, masks)
            first_out_wholeimg = images * (1 - masks) + first_out * masks     
            second_out_wholeimg = images * (1 - masks) + second_out * masks
            
            for i in range(images.size(0)):
                image = images[i].cpu().numpy().transpose(1, 2, 0)
                completed_image = second_out_wholeimg[i].cpu().numpy().transpose(1, 2, 0)

                total_psnr += psnr(image, completed_image)
                total_ssim += criterion_ssim(second_out_wholeimg, images).item()

            total_samples += images.size(0)

    return total_psnr / total_samples, total_ssim / total_samples
