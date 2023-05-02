from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import torch
def evaluate_models(G, D_G, D_L, dataloader, discriminator_loss, criterion_context, generator_loss, face_parsing_model, device):
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
            images = images.to(device)
            masks = masks.to(device)
            masked_images = masked_images.to(device) 
            
            # generator loss
            first_out, second_out  = G(images, masks)
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

def evaluate_model_external(model, dataloader, device):
    model.eval()
    total_psnr = 0
    total_ssim = 0
    total_samples = 0

    with torch.no_grad():
        for i, (images, masks, masked_images) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.to(device) 
            _, completed_images = model(images, masks)
 
            for i in range(images.size(0)):
                image = images[i].cpu().numpy().transpose(1, 2, 0)
                completed_image = completed_images[i].cpu().numpy().transpose(1, 2, 0)

                total_psnr += psnr(image, completed_image)
                total_ssim += ssim(image, completed_image, multichannel=True, win_size=3, data_range=1)

            total_samples += images.size(0)

    return total_psnr / total_samples, total_ssim / total_samples

