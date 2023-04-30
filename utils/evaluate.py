from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

def evaluate_models(G, D_G, D_L, dataloader, discriminator_loss, criterion_context, generator_loss):
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

            completed_images = G(masked_images)

            real_labels = torch.ones(images.size(0), 1).to(device)
            fake_labels = torch.zeros(images.size(0), 1).to(device)

            local_real_output = D_L(images)
            local_fake_output = D_L(completed_images)

            global_real_output = D_G(images)
            global_fake_output = D_G(completed_images)

            # Expand the real and fake labels to match the output size of the discriminators
            real_labels_expanded = real_labels.view(real_labels.size(0), 1, 1, 1).expand_as(global_real_output).clone()
            fake_labels_expanded = fake_labels.view(fake_labels.size(0), 1, 1, 1).expand_as(global_fake_output).clone()

            real_labels_expanded = real_labels_expanded.to(device)
            fake_labels_expanded = fake_labels_expanded.to(device)

            loss_adv_global = discriminator_loss(global_real_output, real_labels_expanded) + discriminator_loss(global_fake_output, fake_labels_expanded)
            loss_adv_local = discriminator_loss(local_real_output, real_labels_expanded) + discriminator_loss(local_fake_output, fake_labels_expanded)
            loss_adv = loss_adv_global + loss_adv_local 
            loss_g_global = generator_loss(global_fake_output)
            loss_g_local = generator_loss(local_fake_output)

            expanded_masks = masks.expand_as(images)
            loss_context = criterion_context(completed_images * expanded_masks, images * expanded_masks)
            loss_perceptual = criterion_context(face_parsing_model(completed_images), face_parsing_model(images))

            loss_generator = loss_g_global + loss_g_local + loss_context + loss_perceptual

            total_gen_loss += loss_generator.item()
            total_disc_global_loss += loss_adv_global.item()
            total_disc_local_loss += loss_adv_local.item()
            total_samples += images.size(0)

    mean_gen_loss = total_gen_loss / total_samples
    mean_disc_global_loss = total_disc_global_loss / total_samples
    mean_disc_local_loss = total_disc_local_loss / total_samples

    return mean_gen_loss, mean_disc_global_loss, mean_disc_local_loss

def evaluate_model_external(model, dataloader):
    model.eval()
    total_psnr = 0
    total_ssim = 0
    total_samples = 0

    with torch.no_grad():
        for i, (images, masks, masked_images) in enumerate(dataloader):
            masked_images = masked_images.to(device) 
            completed_images = model(masked_images)
 
            for i in range(images.size(0)):
                image = images[i].cpu().numpy().transpose(1, 2, 0)
                completed_image = completed_images[i].cpu().numpy().transpose(1, 2, 0)

                total_psnr += psnr(image, completed_image)
                total_ssim += ssim(image, completed_image, multichannel=True, win_size=3, data_range=1)

            total_samples += images.size(0)

    return total_psnr / total_samples, total_ssim / total_samples

