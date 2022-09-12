import torch
import numpy as np
import k_diffusion as K
# from samplers import CompVisDenoiser

from PIL import Image
from torch import autocast
from einops import rearrange, repeat

def pil_img_to_torch(pil_img, half=False):
    image = np.array(pil_img).astype(np.float32) / 255.0
    image = rearrange(torch.from_numpy(image), 'h w c -> c h w')
    if half:
        image = image.half()
    return (2.0 * image - 1.0).unsqueeze(0)

def pil_img_to_latent(model, img, batch_size=1, device='cuda', half=True):
    init_image = pil_img_to_torch(img, half=half).to(device)
    init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
    if half:
        return model.get_first_stage_encoding(model.encode_first_stage(init_image.half()))
    return model.get_first_stage_encoding(model.encode_first_stage(init_image))

def find_noise_for_image(model, modelCS, modelFS, x, prompt, steps=200, cond_scale=0.0, verbose=False, normalize=True, img=None):
    if img:
        x = pil_img_to_latent(modelFS, img, batch_size=1, device='cuda', half=True)

    with torch.no_grad():
        with autocast('cuda'):
            uncond = modelCS.get_learned_conditioning([''])
            cond = modelCS.get_learned_conditioning([prompt])

        s_in = x.new_ones([x.shape[0]])
        dnw = K.external.CompVisDenoiser(model)
        sigmas = dnw.get_sigmas(steps).flip(0)
        sigmas = sigmas.to('cuda')

    if verbose:
        print(sigmas)

    with torch.no_grad():
        with autocast('cuda'):
            for i in range(1, len(sigmas)):
                x_in = torch.cat([x] * 2)
                sigma_in = torch.cat([sigmas[i - 1] * s_in] * 2)
                cond_in = torch.cat([uncond, cond])

                c_out, c_in = [K.utils.append_dims(k, x_in.ndim) for k in dnw.get_scalings(sigma_in)]
                
                if i == 1:
                    t = dnw.sigma_to_t(torch.cat([sigmas[i] * s_in] * 2).to('cpu'))
                else:
                    t = dnw.sigma_to_t(sigma_in.to('cpu'))
                    
                t = t.to('cuda')

                # print("X_in: ", x_in.size())
                # print("C_in: ", c_in.size())
                # print("C_out: ", c_out.size())
                # print("t: ", t.size())
                # print("cond_in: ", cond_in.size())

                eps = model.apply_model(x_in * c_in, t, cond=cond_in)
                # eps = torch.cat([eps] * 10)
                denoised_uncond, denoised_cond = (x_in + eps * c_out).chunk(2)
                
                denoised = denoised_uncond + (denoised_cond - denoised_uncond) * cond_scale
                
                if i == 1:
                    d = (x - denoised) / (2 * sigmas[i])
                else:
                    d = (x - denoised) / sigmas[i - 1]

                dt = sigmas[i] - sigmas[i - 1]
                x = x + d * dt
            
            if normalize:
                return (x / x.std()) * sigmas[-1]
            else:
                return x