# python optimizedSD/optimized_img2img.py \
# --prompt "Ultrarealistic!! A photo of a comfortable chair in a studio, white background,\
# intricate, highly detailed, oil painting, soft shadows, directional lighting, artstation, illustration, art by Norman Rockwell" \
# --init-img inputs/sd_chairs/chair1.jpg --strength 0.65 --n_iter 1 --n_samples 10 \
# --seed 42 --scale 14 --ddim_steps 50 --H 512 --W 512

######################################################




#for n in {0..40};
#do
#	python optimizedSD/optimized_img2img.py --prompt "Ultrarealistic!! A man with a thick beard wearing a blue and red baseball cap, intricate, highly detailed, digital painting, artstation, illustration, art by alphonse mucha and greg rutkowski" --init-img /home/vthamizharas/Documents/Diffusion/stable-diffusion/inputs/jscott_vid/f_${n}.png --strength 0.45 --n_iter 1 --n_samples 10 --seed 42 --scale 14 --ddim_steps 50 --H 512 --W 512
#done



# python optimizedSD/optimized_img2img.py \
# --prompt "highly detailed office chair with blue cushion. smooth curves, muted colours, hyperrealism, dynamic lighting, art by Rebecca Guay, lush detail, award winning, digital painting, comic book style, trending on artstation,8K" \
# --init-img inputs/sd_chairs/chair1.jpg --strength 0.45 --n_iter 1 --n_samples 10 \
# --seed 99 --scale 14 --ddim_steps 50 --H 512 --W 512

#  python optimizedSD/optimized_img2img.py \
# --prompt "highly detailed office chair with blue cushion designed by renowned architect Zaha Hadid. smooth curves, muted colours, hyperrealism, dynamic lighting, art by Edward Hopper, lush detail, award winning, octane render, trending on artstation,8K" \
# --init-img inputs/sd_chairs/chair1.jpg --strength 0.45 --n_iter 1 --n_samples 10 \
# --seed 462 --scale 14 --ddim_steps 50 --H 512 --W 512


# for n in {0..99}; 
# do 
# 	python optimizedSD/optimized_img2img.py --prompt "Ultrarealistic!! A photo of a comfortable chair in a studio, white background,intricate, highly detailed, digital painting, soft shadows, directional lighting, artstation, comic book, art by Rebecca Guay" --init-img /home/vthamizharas/Documents/instant-ngp/data/nerf/chair/images/r_${n}.png --strength 0.40 --n_iter 1 --n_samples 10 --seed 60 --scale 14 --ddim_steps 50 --H 512 --W 512
# done


# for n in {0..99}; 
# do 
# 	python optimizedSD/optimized_img2img.py --prompt "Ultrarealistic!! A photo of a comfortable chair in a studio, white background,intricate, highly detailed, digital painting, soft shadows, directional lighting, artstation, comic book, art by Rebecca Guay" --init-img /home/vthamizharas/Documents/instant-ngp/data/nerf/chair/images/r_${n}.png --strength 0.65 --n_iter 1 --n_samples 10 --seed 34234 --scale 14 --ddim_steps 50 --H 512 --W 512
# done
