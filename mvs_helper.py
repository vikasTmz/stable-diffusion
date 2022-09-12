import os
import cv2

nerf_path = "/home/vthamizharas/Documents/instant-ngp/data/nerf/"
transforms = "/home/vthamizharas/Documents/instant-ngp/data/nerf/chair/transforms.json"
input_path = "/home/vthamizharas/Documents/Diffusion/stable-diffusion/outputs/img2img-samples/"

seed_suffix_map = {"C0WL": 34235, "D8F0": 55, "LUJ3": 60,"1YDM":1000}
strength_suffix_map = {"C0WL": 0.65, "D8F0": 0.5, "LUJ3": 0.40,"1YDM":0.2}

# for suffix in seed_suffix_map.keys():
suffix = "1YDM"
for seed in range(seed_suffix_map[suffix], seed_suffix_map[suffix]+10):
    output = os.path.join(nerf_path, 'chair_%s_%d' %
                          (str(strength_suffix_map[suffix]), seed))
    os.system('mkdir %s' % (output))
    os.system('mkdir %s' % (os.path.join(output, 'images')))
    os.system('cp %s %s' % (transforms, output))
    for i in range(0, 100):
        image = cv2.imread(os.path.join(input_path, 'r_%d_%s' % (
            i, suffix), 'seed_%d_%s.png' % (seed, strength_suffix_map[suffix])))
        image = cv2.resize(image, (800,800), interpolation = cv2.INTER_LANCZOS4)

        cv2.imwrite(os.path.join(output, 'images', 'r_%d.png' % (i)), image)

