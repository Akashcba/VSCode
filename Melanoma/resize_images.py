## Resize the images so that all the images are of same size and the model trains well on them...
'''
bigger the size better will be the o/p
'''
import os
import glob
from PIL import Image, ImageFile
from joblib import Parallel, delayed
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

def resize_images(image_path, output_folder, resize):
    base_name = os.path.basename(image_path)
    outpath = os.path.join(output_folder, base_name)
    img = Image.open(image_path)
    img=img.resize(
        (resize[1], resize[0], resample=Image.BILINEAR)
    )
    img.save(outpath)

## Parallelizing the process of resizing

def __name__ == "__main__":
    input_folder = "input_folder_path"
    output_folder = "output_folder_path"
    images=glob.glob(os.path.join(input_folder, "*.jpg"))
    parallel(n_jobs=12)(
        delayed(resize_images)(
            i,
            output_folder,
            (512,512)
        )for i in tqdm(images)
    )

    print("\nJob Done\n")
    