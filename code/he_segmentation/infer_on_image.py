from he_segmentation.inference import norm_01, ensemble_softmax_list
from nnunet.training.model_restore import load_model_and_checkpoint_files
import os
import imageio.v3 as imageio
from tqdm import tqdm
import argparse
from glob import glob
import numpy as np


def infer_on_images(model_dir, image_dir, output_dir):
    folds = (0, 1, 2, 3, 4)
    model_dir = os.path.join(model_dir, 'he', 
                            'nnUNetTrainerV2_BN_pathology_DA_ignore0_hed005__nnUNet_RGB_scaleTo_0_1_bs8_ps512')
    # breakpoint()
    trainer, params = load_model_and_checkpoint_files(
        folder=os.path.join(model_dir),
        folds=folds,
        mixed_precision=None,
        checkpoint_name="model_best"
    )

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(image_dir), "predictions")
        os.makedirs(output_dir, exist_ok=True)

    for img_path in tqdm(glob(os.path.join(image_dir, "*"))):
        img = imageio.imread(img_path)
        img = np.expand_dims(img, axis=0)
        img = norm_01(img)

        softmax_list = ensemble_softmax_list(trainer, params, img)
        softmax_mean = np.array(softmax_list).mean(0)
        pred_output = softmax_mean.argmax(axis=-1)
        imageio.imwrite(os.path.join(output_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}_pred.tiff"),
                        pred_output.astype(np.uint8), compression="zlib")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", "-m", help="the path where the models are stored")
    parser.add_argument("--image_dir", "-i", help="Directory where images are stored")
    parser.add_argument("--output_dir", "-o", help="Directory where predictions are saved", default=None, required=False)
    args = parser.parse_args()
    infer_on_images(args.model_dir, args.image_dir, args.output_dir)


if __name__ == "__main__":
    main()
