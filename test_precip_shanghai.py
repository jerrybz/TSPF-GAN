import numpy as np
import torch
import os

from tqdm import tqdm

from models.Generator import Generator
from root import ROOT_DIR


from utils.metrics import Evaluator
from functools import partial
import matplotlib.pyplot as plt
from matplotlib import colors


def vis_res(pred_seq, gt_seq, save_path, data_type='vil',
            save_grays=False, do_hmf=False, save_colored=False,
            pixel_scale=None, thresholds=None, gray2color=None, hfm_colors=None,
            ):
    # pred_seq: ndarray, [T, C, H, W], value range: [0, 1] float
    if isinstance(pred_seq, torch.Tensor) or isinstance(gt_seq, torch.Tensor):
        pred_seq = pred_seq.detach().cpu().numpy()
        gt_seq = gt_seq.detach().cpu().numpy()
    pred_seq = pred_seq.squeeze()
    gt_seq = gt_seq.squeeze()
    os.makedirs(save_path, exist_ok=True)

    if save_grays:
        os.makedirs(os.path.join(save_path, 'pred'), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'targets'), exist_ok=True)
        for i, (pred, gt) in enumerate(zip(pred_seq, gt_seq)):

            plt.imsave(os.path.join(save_path, 'pred', f'{i}.png'), pred, cmap='gray', vmax=1.0, vmin=0.0)
            plt.imsave(os.path.join(save_path, 'targets', f'{i}.png'), gt, cmap='gray', vmax=1.0, vmin=0.0)

    if data_type == 'vil':
        pred_seq = pred_seq * pixel_scale
        pred_seq = pred_seq.astype(np.uint8)
        gt_seq = gt_seq * pixel_scale
        gt_seq = gt_seq.astype(np.uint8)

    colored_pred = np.array([gray2color(pred_seq[i], data_type=data_type) for i in range(len(pred_seq))],
                            dtype=np.float64)
    colored_gt = np.array([gray2color(gt_seq[i], data_type=data_type) for i in range(len(gt_seq))], dtype=np.float64)

    if save_colored:
        os.makedirs(os.path.join(save_path, 'pred_colored'), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'targets_colored'), exist_ok=True)
        for i, (pred, gt) in enumerate(zip(colored_pred, colored_gt)):
            plt.imsave(os.path.join(save_path, 'pred_colored', f'{i}.png'), pred)
            plt.imsave(os.path.join(save_path, 'targets_colored', f'{i}.png'), gt)

    grid_pred = np.concatenate([
        np.concatenate([i for i in colored_pred], axis=-2),
    ], axis=-3)
    grid_gt = np.concatenate([
        np.concatenate([i for i in colored_gt], axis=-2, ),
    ], axis=-3)

    grid_concat = np.concatenate([grid_pred, grid_gt], axis=-3, )
    plt.imsave(os.path.join(save_path, 'all.png'), grid_concat)

    if do_hmf:
        def hit_miss_fa(y_true, y_pred, thres):
            mask = np.zeros_like(y_true)
            mask[np.logical_and(y_true >= thres, y_pred >= thres)] = 4
            mask[np.logical_and(y_true >= thres, y_pred < thres)] = 3
            mask[np.logical_and(y_true < thres, y_pred >= thres)] = 2
            mask[np.logical_and(y_true < thres, y_pred < thres)] = 1
            return mask

        grid_pred = np.concatenate([
            np.concatenate([i for i in pred_seq], axis=-1),
        ], axis=-2)
        grid_gt = np.concatenate([
            np.concatenate([i for i in gt_seq], axis=-1),
        ], axis=-2)

        hmf_mask = hit_miss_fa(grid_pred, grid_gt, thres=thresholds[2])
        plt.axis('off')
        plt.imsave(os.path.join(save_path, 'hmf.png'), hmf_mask, cmap=colors.ListedColormap(hfm_colors))


def get_model_losses(model_file, model_name, data_file, save_dir):
    from utils.dataset_shanghai import Shanghai, PIXEL_SCALE, THRESHOLDS, gray2color,HMF_COLORS
    dataset = Shanghai(
        data_path=data_file,
        type="test",
        img_size=128
    )

    test_dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    device = torch.device("cuda")
    model = Generator
    model = model.load_from_checkpoint(f"{model_file}")
    eval = Evaluator(
        seq_len=25,
        value_scale=PIXEL_SCALE,
        thresholds=THRESHOLDS,
        save_path=save_dir,
    )
    color_fn = partial(vis_res,
                    pixel_scale = PIXEL_SCALE,
                    thresholds = THRESHOLDS,
                    gray2color = gray2color,
                    hfm_colors = HMF_COLORS,)
    cnt = 0
    for x,y_true in tqdm(test_dl,desc='Test Samples'):
        x = x.to(device)
        y_true = y_true.to(device)
        y_pred = model(x)
        eval.evaluate(y_true, y_pred)
        if cnt==0:
            color_fn(y_pred, y_true, save_path=os.path.join(save_dir, f"{cnt}-{cnt}"), data_type='vil',save_colored=True)
        cnt += 1

    res = eval.done()
    print(f"Test Results: {res}")
    print("=" * 30)

    return res

def losses_to_csv(losses, path):
    csv = "threshold, name, csi, hss, ssim\n"
    for loss in losses:
      row = ",".join(str(l) for l in loss)
      csv += row + "\n"

    with open(path,"w+") as f:
      f.write(csv)

    return csv

if __name__ == "__main__":
    denormalize=True
    # Models that are compared should be in this folder (the ones with the lowest validation error)
    model_name = "TSPF-GAN"
    results_folder = os.path.join(ROOT_DIR, "path/to/results")
    model_file = os.path.join(ROOT_DIR,
                              "path/to/checkpoints/")
    data_file = (
        os.path.join(ROOT_DIR, "path/to/dataset"),
    )
    test_losses = get_model_losses(model_file, model_name, data_file, results_folder)
