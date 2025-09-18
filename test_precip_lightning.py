import json

import os
import torch
from torch import nn
import numpy as np
import os
from tqdm import tqdm
import lightning.pytorch as pl

from models.Generator import Generator

from root import ROOT_DIR
from utils import dataset_precip
from utils.psnr import wpsnr

def apply_dropout(m):
    if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
        m.train()

def get_metrics(model, model_name, test_dl, denormalize=True, threshold=0.5, k=10):
    with torch.no_grad():
        device = torch.device("cuda")
        if model_name != "Persistence":
            model.eval()  # or model.freeze()?
            model.apply(apply_dropout)
            model.to(device)
        loss_func = nn.functional.mse_loss

        factor = 1
        if denormalize:
            factor = 32.44

        threshold = threshold
        epsilon = 1e-6

        total_tp = 0
        total_fp = 0
        total_tn = 0
        total_fn = 0

        loss_denorm = 0.0
        f1 = 0.0
        csi = 0.0
        uncertainty = 0.0
        
        # Initialize PSNR and SSIM related variables
        total_psnr = 0.0
        total_ssim = 0.0
        batch_count = 0
        
        # Initialize SSIM calculator
        for x, mask, y_true, _ in tqdm(test_dl, leave=False):

            x = x.to(device)
            mask = mask.to(device)
            y_true = y_true.to(device).squeeze()
            y_true = y_true
            y_pred = None
            y_preds = []

            for _ in range(k):
                pred = model(x)
                y_preds.append(pred.squeeze())
            y_preds = torch.stack(y_preds, dim=0)
            y_pred = torch.mean(y_preds, dim=0)

            # denormalize
            y_pred_adj = y_pred * factor
            y_true_adj = y_true * factor
            # calculate loss on denormalized data
            loss_denorm += loss_func(y_pred_adj, y_true_adj, reduction="sum")

            # Calculate PSNR (before denormalization, using normalized data)
            # Adjust data range to [0,1] for PSNR calculation
            y_pred_norm = y_pred[:12]
            y_true_norm = y_true[:12]
            heavy_mask = (y_pred_adj[:12]>threshold).float().unsqueeze(0)
            # sum all output frames
            y_pred_adj = torch.sum(y_pred_adj[:12], axis=0)
            y_true_adj = torch.sum(y_true_adj[:12], axis=0)

            # convert to masks for comparison
            y_pred_mask = y_pred_adj > threshold
            y_true_mask = y_true_adj > threshold
            y_pred_mask = y_pred_mask.cpu()
            y_true_mask = y_true_mask.cpu()
            if len(y_pred_norm.shape) == 3:
                y_pred_norm = y_pred_norm.unsqueeze(0)  # [20, 64, 64] -> [1, 20, 64, 64]
                y_true_norm = y_true_norm.unsqueeze(0)  # [20, 64, 64] -> [1, 20, 64, 64]
            # Calculate PSNR
            psnr_value = wpsnr([y_pred_norm, y_true_norm],weight_map=heavy_mask, max_val=1.0)
            total_psnr += psnr_value

            batch_count += 1



            tn, fp, fn, tp = np.bincount(y_true_mask.view(-1) * 2 + y_pred_mask.view(-1), minlength=4)
            total_tp += tp
            total_fp += fp
            total_tn += tn
            total_fn += fn

        # uncertainty /= len(test_dl)
        mse_image = loss_denorm / len(test_dl)
        mse_pixel = mse_image / torch.numel(y_true)
        
        # 计算平均PSNR和SSIM
        avg_psnr = total_psnr / batch_count if batch_count > 0 else 0.0
        avg_ssim = total_ssim / batch_count if batch_count > 0 else 0.0
        
        # get metrics
        precision = total_tp / (total_tp + total_fp + epsilon)
        recall = total_tp / (total_tp + total_fn + epsilon)
        f1 = 2 * precision * recall / (precision + recall + epsilon)
        csi = total_tp / (total_tp + total_fn + total_fp + epsilon)
        hss = (total_tp * total_tn - total_fn * total_fp) / (
                    (total_tp + total_fn) * (total_fn + total_tn) + (total_tp + total_fp) * (
                        total_fp + total_tn) + epsilon)
        mcc = calculate_mcc(total_tp, total_tn, total_fp, total_fn)
    return round(mse_pixel.item(),5), round(f1,5), round(csi,5), round(hss,5), round(mcc,5), round(avg_psnr,5), round(avg_ssim,5)

def calculate_mcc(total_tp, total_tn, total_fp, total_fn):
    total_tp = np.array(total_tp, dtype=np.float64)
    total_tn = np.array(total_tn, dtype=np.float64)
    total_fp = np.array(total_fp, dtype=np.float64)
    total_fn = np.array(total_fn, dtype=np.float64)

    numerator = (total_tp * total_tn) - (total_fp * total_fn)
    denominator = np.sqrt((total_tp + total_fp) * (total_tp + total_fn) * (total_tn + total_fp) * (total_tn + total_fn))
    mcc = numerator / denominator if denominator != 0 else 0
    return mcc



def get_model_losses(model_file, model_name, data_file, denormalize, random_seeds=[42, 123, 456]):
    # Ensure at least 3 random seeds are used for statistical significance evaluation
    if len(random_seeds) < 3:
        print(f"Warning: At least 3 random seeds are required for statistical significance evaluation, currently only {len(random_seeds)} available")
        # If the number of seeds is insufficient, add more seeds
        while len(random_seeds) < 3:
            random_seeds.append(random_seeds[-1] + 1)
    
    # Initialize metrics storage for all seeds
    all_seed_losses = {seed: {} for seed in random_seeds}

    # Create dataset and dataloader (fixed across seeds)
    dataset = dataset_precip.precipitation_maps_oversampled_h5(
        in_file=data_file,
        num_input_images=5,
        num_output_images=20,
        train=False)

    test_dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    # load the model
    model_class = Generator

    thresholds = [0.5,10,20]
    for threshold in thresholds:
        print(str(int(threshold * 100)))
        for seed in random_seeds:
            all_seed_losses[seed][f"binary_{str(int(threshold * 100))}"] = []

    # Evaluate across all seeds
    for seed in random_seeds:
        print(f"Evaluating with seed {seed}...")
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Load model with current seed
        model = model_class.load_from_checkpoint(f"{model_file}")

        for threshold in thresholds:
            losses = get_metrics(model, model_name, test_dl, denormalize, threshold=threshold, k=5)
            all_seed_losses[seed][f"binary_{str(int(threshold * 100))}"].append([threshold, model_name] + list(losses))

    # Calculate mean and std across seeds
    test_losses = {}
    for threshold in thresholds:
        threshold_key = f"binary_{str(int(threshold * 100))}"
        test_losses[threshold_key] = []

        # Collect all metrics across seeds
        all_metrics = []
        for seed in random_seeds:
            all_metrics.extend(all_seed_losses[seed][threshold_key])

        # Group by threshold and model name
        metrics_by_model = {}
        for metrics in all_metrics:
            threshold_val, model_name_val = metrics[0], metrics[1]
            key = (threshold_val, model_name_val)
            if key not in metrics_by_model:
                metrics_by_model[key] = []
            metrics_by_model[key].append(metrics[2:])  # mse, f1, csi, hss, mcc, psnr, ssim, uncertainty

        # Calculate mean and std for each model and threshold
        for (threshold_val, model_name_val), metrics_list in metrics_by_model.items():
            metrics_array = np.array(metrics_list)
            mean_metrics = np.mean(metrics_array, axis=0)
            std_metrics = np.std(metrics_array, axis=0)
            
            # Calculate coefficient of variation (CV = std/mean) to evaluate result stability
            cv_metrics = np.divide(std_metrics, mean_metrics, out=np.zeros_like(std_metrics), where=mean_metrics!=0)
            
            # Print statistical information
            metric_names = ['MSE', 'F1', 'CSI', 'HSS', 'MCC', 'PSNR']
            print(f"\n{model_name_val} (threshold={threshold_val}) Statistical Results:")
            print(f"{'Metric':<8} {'Mean':<12} {'Std Dev':<12} {'CV':<12}")
            print("-" * 50)
            for i, (mean, std, cv) in enumerate(zip(mean_metrics, std_metrics, cv_metrics)):
                if i < len(metric_names):
                    print(f"{metric_names[i]:<8} {mean:<12.5f} {std:<12.2e} {cv:<12.5f}")

            # Create mean ± std strings
            formatted_metrics = []
            for mean, std in zip(mean_metrics, std_metrics):
                formatted_metrics.append(f"{mean:.5f} ± {std:.2e}")

            # Add to test_losses
            test_losses[threshold_key].append([threshold_val, model_name_val] + formatted_metrics)

    return test_losses

def losses_to_csv(losses, path):
    # Check if we have mean ± std format
    has_std = any(isinstance(item, str) and '±' in item for loss in losses for item in loss[2:])

    if has_std:
        csv = "threshold, name, mse_mean_std, f1_mean_std, csi_mean_std, hss_mean_std, mcc_mean_std, psnr_mean_std\n"
    else:
        csv = "threshold, name, mse, f1, csi, hss, mcc, psnr\n"

    for loss in losses:
        # Uniformly display standard deviation in scientific notation in "mean ± std" format
        def fmt(item):
            if isinstance(item, str) and '±' in item:
                try:
                    mean_str, std_str = item.split('±')
                    mean_val = float(mean_str.strip())
                    std_val = float(std_str.strip())
                    return f"{mean_val:.5f} ± {std_val:.2e}"
                except Exception:
                    return item
            return str(item)

        row = ",".join(fmt(l) for l in loss)
        csv += row + "\n"

    with open(path,"w+") as f:
        f.write(csv)

    return csv

if __name__ == "__main__":
    denormalize=True
    # Models that are compared should be in this folder (the ones with the lowest validation error) ,"GA-SmaAt-GNet","SmaAt-UNet"
    data_file = (
        os.path.join(ROOT_DIR,"path/to/dataset"),
    )
    results_folder = os.path.join(ROOT_DIR, "path/to/results")
    model_file = os.path.join(ROOT_DIR, "path/to/model.ckpt")
    model_name = "TSPF-GAN"

    # Use 5 random seeds for statistical significance evaluation
    test_losses = get_model_losses(model_file, model_name, data_file, denormalize)

    # Save CSV results
    print(losses_to_csv(test_losses['binary_50'], (os.path.join(results_folder, f"{model_name}_res_50.csv"))))
    print(losses_to_csv(test_losses['binary_1000'], (os.path.join(results_folder, f"{model_name}_res_1000.csv"))))
    print(losses_to_csv(test_losses['binary_2000'], (os.path.join(results_folder, f"{model_name}_res_2000.csv"))))
