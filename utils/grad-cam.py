import numpy as np
import os
import torch
import torch.nn as nn

from models.Generator import Generator
import dataset_precip
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM
from root import ROOT_DIR

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


class SemanticSegmentationTarget:
    def __init__(self, category, mask, device):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if device == 'cuda':
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        return (model_output[self.category, :, :] * self.mask).sum()


def load_model(model, model_folder, device):
    models = [m for m in os.listdir(model_folder) if ".ckpt" in m]
    model_file = models[-1]
    model = model.load_from_checkpoint(f"{model_folder}/{model_file}")
    model.eval()
    model.to(torch.device(device))
    return model


def get_segmentation_data():
    data_file = os.path.join(ROOT_DIR,
                             "path/to/dataset")
    dataset_masked = dataset_precip.precipitation_maps_oversampled_h5(
        in_file=data_file,
        num_input_images=5,
        num_output_images=20,
        train=False,

    )

    test_dl_masked = torch.utils.data.DataLoader(
        dataset_masked,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    return test_dl_masked


def run_cam(model, model_name, target_layers, device):
    test_dl = get_segmentation_data()
    count = 0
    for x, masks, y_true, _ in tqdm(test_dl, leave=False):
        count += 1
        if count < 3333:
            continue
        x = x.to(torch.device(device))
        masks = masks.to(torch.device(device))
        model = model.to(torch.device(device))
        if model_name=="ga_smaAt_gnet":
            input = torch.cat([x, masks], dim=1)
        else:
            input = x
        output = model(input)
        x = torch.sum(x, dim=1)
        output = torch.sum(output[:12], dim=1)

        mask = np.digitize((output[0] * 32.44).detach().cpu().numpy(), np.array([0.5]), right=True)
        mask_float = np.float32(mask)
        image = torch.stack([x[0], x[0], x[0]], dim=2)
        image = image.cpu().numpy()
        targets = [SemanticSegmentationTarget(0, mask_float, device)]
        cam_image = []

        for layer in target_layers:
            with GradCAM(model=model, target_layers=layer) as cam:
                grayscale_cam = cam(input_tensor=input, targets=targets)[0, :]
                cam_image.append(show_cam_on_image(image, grayscale_cam, use_rgb=True))
        return cam_image

# Wrapper for GA-SmaAt-GNet that splits 1 input into 2 because gradcam needs 1 unput
class GradCAMWrapper(nn.Module):
    def __init__(self, model, model_name):
        super(GradCAMWrapper, self).__init__()
        self.model = model
        self.model_name = model_name

    def forward(self, input_tensor):
        # print(input_tensor.shape)
        # Split the input tensor into multiple tensors
        if self.model_name == "ga_smaAt_gnet":
            input_tensor1 = input_tensor[:, :5]
            input_tensor2 = torch.zeros(1,25,64,64)
            output = self.model(input_tensor1, input_tensor2)
        else:
            output = self.model(input_tensor)


        return output


def plot_combined_heatmaps(cbam_heatmaps, tspfm_heatmaps, output_path):
    """Combine and plot CBAM and TSPFM activation heatmaps (2 columns, multiple rows)"""
    num_rows = len(cbam_heatmaps)  # Assuming both models have the same number of layers
    fig, axes = plt.subplots(num_rows, 2, figsize=(4, 2 * num_rows))
    # fig.suptitle("CBAM vs TSPFM Activation Heatmaps", fontsize=16)

    # Set column titles (top)
    axes[0, 0].set_title("CBAM", fontsize=14)
    axes[0, 1].set_title("TSPFM", fontsize=14)

    for row in range(num_rows):
        # Left column: CBAM heatmaps
        axes[row, 0].imshow(cbam_heatmaps[row])
        axes[row, 0].axis("off")
        # Left row labels (Encoder Depth)
        axes[row, 0].text(
            -0.01, 0.5,
            f"Encoder Depth {row}",  # Keep the original order of row numbers (0→lowest layer)
            va='center', ha='right',  # Vertically centered, horizontally right-aligned (close to subplot)
            rotation=90,
            transform=axes[row, 0].transAxes,# Position based on subplot coordinate system
            fontsize=12
        )

        # Right column: TSPFM heatmaps
        axes[row, 1].imshow(tspfm_heatmaps[row])
        axes[row, 1].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, format="eps", bbox_inches='tight')
    plt.close(fig)

def run_tspfm_individual_attention_visualization():
    """Run individual visualization of temporal, spatial, and pixel attention modules in TSPFM"""
    data_file = os.path.join(ROOT_DIR,
                             "path/to/dataset")
    device = 'cpu'

    # Load RadarCast model
    radar_model = Generator.load_from_checkpoint(
        os.path.join(ROOT_DIR,
                     "path/to/checkpoints")
    )
    radar_model.eval()
    radar_model.to(torch.device(device))
    
    # Get test data
    dataset_masked = dataset_precip.precipitation_maps_oversampled_h5(
        in_file=data_file,
        num_input_images=5,
        num_output_images=20,
        train=False,
    )
    
    test_dl_masked = torch.utils.data.DataLoader(
        dataset_masked,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Get a sample data
    for x, masks, y_true, _ in test_dl_masked:
        x = radar_model.inc(x.to(torch.device(device)))
        # Use RadarCast's TSPFM module for inference
        # Assume we use the first TSPFM layer as an example
        tspfm_layer = radar_model.tspfm_layers['layer1']
        
        # Get visualization heatmaps for each attention module
        visualizations = tspfm_layer.visualize_individual_attentions(x, device)
        
        # Generate original input image for comparison
        x_vis = torch.sum(x, dim=1).squeeze().cpu().detach().numpy()
        x_vis = (x_vis - x_vis.min()) / (x_vis.max() - x_vis.min() + 1e-8)  # Normalize to 0-1
        
        # Create a large figure containing original input and three attention heatmaps
        fig, axes = plt.subplots(1, 4, figsize=(12, 4))
        
        # Plot original input
        axes[0].imshow(x_vis, cmap='viridis')
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        # Plot three attention heatmaps
        attention_types = {
            'temporal_attention': 'Temporal Attention',
            'spatial_attention': 'Spatial Attention',
            'pixel_attention': 'Pixel Attention'
        }
        
        for i, (att_type, title) in enumerate(attention_types.items(), 1):
            # Get the corresponding heatmap from visualizations (assuming batch_size=1)
            att_map = visualizations[att_type][0]  # Take results from the first batch
            axes[i].imshow(att_map, cmap='hot')
            axes[i].set_title(title)
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Save in png and pdf formats
        output_path_png = os.path.join(ROOT_DIR, "imgs/tspfm_individual_attention_heatmaps.png")
        output_path_pdf = os.path.join(ROOT_DIR, "imgs/tspfm_individual_attention_heatmaps.pdf")
        
        plt.savefig(output_path_png, bbox_inches='tight', dpi=300)
        plt.savefig(output_path_pdf, bbox_inches='tight')
        plt.close(fig)
        
        print(f"TSPFM attention heatmaps have been saved to:")
        print(f"- PNG: {output_path_png}")
        print(f"- PDF: {output_path_pdf}")
        break  # Process only one sample


def run_tspfm_all_layers_visualization():
    """Run and visualize attention heatmaps for all TSPFM layers, displaying them in a single figure"""
    data_file = os.path.join(ROOT_DIR,
                             "path/to/dataset")
    device = 'cpu'
    
    # Load TSPF-GAN model
    radar_model = Generator.load_from_checkpoint(
        os.path.join(ROOT_DIR,
                     "path/to/checkpoints")
    )
    radar_model.eval()
    radar_model.to(torch.device(device))
    
    # Get test data
    dataset_masked = dataset_precip.precipitation_maps_oversampled_h5(
        in_file=data_file,
        num_input_images=5,
        num_output_images=20,
        train=False,
    )
    
    test_dl_masked = torch.utils.data.DataLoader(
        dataset_masked,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    count = 0
    # Get a sample data
    for x, masks, y_true, _ in test_dl_masked:
        if count<3333:
            count+=1
            continue
        # Save raw input and visualization results for each layer
        layer_inputs = []
        layer_visualizations = []
        layer_names = []
        
        # Process input to get the input of the first layer
        x1 = radar_model.inc(x.to(torch.device(device)))
        layer_inputs.append(x1)
        layer_names.append('layer1')
        
        # Get visualization results for the first layer
        # Enable debug mode for the first layer to get detailed temporal attention information
        layer_visualizations.append(radar_model.tspfm_layers['layer1'].visualize_individual_attentions(x1, device, debug=True))
        
        # Process remaining layers
        x2 = radar_model.down1(x1)
        layer_inputs.append(x2)
        layer_names.append('layer2')
        layer_visualizations.append(radar_model.tspfm_layers['layer2'].visualize_individual_attentions(x2, device))
        
        x3 = radar_model.down2(x2)
        layer_inputs.append(x3)
        layer_names.append('layer3')
        layer_visualizations.append(radar_model.tspfm_layers['layer3'].visualize_individual_attentions(x3, device))
        
        x4 = radar_model.down3(x3)
        layer_inputs.append(x4)
        layer_names.append('layer4')
        layer_visualizations.append(radar_model.tspfm_layers['layer4'].visualize_individual_attentions(x4, device))
        
        x5 = radar_model.down4(x4)
        layer_inputs.append(x5)
        layer_names.append('layer5')
        layer_visualizations.append(radar_model.tspfm_layers['layer5'].visualize_individual_attentions(x5, device))
        
        # Attention types
        attention_types = {
            'temporal_attention': 'Temporal Attention',
            'spatial_attention': 'Spatial Attention',
            'pixel_attention': 'Pixel Attention'
        }
        
        # Create a large figure where each row represents a layer and each column represents an attention type
        fig, axes = plt.subplots(len(layer_names), len(attention_types) + 1, figsize=(16, 20))
        
        # Set column titles
        axes[0, 0].set_title('input', fontsize=20)
        for i, title in enumerate(attention_types.values(), 1):
            axes[0, i].set_title(title, fontsize=20)
        
        # Iterate through each layer and plot input and three attention heatmaps
        for row, (layer_name, layer_input, vis) in enumerate(zip(layer_names, layer_inputs, layer_visualizations)):
            # Plot row titles
            axes[row, 0].text(-0.01, 0.5, layer_name, transform=axes[row, 0].transAxes,
                             va='center', ha='right', fontsize=20, rotation=90)
            
            # Plot input feature map (normalized)
            input_vis = torch.sum(layer_input, dim=1).squeeze().cpu().detach().numpy()
            input_vis = (input_vis - input_vis.min()) / (input_vis.max() - input_vis.min() + 1e-8)
            # Use interpolation='nearest' to avoid blurring caused by downsampling
            axes[row, 0].imshow(input_vis, cmap='viridis', interpolation='nearest')
            axes[row, 0].axis('off')
            
            # Plot three attention heatmaps
            for col, (att_type, title) in enumerate(attention_types.items(), 1):
                att_map = vis[att_type][0]  # Take results from the first batch
                # Use interpolation='nearest' to maintain heatmap clarity
                axes[row, col].imshow(att_map, cmap='hot', interpolation='nearest')
                axes[row, col].axis('off')
        
        plt.tight_layout()
        
        # Save in png and pdf formats
        output_path_png = os.path.join(ROOT_DIR, "imgs/tspfm_all_layers_attention_heatmaps.png")
        output_path_pdf = os.path.join(ROOT_DIR, "imgs/tspfm_all_layers_attention_heatmaps.pdf")
        
        plt.savefig(output_path_png, bbox_inches='tight', dpi=300)
        plt.savefig(output_path_pdf, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Attention heatmaps for all TSPFM layers have been saved to:")
        print(f"- PNG: {output_path_png}")
        print(f"- PDF: {output_path_pdf}")
        break  # Process only one sample


if __name__ == '__main__':
    # Run TSPFM individual attention visualization
    run_tspfm_individual_attention_visualization()
    
    # Run attention visualization for all TSPFM layers
    run_tspfm_all_layers_visualization()