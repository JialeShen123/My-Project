import argparse
from os.path import join, exists
import pathlib
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from dataset import Fluxnet
from model import FluxAttention


class FluxFormerInfer:
    def __init__(self, model_path, data_dir, veg_type_mapping, vis_attention=False):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = FluxAttention()
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.vis_attention = vis_attention
        if vis_attention:
            self.model.decoder.layers[0].multihead_attn.register_forward_hook(self.attention_hook)
        self.data_dir = data_dir
        self.attention_map = []
        self.veg_type_mapping = veg_type_mapping

    def attention_hook(self, model, input, output):
        attention_weight = output[1][0].detach().cpu().numpy()
        self.attention_map.append(attention_weight)

    def infer_site(self, split):
        """Inference for a single site"""
        self.attention_map = []
        dataset = Fluxnet(root_dir=self.data_dir, site_csv_names=[split], is_train=False)
        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, drop_last=True)

        df = pd.DataFrame(columns=["date", "LE_predict", "LE_measure",
                                   "norm_LE_predict", "norm_LE_measure"])

        with torch.no_grad():
            for sample in loader:
                data = sample['inputs'].to(self.device)
                le_all = sample["le_all"].to(self.device)
                sparse = sample['sparse'].to(self.device)
                sparse_max = sample["sparse_max"].to(self.device)
                sparse_min = sample["sparse_min"].to(self.device)
                site_name = sample["site_name"]
                site_date = sample["site_date"]

                # Normalization
                naive_output = torch.mean(sparse, dim=1).squeeze()
                sparse = (sparse - sparse_min) / (sparse_max - sparse_min + 1e-2)

                target = torch.mean(le_all, dim=1, keepdim=True)
                normed_target = (target - sparse_min) / (sparse_max - sparse_min + 1e-2)
                normed_output = self.model(data, sparse)
                output = (normed_output * (sparse_max - sparse_min) + sparse_min).squeeze()

                # Prepare data for DataFrame
                new_data = pd.DataFrame({
                    "date": site_date,
                    "LE_predict": output.detach().cpu().numpy(),
                    "LE_measure": target.detach().cpu().numpy().squeeze(),
                    "norm_LE_predict": normed_output.squeeze().detach().cpu().numpy(),
                    "norm_LE_measure": normed_target.squeeze().detach().cpu().numpy()
                })

                df = pd.concat([df, new_data], ignore_index=True)

        df = df.sort_values(by="date")
        if self.vis_attention:
            attention_map = np.concatenate(self.attention_map, axis=0) if self.attention_map else None
            return df, attention_map
        else:
            return df

    def visualize_attention(self, attention_weights, output_dir, site_name):
        """Optimized attention heatmap visualization with vegetation type"""
        if attention_weights is None:
            print(f"No attention weights available for {site_name}")
            return None

        veg_type = self.veg_type_mapping.get(site_name, "UNK")

        attention_mean = attention_weights.mean(axis=0).reshape(7, 8)
        normalized_attention = (attention_mean - np.min(attention_mean)) / (
                np.max(attention_mean) - np.min(attention_mean) + 1e-8)

        plt.figure(figsize=(16, 12), dpi=120)
        ax = sns.heatmap(
            normalized_attention,
            annot=False,
            cmap="YlOrRd",
            linewidths=0.5,
            linecolor='gray',
            cbar_kws={
                'shrink': 0.8,
                'label': 'Normalized Attention Weight',
                'drawedges': False,
                'ticks': None
            },
            vmin=0,
            vmax=1,
            square=True
        )

        cbar = ax.collections[0].colorbar
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(size=0)

        for i in range(7):
            for j in range(8):
                weight = normalized_attention[i, j]
                ax.text(j + 0.5, i + 0.3, "____",
                        ha='center', va='center',
                        color='white', fontsize=16,
                        fontweight='bold', alpha=0.7)
                text_color = 'white' if weight > 0.6 else 'black'
                font_weight = 'bold' if weight > 0.7 else 'normal'
                ax.text(j + 0.5, i + 0.5, f"{weight:.2f}",
                        ha='center', va='center',
                        color=text_color, fontsize=18,
                        fontweight=font_weight)

        ax.set_title(f"Attention Weights Analysis - {site_name} ({veg_type})",
                     fontsize=24, pad=20, fontweight='bold')
        ax.set_xlabel("Local Time (hour)",
                      fontsize=20, labelpad=15, fontweight='bold')
        ax.set_ylabel("Influencing factors",
                      fontsize=20, labelpad=15, fontweight='bold')

        time_labels = ['1:30', '4:30', '7:30', '10:30',
                       '13:30', '16:30', '19:30', '22:30']
        var_labels = ['VPD', 'TA', 'PA', 'WS',
                      'P', 'LW_IN', 'SW_IN']

        ax.set_xticks(np.arange(8) + 0.5)
        ax.set_yticks(np.arange(7) + 0.5)

        ax.set_xticklabels(time_labels,
                           rotation=45,
                           ha='center',
                           va='top',
                           fontsize=18,
                           fontweight='medium',
                           rotation_mode='anchor',
                           position=(0, -0.02))

        ax.set_yticklabels(var_labels,
                           rotation=0,
                           fontsize=18,
                           fontweight='medium')

        plt.tight_layout()
        heatmap_path = join(output_dir, f"{site_name}_attention.png")
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight',
                    facecolor='white', transparent=False)
        plt.close()
        print(f"Saved attention heatmap to: {heatmap_path}")
        return heatmap_path

    def save_attention_weights(self, attention_weights, output_dir, site_name):
        """Save attention weights to numpy file and CSV file"""
        if attention_weights is None:
            print(f"No attention weights available for {site_name}")
            return None, None

        # Calculate mean attention weights
        attention_mean = attention_weights.mean(axis=0).reshape(7, 8)
        normalized_attention = (attention_mean - np.min(attention_mean)) / (
                np.max(attention_mean) - np.min(attention_mean) + 1e-8)

        # Save as numpy binary file
        npz_path = join(output_dir, f"{site_name}_attention_weights.npz")
        np.savez(npz_path,
                 raw_attention=attention_mean,
                 normalized_attention=normalized_attention)

        # Save as CSV file with labels
        var_labels = ['VPD', 'TA', 'PA', 'WS', 'P', 'LW_IN', 'SW_IN']
        time_labels = ['1:30', '4:30', '7:30', '10:30', '13:30', '16:30', '19:30', '22:30']

        csv_df = pd.DataFrame(normalized_attention,
                              index=var_labels,
                              columns=time_labels)
        csv_path = join(output_dir, f"{site_name}_attention_weights.csv")
        csv_df.to_csv(csv_path)

        print(f"Saved attention weights to: {npz_path} and {csv_path}")
        return npz_path, csv_path

    def get_statistic(self, prediction, measurement):
        residual = prediction - measurement
        residual_square = residual ** 2
        R2 = 1 - residual_square.sum() / measurement.var() / measurement.shape[0]
        RMSE = residual_square.mean() ** 0.5
        MB = residual.mean()
        R = np.corrcoef(prediction, measurement)[0, 1]
        return {"R": R, "R2": R2, "RMSE": RMSE, "MB": MB}


def load_vegetation_types(file_path):
    """Load vegetation types from text file"""
    veg_types = {}
    if exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                if ',' in line:
                    site, veg_type = line.strip().split(',')
                    veg_types[site] = veg_type
    return veg_types


def main():
    plt.ioff()
    parser = argparse.ArgumentParser(description='FLUXNET Data Processing')
    parser.add_argument('--vis', default=True, action=argparse.BooleanOptionalAction,
                        help='Enable attention visualization')
    args = parser.parse_args()

    output_dir = r"E:\Pytorch_Edu\Pytorch\Project610\Model_Output" # Please change it to your actual output path
    veg_type_file = join(output_dir, "Veg-Type.txt")
    veg_type_mapping = load_vegetation_types(veg_type_file)
    print(f"Loaded Land cover types for {len(veg_type_mapping)} sites")

    os.makedirs(output_dir, exist_ok=True)
    print(f"All results will be saved to: {output_dir}")

    config = {
        "model_path": os.path.abspath(
            r"Model_pth/fluxAttention_train62val10test32_single_LE_1030.pth"), # Please change it to your actual output path
        "data_root": os.path.abspath(r"data"),
        "sites": [
            # 10-Typical_Sites
            "FLX_AT-Neu.csv",
            "FLX_AU-ASM.csv",
            "FLX_AU-RDF.csv",
            "FLX_AU-Wom.csv",
            "FLX_CA-Gro.csv",
            "FLX_CA-SF3.csv",
            "FLX_CZ-wet.csv",
            "FLX_IT-CA1.csv",
            "FLX_US-Prr.csv",
            "FLX_DE-Geb.csv",

            # 32-testing sites
            # "FLX_DE-Geb.csv", "FLX_AU-ASM.csv", "FLX_AU-Whr.csv",
            # "FLX_CA-TP4.csv", "FLX_AU-Gin.csv", "FLX_IT-Isp.csv",
            # "FLX_CZ-wet.csv", "FLX_US-SRG.csv", "FLX_IT-Col.csv",
            # "FLX_US-Tw2.csv", "FLX_IT-Noe.csv", "FLX_AU-Cpr.csv",
            # "FLX_AU-RDF.csv", "FLX_US-AR2.csv", "FLX_IT-CA1.csv",
            # "FLX_FR-Pue.csv", "FLX_AU-Ade.csv", "FLX_CH-Fru.csv",
            # "FLX_US-Prr.csv", "FLX_CA-SF3.csv", "FLX_US-NR1.csv",
            # "FLX_AU-GWW.csv", "FLX_US-Ivo.csv", "FLX_US-GLE.csv",
            # "FLX_FI-Lom.csv", "FLX_BE-Lon.csv", "FLX_AU-Wom.csv",
            # "FLX_CA-Obs.csv", "FLX_BE-Bra.csv", "FLX_AT-Neu.csv",
            # "FLX_DE-Tha.csv", "FLX_CA-Gro.csv",
        ]
    }

    if not os.path.exists(config["model_path"]):
        raise FileNotFoundError(f"Model file not found at: {config['model_path']}")

    processor = FluxFormerInfer(
        model_path=config["model_path"],
        data_dir=os.path.join(config["data_root"], "data_dir"),
        veg_type_mapping=veg_type_mapping,
        vis_attention=args.vis
    )

    success_count = 0
    for site in config["sites"]:
        try:
            site_name = site.split(".")[0]
            df, attention_map = processor.infer_site(site)

            # 确保数据有效
            if df is None or df.empty:
                print(f"No valid data for {site_name}")
                continue

            stats = processor.get_statistic(df["LE_predict"].values,
                                            df["LE_measure"].values)

            if args.vis and attention_map is not None:
                # Save visualization and attention weights
                processor.visualize_attention(attention_map, output_dir, site_name)
                processor.save_attention_weights(attention_map, output_dir, site_name)

            df["R"] = stats["R"]
            df["R2"] = stats["R2"]
            df["RMSE"] = stats["RMSE"]
            df["MB"] = stats["MB"]

            csv_path = os.path.join(output_dir, site)
            df.to_csv(csv_path, index=False)

            print(f"\n{site_name} Results:")
            print(f"  R: {stats['R']:.3f}  R²: {stats['R2']:.3f}")
            print(f"  RMSE: {stats['RMSE']:.3f}  MB: {stats['MB']:.3f}")
            print(f"  Files saved to: {csv_path}")

            success_count += 1
        except Exception as e:
            print(f"\nError processing {site}: {str(e)}")

    print(f"\nProcessing complete. Successfully processed {success_count}/{len(config['sites'])} sites.")
    print(f"All results saved to: {output_dir}")


if __name__ == '__main__':
    main()