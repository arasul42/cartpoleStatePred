# import numpy as np
# import matplotlib.pyplot as plt

# def analyze_prediction_regions(log_true, log_pred, save_path="rmse_vs_pole_angle.png"):
#     """
#     Save RMSE vs. pole angle plot, binned over the angle range.
#     """
#     pole_angle = log_true[:, 2]

#     # --- RMSE vs. Pole Angle (binned curve) ---
#     bins = np.linspace(-0.25, 0.25, num=20)
#     bin_indices = np.digitize(pole_angle, bins)
#     rmses = {i: [] for i in range(1, len(bins))}

#     for i in range(1, len(bins)):
#         idxs = np.where(bin_indices == i)[0]
#         if len(idxs) > 0:
#             bin_rmse = np.sqrt(np.mean((log_pred[idxs] - log_true[idxs])**2, axis=0))
#             rmses[i] = bin_rmse
#         else:
#             rmses[i] = np.full(4, np.nan)

#     bin_centers = 0.5 * (bins[:-1] + bins[1:])
#     rmses_array = np.array([rmses[i] for i in range(1, len(bins))])  # shape: [num_bins, 4]

#     labels = ['Cart Pos', 'Cart Vel', 'Pole Angle', 'Ang Vel']  # ✅ add this

#     plt.figure(figsize=(4.72, 4.72))
#     for i in range(4):
#         plt.plot(bin_centers, rmses_array[:, i], label=labels[i])
#     plt.xlabel("Pole Angle (radians)")
#     plt.ylabel("RMSE")
#     plt.title("RMSE vs Pole Angle")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()
#     print(f"✅ RMSE curve saved to: {save_path}")


import numpy as np
import matplotlib.pyplot as plt

def analyze_prediction_regions(log_true, log_pred, save_path="rmse_cartpos_and_poleangle.png"):
    """
    Generates two subplots:
    
    - Left subplot: RMSE as a function of cart position (binned over cart position)
      with two lines:
          * RMSE for Cart Position (state index 0)
          * RMSE for Pole Angle (state index 2)
    
    - Right subplot: RMSE as a function of pole angle (binned over pole angle)
      with two lines:
          * RMSE for Cart Position (state index 0)
          * RMSE for Pole Angle (state index 2)
    
    Parameters:
      log_true (ndarray): True state array of shape [T, 4]
      log_pred (ndarray): Predicted state array of shape [T, 4]
      save_path (str): File path to save the plot.
    """
    # Define two conditioning settings:
    # For the left plot, we condition on cart position (state index 0)
    # For the right plot, we condition on pole angle (state index 2)
    settings = [
        (0, 'Cart Position (m)', 'RMSE vs Cart Position', (-2.4, 2.4)),
        (2, 'Pole Angle (rad)', 'RMSE vs Pole Angle', (-0.25, 0.25))
    ]
    
    # We want to show RMSE curves for two states:
    # - Cart Position (state index 0)
    # - Pole Angle (state index 2)
    state_indices = [0, 2]
    state_labels = ['Cart Position', 'Pole Angle']
    
    plt.figure(figsize=(12, 5))
    
    for subplot_idx, (cond_index, xlabel, title, bin_range) in enumerate(settings, start=1):
        # Condition on the chosen variable
        feature = log_true[:, cond_index]
        bins = np.linspace(bin_range[0], bin_range[1], num=20)
        bin_indices = np.digitize(feature, bins)
        
        # For each bin, compute RMSE separately for the two states
        rmse_values = {s: [] for s in state_indices}
        for i in range(1, len(bins)):
            idxs = np.where(bin_indices == i)[0]
            for s in state_indices:
                if len(idxs) > 0:
                    rmse = np.sqrt(np.mean((log_pred[idxs, s] - log_true[idxs, s]) ** 2))
                else:
                    rmse = np.nan
                rmse_values[s].append(rmse)
        
        # Compute the bin centers for plotting
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        
        plt.subplot(1, 2, subplot_idx)
        for s in state_indices:
            plt.plot(bin_centers, rmse_values[s],
                     label=f'RMSE for {state_labels[state_indices.index(s)]}',
                     linewidth=2)
        plt.xlabel(xlabel)
        plt.ylabel("RMSE")
        plt.title(title)
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ RMSE plots for Cart Position and Pole Angle saved to: {save_path}")
