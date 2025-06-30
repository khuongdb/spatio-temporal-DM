import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os


def pad_to_square(arr):
    """
    Pad a 2D array 'arr' so it becomes NxN, with zeros around the edges.
    The original array is centered in the new square.
    """
    h, w = arr.shape
    s = max(h, w)
    padded = np.zeros((s, s), dtype=arr.dtype)
    start_h = (s - h) // 2
    start_w = (s - w) // 2
    padded[start_h:start_h + h, start_w:start_w + w] = arr
    return padded


def plot_comparison_columns_by_view(slices_list, titles, num_slices=3):
    """
    slices_list: a list of 3 sets of slices: [reconstructed, manipulated, difference]
    titles: e.g., ['Reconstructed', 'Manipulated', 'Difference']
    """
    nrows = 3                # Reconstructed / Manipulated / Difference
    ncols = 3 * num_slices   # Axial x N, Coronal x N, Sagittal x N

    # Adjust figure size and create a GridSpec with an extra column for the colorbar
    fig = plt.figure(figsize=(26, 8))  # Increased width for the colorbar
    gs = GridSpec(nrows, ncols + 3, figure=fig, wspace=0.05, hspace=0.2, 
                  width_ratios=[1]*num_slices + [0.2] + [1]*num_slices + [0.2] + [1]*num_slices + [0.2])

    views = ['Axial', 'Coronal', 'Sagittal']
    diff_mappable = None  # To store the mappable for the colorbar

    for view_idx, view in enumerate(views):
        col_offset = view_idx * (num_slices + 1)  # Add 1 for the gap column
        for slice_idx in range(num_slices):
            col_idx = col_offset + slice_idx

            for row_idx, (slices, title) in enumerate(zip(slices_list, titles)):
                ax = fig.add_subplot(gs[row_idx, col_idx])
                
                slice_img, slice_num = slices[view_idx][slice_idx]
                slice_img = pad_to_square(slice_img)
                rotated_img = np.rot90(slice_img)
                cmap = 'jet' if title == 'Difference' else 'gray'
                im = ax.imshow(rotated_img, cmap=cmap)

                # Store the mappable for the 'Difference' row
                if title == 'Difference':
                    diff_mappable = im

                ax.axis('off')
                ax.set_aspect('equal', adjustable='box')

                # Label the top row with the view title
                if row_idx == 0 and slice_idx == round(num_slices / 2) - 1:
                    ax.set_title(f'{view}', fontsize=14, pad=10)

                # Add row titles on the left side
                if col_idx == 0 and slice_idx == 0:
                    ax.text(-0.15, 0.5, title, fontsize=12, va='center', ha='center', 
                            transform=ax.transAxes, rotation=90)

    # Add gaps to separate the slicing planes
    for gap_col in [num_slices, 2*num_slices + 1]:
        for row in range(nrows):
            ax = fig.add_subplot(gs[row, gap_col])
            ax.axis('off')

    # Add a full-height colorbar as a "4th plot"
    if diff_mappable is not None:
        ax = fig.add_subplot(gs[:, -1])
        ax.axis('off')
        cbar_ax = fig.add_subplot(gs[:, -1])  # Span all rows in the last column
        cbar = fig.colorbar(diff_mappable, cax=cbar_ax, aspect=10)
        cbar.set_label('Absolute Difference', fontsize=12, labelpad=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()



def plot_comparison_starmen(imgs, labels, is_errors=None, save=False, save_path=None, show=False, opt=None):
    """
    Plot comparison for starmen dataset. 
    Args: 
        imgs: list[np.array] a list of sets of images. Each list[i] will be a row (of 10 images)
        labels: list[str] a list of row labels, e.g: Origin, Reconstruction, Error
        is_errors: list[boolean] define if a row is a error (difference) images, e.g: x_org - x_recons.  
        row_label that contains "error" will be mapped to "jet" cmap, otherwise it will be "gray"
    """
    if not isinstance(imgs, list):
        imgs = [imgs]

    if is_errors is None:
        is_errors = [False] * len(imgs)
        is_errors[-1] = True

    if opt is None:
        opt = {}

    nrows = len(imgs)
    num_slices = len(imgs[0])
    ncols = 1 * num_slices
    diff_mappable = None

    # Adjust figure size and create a GridSpec with an extra column for the colorbar
    base_size = opt.get("base_size", 2.5)
    w = base_size * (ncols + 0.15)
    h = base_size * nrows
    fig = plt.figure(figsize=(w, h))  # Increased width for the colorbar
    gs = GridSpec(nrows, ncols + 1, figure=fig, wspace=0.05, hspace=0.05, 
                width_ratios=[1]*num_slices + [0.15])

    for i in range(num_slices):
        col_idx = i

        for row_idx, (row_imgs, row_label) in enumerate(zip(imgs, labels)):
            ax = fig.add_subplot(gs[row_idx, col_idx])

            img = row_imgs[i]
            cmap = "jet" if is_errors[row_idx] else "gray"

            im = ax.imshow(img, cmap=cmap)

            # Store the mappable for the error row
            if is_errors[row_idx]:
                diff_mappable = im

            ax.axis("off")
            ax.set_aspect("equal", adjustable="box")

        # Add row titles on the left side
            if col_idx == 0 and i == 0:
                ax.text(-0.15, 0.5, row_label, fontsize=12*base_size/2.5, va='center', ha='center', 
                    transform=ax.transAxes, rotation=90)


    # Add a full-height colorbar as a "4th plot"
    if diff_mappable is not None:
        ax = fig.add_subplot(gs[:, -1])
        ax.axis('off')
        cbar_ax = fig.add_subplot(gs[:, -1])  # Span all rows in the last column
        cbar = fig.colorbar(diff_mappable, cax=cbar_ax, aspect=10)
        cbar.set_label('Absolute Error', fontsize=12*base_size/2.5, labelpad=10)
    
    title = opt.get("title", None)
    if title: 
        plt.suptitle(title, fontsize=16*base_size/2.5, y=0.92)

    # plt.tight_layout()

    if show:
        plt.show()
    if save and save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    return fig




def plot_error_histogram(diff_t, anomaly_diff_t=None, save=False, save_path=None, show=False, opt=None):
    """
    Plot the error histogram for each time point (10 timepoints for starmen dataset)
    diff_t: np.ndarray [T, (B H W)]: pixel-wise error for each timepoint. 
    anomaly_diff_t: np.ndarray [T, (hw)]: pixel-wise error within the anomaly region. 
    """

    if opt is None:
        opt = {}

    nrows, ncols = 2, 5
    minx, maxx = np.min(diff_t), np.max(diff_t)
    bins = np.linspace(minx, maxx, 50)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 6), sharey=True, sharex=True)  

    axes = axes.flatten()

    for i in range(diff_t.shape[0]):  
        axes[i].hist(diff_t[i], bins=bins, color='skyblue', edgecolor='black', alpha=0.7, label="all_img" if i == 0 else None)
        if anomaly_diff_t is not None: 
            axes[i].hist(anomaly_diff_t[i], bins=bins, color='red', edgecolor='black', alpha=0.7, label="anomaly_region" if i == 0 else None)
        axes[i].set_title(f't = {i}')
        axes[i].set_yscale("log")

        # Compute percentiles
        cmap = plt.get_cmap("inferno")

        p95, p975, p99 = np.percentile(diff_t[i], [95, 97.5, 99])

        axes[i].axvline(p95, color=cmap(0.2), linestyle='--', label='95th percentile' if i == 0 else None)
        axes[i].axvline(p975, color=cmap(0.4), linestyle='--', label='97.5th percentile' if i == 0 else None)
        axes[i].axvline(p99, color=cmap(0.6), linestyle='--', label='97.5th percentile' if i == 0 else None)

        y_max = axes[i].get_ylim()[1] 
        axes[i].text(p95, y_max * 0.85, f'p95={p95:.2f}', color=cmap(0.2), rotation=90, va='top', ha='right', fontsize=7)
        axes[i].text(p975, y_max * 0.7, f'p97.5={p975:.2f}', color=cmap(0.4), rotation=90, va='top', ha='right', fontsize=7)
        axes[i].text(p99, y_max * 0.55, f'p99={p99:.2f}', color=cmap(0.6), rotation=90, va='top', ha='right', fontsize=7)


        if i in (0, 5):
            axes[i].set_ylabel('Frequency (log)')
        if i in range(5, 10):
            axes[i].set_xlabel('L1 error')

    # Hide any unused subplots (if any)
    for j in range(i+1, nrows * ncols):
        axes[j].axis('off')

    title = opt.get("title", None)
    if title: 
        plt.suptitle(title, fontsize=12, y=0.95)


    # Add legend: 
    fig.legend(loc='lower center', ncol=5, fontsize=10, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    if show: 
        plt.show()

    if save and save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    return fig