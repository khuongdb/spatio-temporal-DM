import os
from typing import Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.gridspec import GridSpec


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


def plot_comparison_starmen(
    imgs, 
    labels, 
    is_errors=None, 
    save=False, 
    save_path=None, 
    show=False, 
    opt=None,
    same_cbar=True,
    display_cbar=True
):
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

    for i in range(len(imgs)):
        if isinstance(imgs[i], torch.Tensor):
            imgs[i] = imgs[i].detach().cpu().numpy()

    nrows = len(imgs)
    num_slices = len(imgs[0])
    ncols = 1 * num_slices
    diff_mappable = None

    # Adjust figure size and create a GridSpec with an extra column for the colorbar
    base_size = opt.get("base_size", 2.5)
    w = base_size * (ncols + 0.15)
    h = base_size * nrows
    fig = plt.figure(figsize=(w, h))  # Increased width for the colorbar
    gs = GridSpec(
        nrows,
        ncols + 1,
        figure=fig,
        wspace=0.05,
        hspace=0.05,
        width_ratios=[1] * num_slices + [0.15],
    )

    # Colormap vmin and vmax
    if same_cbar:
        cbar_vmin = opt.get("cbar_vmin", None)
        cbar_vmax = opt.get("cbar_vmax", None)

        error_vals = [imgs[i] for i, is_err in enumerate(is_errors) if is_err]
        error_vals = np.stack(error_vals).squeeze()
        if cbar_vmax is None:
            cbar_vmax = np.max(error_vals)
        if cbar_vmin is None:
            cbar_vmin = np.min(error_vals)

        imshow_kwargs = {}
        if cbar_vmin is not None:
            imshow_kwargs["vmin"] = cbar_vmin
        if cbar_vmax is not None:
            imshow_kwargs["vmax"] = cbar_vmax
        
        display_cbar = True
    else:
        imshow_kwargs = {}
        display_cbar = False

    for i in range(num_slices):
        col_idx = i

        for row_idx, (row_imgs, row_label) in enumerate(zip(imgs, labels)):
            ax = fig.add_subplot(gs[row_idx, col_idx])

            img = row_imgs[i]
            cmap = "jet" if is_errors[row_idx] else "gray"

            # Store the mappable for the error row
            if is_errors[row_idx]:
                im = ax.imshow(img, cmap=cmap, **imshow_kwargs)
                diff_mappable = im
            else:
                im = ax.imshow(img, cmap="gray")

            ax.axis("off")
            ax.set_aspect("equal", adjustable="box")

            # Add row titles on the left side
            if col_idx == 0 and i == 0:
                ax.text(
                    -0.15,
                    0.5,
                    row_label,
                    fontsize=12 * base_size / 2.5,
                    va="center",
                    ha="center",
                    transform=ax.transAxes,
                    rotation=90,
                )

    # Add a full-height colorbar as a "4th plot"
    if display_cbar:
        if diff_mappable is not None:

            ax = fig.add_subplot(gs[:, -1])
            ax.axis("off")
            cbar_ax = fig.add_subplot(gs[:, -1])  # Span all rows in the last column
            cbar = fig.colorbar(diff_mappable, cax=cbar_ax, aspect=10)
            cbar.set_label("Absolute Error", fontsize=12 * base_size / 2.5, labelpad=10)

    title = opt.get("title", None)
    if title:
        plt.suptitle(title, fontsize=16 * base_size / 2.5, y=0.92)

    # plt.tight_layout()

    if show:
        plt.show()
    if save and save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_error_histogram(diff_t, anomaly_diff_t=None, save=False, save_path=None, show=False, y_scale=False, opt=None):
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
        if y_scale: 
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


def plot_kde_pixel(
    imgs,
    labels,
    title=None,
    max_plot=5,
    remove_zeros=False,
    show=False,
    save=False,
    save_path=None,
    plot_type="kde",
    **plot_args,
):
    """
    Plot histogram of pixel intensity for a list of images
    """
    import seaborn as sns

    if not isinstance(imgs, list):
        imgs = [imgs]

    for i in range(len(imgs)):
        if isinstance(imgs[i], torch.Tensor):
            imgs[i] = imgs[i].detach().cpu().numpy()

            # # remove NaN value
            # valid_img = imgs[i][~np.isnan(imgs[i])]
            # imgs[i] = valid_img

    B = len(imgs[0])
    if B > max_plot:
        B = max_plot

    fig, axes = plt.subplots(1, B, figsize=(10, 2.5), sharey=True, sharex=True)  
    

    for i in range(B):

        ax = axes[i]
        for id, img in enumerate(imgs):
            im = img[i]
            if isinstance(im, np.ndarray):
                im = im.squeeze().flatten()
            if remove_zeros:
                im = im[im != 0.]

            if im.size == 0:
                print(f"Warning: Data for label {labels[id]} is empty after zero removal, skipping plot.")
            else:
                if plot_type == "kde":
                    sns.kdeplot(im, bw_adjust=0.5, linewidth=1.0, ax=ax, label=labels[id], **plot_args)
                elif plot_type == "hist": 
                    sns.histplot(im, bins=50, ax=ax, label=labels[id], stat="count", **plot_args)

        # ax.hist(pixels, bins=30, range=(0, 1), histtype='step', color='blue')

        # # KLD annotation
        # ax.text(0.01, 0.95, f"KLD= {klds[idx]:.3f}", ha='left', va='top', transform=ax.transAxes, fontsize=9)

        ax.set_title(f"Img {i}", fontsize=8)
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        ax.set_ylabel("")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=len(imgs))

    if title: 
        plt.suptitle(title, fontsize=10, y=0.95)

    plt.tight_layout(rect=[0, 0.005, 1, 1])

    if show: 
        plt.show()
    if save and save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

    fig.canvas.draw()
    return fig


def draw_featmap(featmap: torch.Tensor,
                 
                    overlaid_image: Optional[np.ndarray] = None,
                    channel_reduction: Optional[str] = 'squeeze_mean',
                    topk: int = 20,
                    arrangement: Tuple[int, int] = (4, 5),
                    resize_shape: Optional[tuple] = None,
                    alpha: float = 0.5,
                    imshow_args: dict = None) -> np.ndarray:
    """Draw featmap.

    original from MMEngine 
    https://github.com/open-mmlab/mmengine/blob/main/mmengine/visualization/visualizer.py#L918

    - If `overlaid_image` is not None, the final output image will be the
        weighted sum of img and featmap.

    - If `resize_shape` is specified, `featmap` and `overlaid_image`
        are interpolated.

    - If `resize_shape` is None and `overlaid_image` is not None,
        the feature map will be interpolated to the spatial size of the image
        in the case where the spatial dimensions of `overlaid_image` and
        `featmap` are different.

    - If `channel_reduction` is "squeeze_mean" and "select_max",
        it will compress featmap to single channel image and weighted
        sum to `overlaid_image`.

    - If `channel_reduction` is None

        - If topk <= 0, featmap is assert to be one or three
        channel and treated as image and will be weighted sum
        to ``overlaid_image``.
        - If topk > 0, it will select topk channel to show by the sum of
        each channel. At the same time, you can specify the `arrangement`
        to set the window layout.

    Args:
        featmap (torch.Tensor): The featmap to draw which format is
            (C, H, W).
        overlaid_image (np.ndarray, optional): The overlaid image.
            Defaults to None.
        channel_reduction (str, optional): Reduce multiple channels to a
            single channel. The optional value is 'squeeze_mean'
            or 'select_max'. Defaults to 'squeeze_mean'.
        topk (int): If channel_reduction is not None and topk > 0,
            it will select topk channel to show by the sum of each channel.
            if topk <= 0, tensor_chw is assert to be one or three.
            Defaults to 20.
        arrangement (Tuple[int, int]): The arrangement of featmap when
            channel_reduction is None and topk > 0. Defaults to (4, 5).
        resize_shape (tuple, optional): The shape to scale the feature map.
            Defaults to None.
        alpha (Union[int, List[int]]): The transparency of featmap.
            Defaults to 0.5.

    Returns:
        np.ndarray: RGB image.
    """
    import matplotlib.pyplot as plt
    assert isinstance(featmap,
                        torch.Tensor), (f'`featmap` should be torch.Tensor,'
                                        f' but got {type(featmap)}')
    assert featmap.ndim == 3, f'Input dimension must be 3, ' \
                                f'but got {featmap.ndim}'
    
    featmap = featmap.detach().cpu()

    if overlaid_image is not None:

        if isinstance(overlaid_image, torch.Tensor):
            overlaid_image = overlaid_image.detach().cpu().squeeze().numpy()


        if overlaid_image.ndim == 2:
            overlaid_image = cv2.cvtColor(overlaid_image,
                                            cv2.COLOR_GRAY2RGB)

        if overlaid_image.shape[:2] != featmap.shape[1:]:
            # warnings.warn(
            #     f'Since the spatial dimensions of '
            #     f'overlaid_image: {overlaid_image.shape[:2]} and '
            #     f'featmap: {featmap.shape[1:]} are not same, '
            #     f'the feature map will be interpolated. '
            #     f'This may cause mismatch problems !')
            if resize_shape is None:
                featmap = F.interpolate(
                    featmap[None],
                    overlaid_image.shape[:2],
                    mode='bilinear',
                    align_corners=False)[0]

    if resize_shape is not None:
        featmap = F.interpolate(
            featmap[None],
            resize_shape,
            mode='bilinear',
            align_corners=False)[0]
        if overlaid_image is not None:
            overlaid_image = cv2.resize(overlaid_image, resize_shape[::-1])

    if channel_reduction is not None:
        assert channel_reduction in [
            'squeeze_mean', 'select_max'], \
            f'Mode only support "squeeze_mean", "select_max", ' \
            f'but got {channel_reduction}'
        if channel_reduction == 'select_max':
            sum_channel_featmap = torch.sum(featmap, dim=(1, 2))
            _, indices = torch.topk(sum_channel_featmap, 1)
            feat_map = featmap[indices]
        else:
            feat_map = torch.mean(featmap, dim=0)
        return convert_overlay_heatmap(feat_map, overlaid_image, alpha)
    elif topk <= 0:
        featmap_channel = featmap.shape[0]
        assert featmap_channel in [
            1, 3
        ], ('The input tensor channel dimension must be 1 or 3 '
            'when topk is less than 1, but the channel '
            f'dimension you input is {featmap_channel}, you can use the'
            ' channel_reduction parameter or set topk greater than '
            '0 to solve the error')
        return convert_overlay_heatmap(featmap, overlaid_image, alpha)
    else:
        row, col = arrangement
        channel, height, width = featmap.shape
        assert row * col >= topk, 'The product of row and col in ' \
                                    'the `arrangement` is less than ' \
                                    'topk, please set the ' \
                                    '`arrangement` correctly'

        # Extract the feature map of topk
        topk = min(channel, topk)
        sum_channel_featmap = torch.sum(featmap, dim=(1, 2))
        _, indices = torch.topk(sum_channel_featmap, topk)
        topk_featmap = featmap[indices]

        fig = plt.figure(frameon=False)
        # Set the window layout
        fig.subplots_adjust(
            left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        dpi = fig.get_dpi()
        fig.set_size_inches((width * col + 1e-2) / dpi,
                            (height * row + 1e-2) / dpi)
        for i in range(topk):
            axes = fig.add_subplot(row, col, i + 1)
            axes.axis('off')
            axes.text(2, 15, f'channel: {indices[i]}', fontsize=10)
            axes.imshow(
                convert_overlay_heatmap(topk_featmap[i], overlaid_image,
                                        alpha), **imshow_args)
        image = img_from_canvas(fig.canvas)
        plt.close(fig)
        return image

def convert_overlay_heatmap(feat_map: Union[np.ndarray, torch.Tensor],
                            img: Optional[np.ndarray] = None,
                            alpha: float = 0.5) -> np.ndarray:
    """Convert feat_map to heatmap and overlay on image, if image is not None.

    Args:
        feat_map (np.ndarray, torch.Tensor): The feat_map to convert
            with of shape (H, W), where H is the image height and W is
            the image width.
        img (np.ndarray, optional): The origin image. The format
            should be RGB. Defaults to None.
        alpha (float): The transparency of featmap. Defaults to 0.5.

    Returns:
        np.ndarray: heatmap
    """
    assert feat_map.ndim == 2 or (feat_map.ndim == 3
                                  and feat_map.shape[0] in [1, 3])
    if isinstance(feat_map, torch.Tensor):
        feat_map = feat_map.detach().cpu().numpy()

    if feat_map.ndim == 3:
        feat_map = feat_map.transpose(1, 2, 0)

    norm_img = np.zeros(feat_map.shape)
    norm_img = cv2.normalize(feat_map, norm_img, 0, 255, cv2.NORM_MINMAX)
    norm_img = np.asarray(norm_img, dtype=np.uint8)
    heat_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)
    heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)

    if img is not None:
        if img.dtype != np.uint8:
            img = (img * 255).clip(0, 255).astype(np.uint8)

        if img.ndim == 2 or img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        heat_img = cv2.addWeighted(img, 1 - alpha, heat_img, alpha, 0)
    return heat_img

def img_from_canvas(canvas) -> np.ndarray:
    """Get RGB image from ``FigureCanvasAgg``.

    Args:
        canvas (FigureCanvasAgg): The canvas to get image.

    Returns:
        np.ndarray: the output of image in RGB.
    """  # noqa: E501
    s, (width, height) = canvas.print_to_buffer()
    buffer = np.frombuffer(s, dtype='uint8')
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    return rgb.astype('uint8')


def filter_gt_ano_region(x, gt_mask):
    """
    Filter (flatten) tensor based on the ground truth mask. 
    Args: 
        x: torch.tensor(B, C, H, W)
        gt_mask: torch.tensor(B, C, H, W)

    Returns: 
        x_in: x values within the gt_mask region.
        x_out: x values outside of the gt_mask region. 
    """
    x = x.squeeze()
    gt_mask = gt_mask.squeeze()
    B = x.shape[0]
    gt_mask[gt_mask != 0] = 1
    x_in = []
    x_out = []
    for i in range(B):
        in_mask = gt_mask[i] != 0
        pin = x[i][in_mask]
        pout = x[i][~in_mask]

        x_in.append(pin.detach().cpu().numpy())
        x_out.append(pout.detach().cpu().numpy())
    return x_in, x_out

def plot_difference(vol1, vol2, title1, title2, offset=10):
    difference = np.abs(vol1 - vol2)
    slices_vol1 = get_slices(vol1, offset=offset)
    slices_vol2 = get_slices(vol2, offset=offset)
    slices_diff = get_slices(difference, offset=offset)

    plot_comparison_columns_by_view(
        [slices_vol1, slices_vol2, slices_diff],
        [title1, title2, "Difference"]
    )

def get_slices(volume, num_slices=3, total_slices_per_plane=None, offset=None):
    """
    Return axial, coronal, sagittal slices from 'volume', with 'num_slices' slices per plane.
    Slices are selected around the center slice (e.g., 80 out of 160), with spacing defined by 'offset'.
    If total_slices_per_plane is not provided, it is inferred from the volume shape.
    If offset is not provided, it defaults to a reasonable value based on the center.
    """ 
    dims = np.array(volume.shape)
    if total_slices_per_plane is None:
        total_slices_per_plane = dims
    center_slices = [total_slices // 2 for total_slices in total_slices_per_plane]
    if offset is None:
        offset = center_slices[0] // 2
    if num_slices == 1:
        idxs = [[center] for center in center_slices]
    else:
        idxs = [
            np.array([center_slices[0] + offset * (i - (num_slices // 2)) for i in range(num_slices)], dtype=int),
            np.array([center_slices[1] + offset * (i - (num_slices // 2)) for i in range(num_slices)], dtype=int),
            np.array([center_slices[2] + offset * (i - (num_slices // 2)) for i in range(num_slices)], dtype=int)
        ]
    for i in range(3):
        idxs[i] = np.clip(idxs[i], 0, dims[i] - 1)
    axial = [(volume[:, :, i], i) for i in idxs[2]]
    coronal = [(volume[:, i, :], i) for i in idxs[1]]
    sagittal = [(volume[i, :, :], i) for i in idxs[0]]
    return axial, coronal, sagittal
