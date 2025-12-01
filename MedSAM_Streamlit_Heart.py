#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import os, io, gc, tempfile, logging
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import cv2
from skimage.measure import label, regionprops
import torch
from segment_anything import sam_model_registry

# ----------------- Settings -----------------
DEFAULT_SIZE = 1024
USE_BUTTON   = True    # Rechen-Button verwenden (verhindert Re-Run bei jeder Slider-Änderung)

# ----------------- Logging -----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

# ----------------- Utils -----------------
def load_nifti(path):
    try:
        return nib.load(path)
    except Exception as e:
        st.error(f"Error loading file {path}: {e}")
        return None

def resize_cv2(img, size=DEFAULT_SIZE):
    # erwartet 2D float32/float64
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)

def normalize_01(x):
    x = x.astype(np.float32, copy=False)
    mn, mx = float(x.min()), float(x.max())
    d = mx - mn
    if d < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / d

def show_mask(mask, ax):
    color = np.array([251/255, 252/255, 30/255, 0.6], dtype=np.float32)
    h, w = mask.shape[-2:]
    ax.imshow(mask.reshape(h, w, 1) * color.reshape(1, 1, -1))

def show_box(box, ax):
    x0, y0 = float(box[0]), float(box[1])
    w, h  = float(box[2]-box[0]), float(box[3]-box[1])
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="blue",
                               facecolor=(0,0,0,0), lw=2))

def get_bounding_boxes_from_mask(mask, min_area=50):
    labeled = label(mask > 0)
    boxes = []
    for r in regionprops(labeled):
        if r.area >= min_area:
            minr, minc, maxr, maxc = r.bbox
            boxes.append([minc, minr, maxc, maxr])  # [x_min,y_min,x_max,y_max]
    return boxes

# ----------------- MedSAM: CPU-Decoding (robust, auch ohne GPU nutzbar) -----------------
@torch.no_grad()
def medsam_decode_boxes_cpu(medsam_model, img_embed, boxes_1024, out_h=DEFAULT_SIZE, out_w=DEFAULT_SIZE, device="cpu"):
    """
    CPU-schneller Pfad: pro Box decodieren, in Low-Res thresholden,
    anschl. mit OpenCV NEAREST auf (out_h,out_w) hochskalieren.
    Returns: (combined (H,W) uint8, per_box (N,H,W) uint8 or None)
    """
    if len(boxes_1024) == 0:
        return None, None

    combined = np.zeros((out_h, out_w), dtype=np.uint8)
    per_box = []
    img_pe = medsam_model.prompt_encoder.get_dense_pe()  # (1,C,H',W')

    for box in boxes_1024:
        box_t = torch.as_tensor(box, dtype=torch.float, device=device).view(1, 1, 4)
        sparse, dense = medsam_model.prompt_encoder(points=None, boxes=box_t, masks=None)

        logits, _ = medsam_model.mask_decoder(
            image_embeddings=img_embed,
            image_pe=img_pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        probs = torch.sigmoid(logits).squeeze().detach().cpu().numpy()   # (h0,w0)
        mask_lr = (probs > 0.5).astype(np.uint8)
        mask = cv2.resize(mask_lr, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

        per_box.append(mask)
        combined = np.maximum(combined, mask)

    per_box = np.stack(per_box, axis=0) if per_box else None
    return combined, per_box

@torch.no_grad()
def process_slice_multi(image_slice, mask_slice, medsam_model, device, min_area=50):
    """
    Erzeugt Bounding Boxes aus der GT-Maske, berechnet Embedding einmal
    und decodiert pro Box (CPU-Pfad).
    """
    if np.sum(mask_slice) == 0:
        return None

    boxes = get_bounding_boxes_from_mask(mask_slice, min_area=min_area)
    if len(boxes) == 0:
        return None

    H, W = image_slice.shape
    resized = resize_cv2(image_slice, DEFAULT_SIZE)
    norm = normalize_01(resized)
    image_rgb = np.stack([norm]*3, axis=-1).astype(np.float32)

    boxes = np.asarray(boxes, dtype=np.float32)
    boxes_1024 = boxes / np.array([W, H, W, H], dtype=np.float32) * float(DEFAULT_SIZE)

    image_tensor = torch.from_numpy(image_rgb).permute(2,0,1).unsqueeze(0).to(device, non_blocking=True)

    # Embedding einmal berechnen (CPU oder GPU – der Decode bleibt CPU-Style pro Box)
    img_embed = medsam_model.image_encoder(image_tensor)

    seg_combined, seg_per_box = medsam_decode_boxes_cpu(
        medsam_model, img_embed, boxes_1024, out_h=DEFAULT_SIZE, out_w=DEFAULT_SIZE, device=device
    )

    return norm, boxes_1024, seg_combined, seg_per_box

# ----------------- Streamlit App -----------------
@st.cache_resource(show_spinner=False)
def load_model(ckpt_path, device):
    model = sam_model_registry["vit_b"](checkpoint=ckpt_path).to(device).eval()
    return model

def main():
    st.set_page_config(page_title="MedSAM — Multi-Box (CPU-stabil)", layout="wide")
    st.title("MedSAM Medical Image Segmentation")

    image_file = st.sidebar.file_uploader("Upload Image (NIfTI)", type=["nii", "nii.gz"])
    mask_file  = st.sidebar.file_uploader("Upload Mask (NIfTI)",  type=["nii", "nii.gz"])
    min_area   = st.sidebar.slider("Min. Component Area (px)", 1, 5000, 10, step=1)

    # Für die Thesis CPU-only, aber wir wählen trotzdem automatisch:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = "Checkpoint/sam_vit_b_01ec64.pth"
    if not os.path.isfile(ckpt):
        st.error(f"Checkpoint nicht gefunden: {ckpt}")
        st.stop()

    with st.spinner("Lade MedSAM (ViT-B)…"):
        medsam = load_model(ckpt, device)

    if image_file and mask_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp_img, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp_msk:
            tmp_img.write(image_file.getbuffer()); img_path = tmp_img.name
            tmp_msk.write(mask_file.getbuffer());   msk_path = tmp_msk.name

        img_nii = load_nifti(img_path); msk_nii = load_nifti(msk_path)
        if img_nii is None or msk_nii is None: st.stop()

        # float32 laden (schneller + weniger RAM)
        image = img_nii.get_fdata(dtype=np.float32)
        mask  = msk_nii.get_fdata(dtype=np.float32)
        if image.ndim != 3:
            st.error("Erwarte 3D-Volumen (H,W,Slices).")
            st.stop()
        st.write(f"Loaded Image: {image.shape}")

        slice_idx = st.slider("Select Slice", 0, image.shape[2]-1, 0)

        run = True
        if USE_BUTTON:
            run = st.button("Segment Slice")

        if run:
            img_slice = image[:, :, slice_idx]
            msk_slice = mask[:, :, slice_idx]

            out = process_slice_multi(img_slice, msk_slice, medsam, device, min_area=min_area)
            if out is None:
                st.warning("Keine Komponenten gefunden (oder Maske leer). Slice wechseln oder min_area senken.")
                # Aufräumen
                try:
                    os.remove(img_path); os.remove(msk_path)
                except Exception:
                    pass
                gc.collect()
                return

            resized_img, boxes_1024, seg_combined, seg_per_box = out

            # Anzeige
            fig1, ax1 = plt.subplots(figsize=(6,6))
            ax1.imshow(resized_img, cmap="gray")
            for b in boxes_1024: show_box(b, ax1)
            ax1.set_title(f"Original image with {len(boxes_1024)} box(es)")
            ax1.axis("off")

            fig2, ax2 = plt.subplots(figsize=(6,6))
            ax2.imshow(resized_img, cmap="gray")
            show_mask(seg_combined, ax2)
            ax2.set_title("Segmentation")
            ax2.axis("off")

            c1, c2 = st.columns(2)
            with c1: st.pyplot(fig1, clear_figure=True)
            with c2: st.pyplot(fig2, clear_figure=True)

            # Download (Overlay)
            os.makedirs("output", exist_ok=True)
            overlay_fig, overlay_ax = plt.subplots(figsize=(6,6))
            overlay_ax.imshow(resized_img, cmap="gray")
            show_mask(seg_combined, overlay_ax)
            overlay_ax.axis("off")
            overlay_fig.tight_layout(pad=0)
            buf = io.BytesIO()
            overlay_fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
            plt.close(overlay_fig)
            st.download_button("Download Segmented Overlay (PNG)",
                               data=buf.getvalue(),
                               file_name=f"seg_multi_{slice_idx}.png",
                               mime="image/png")

        # Aufräumen
        try:
            os.remove(img_path); os.remove(msk_path)
        except Exception:
            pass
        gc.collect()

    else:
        st.info("Please upload image + mask (NIfTI).")

if __name__ == "__main__":
    main()
