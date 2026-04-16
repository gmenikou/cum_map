import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import zipfile

st.set_page_config(page_title="Cumulative Dose Map GUI", layout="wide")
st.title("Cumulative Dose Map GUI")

st.write(
    "Upload OpenREM skin dose map PNG images. For each image, enter the displayed maximum dose from the grayscale scale. "
    "The app crops the dose-map panel, converts grayscale display values back into approximate dose values, "
    "shows each reconstructed map in colour with a Gy colourbar, and builds a cumulative map."
)

st.warning(
    "This reconstructs dose from the DISPLAYED 2D grayscale map, not from raw openSkin/OpenREM numeric data. "
    "Use for visualization, QA, and exploratory review. Do not treat it as validated clinical dosimetry."
)

st.info(
    "Recommended for OpenREM screenshots like the one you showed: the map panel is grayscale, white is near 0 Gy, black is near the displayed maximum Gy, and the right-side colourbar is excluded from reconstruction."
)


def render_dose_figure(dose_map, title, vmax=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    use_vmax = vmax if vmax is not None and vmax > 0 else max(float(np.max(dose_map)), 1e-6)
    im = ax.imshow(dose_map, cmap="jet", vmin=0, vmax=use_vmax)
    ax.set_title(title)
    ax.axis("off")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Dose (Gy)")
    fig.tight_layout()
    return fig


def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf


def center_crop(arr, target_h, target_w):
    h, w = arr.shape
    sy = max((h - target_h) // 2, 0)
    sx = max((w - target_w) // 2, 0)
    return arr[sy:sy + target_h, sx:sx + target_w]


def center_pad(arr, target_h, target_w, pad_value=0.0):
    h, w = arr.shape
    out = np.full((target_h, target_w), pad_value, dtype=arr.dtype)
    sy = max((target_h - h) // 2, 0)
    sx = max((target_w - w) // 2, 0)
    out[sy:sy + h, sx:sx + w] = arr
    return out


def auto_find_map_bbox(gray):
    """
    Heuristic for OpenREM-like screenshots.
    Finds the large light-gray map panel and excludes the right colorbar when possible.
    Returns x1, y1, x2, y2 in image coordinates.
    """
    h, w = gray.shape

    panel_mask = gray < 248  # includes borders, text, map content, colorbar
    rows = np.where(panel_mask.sum(axis=1) > max(20, w * 0.10))[0]
    cols = np.where(panel_mask.sum(axis=0) > max(20, h * 0.10))[0]

    if len(rows) == 0 or len(cols) == 0:
        return 0, 0, w, h

    y1 = max(0, int(rows[0]))
    y2 = min(h, int(rows[-1] + 1))
    x1 = max(0, int(cols[0]))
    x2 = min(w, int(cols[-1] + 1))

    crop = gray[y1:y2, x1:x2]
    ch, cw = crop.shape

    # Try to detect the vertical colorbar on the far right as a dark narrow stripe.
    right_start = int(cw * 0.78)
    if right_start < cw - 5:
        region = crop[:, right_start:]
        darkness = 255.0 - region.mean(axis=0)
        if np.max(darkness) > 25:
            idx = int(np.argmax(darkness))
            bar_x = right_start + idx
            # exclude stripe plus label area to the right
            x2 = min(w, x1 + max(bar_x - 10, int(cw * 0.70)))

    # Trim a small border margin if present
    pad_x = max(0, int((x2 - x1) * 0.01))
    pad_y = max(0, int((y2 - y1) * 0.01))
    x1 = min(max(x1 + pad_x, 0), w)
    x2 = min(max(x2 - pad_x, x1 + 1), w)
    y1 = min(max(y1 + pad_y, 0), h)
    y2 = min(max(y2 - pad_y, y1 + 1), h)

    return x1, y1, x2, y2


def reconstruct_displayed_dose(gray_crop, displayed_max_dose, mask_threshold=245):
    """
    Convert grayscale display image into approximate dose.
    White/light gray background is treated as 0 Gy.
    Darker pixels map toward displayed_max_dose.
    """
    arr = gray_crop.astype(np.float32)
    dose = (1.0 - arr / 255.0) * float(displayed_max_dose)
    dose[arr >= mask_threshold] = 0.0
    dose = np.clip(dose, 0.0, None)
    return dose


uploaded_files = st.file_uploader(
    "Upload OpenREM PNG images",
    type=["png"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.subheader("Reconstruction settings")

    crop_mode = st.selectbox(
        "Map area selection",
        [
            "Auto-detect map panel",
            "Use full image",
            "Manual crop for all images",
        ],
        index=0,
        help="Auto-detect works best for OpenREM screenshots with a large map panel and a right-side colorbar.",
    )

    size_mode = st.selectbox(
        "How should mismatched reconstructed sizes be handled?",
        [
            "Strict (all reconstructed maps must match exactly)",
            "Crop to smallest common size",
            "Pad to largest common size",
        ],
        index=1,
    )

    use_global_scale = st.checkbox(
        "Use the same colour scale for all individual maps",
        value=True,
        help="When enabled, all reconstructed session maps use the same Gy range for easier comparison.",
    )

    mask_threshold = st.slider(
        "Background threshold",
        min_value=220,
        max_value=255,
        value=245,
        step=1,
        help="Pixels brighter than or equal to this are treated as background and set to 0 Gy.",
    )

    sample_img = Image.open(uploaded_files[0]).convert("L")
    sample_arr = np.array(sample_img)
    H, W = sample_arr.shape

    manual_vals = None
    if crop_mode == "Manual crop for all images":
        st.write("Manual crop is applied to all uploaded images using the same pixel coordinates.")
        c1, c2 = st.columns(2)
        with c1:
            x1 = st.number_input("x1", min_value=0, max_value=W - 1, value=int(W * 0.08), step=1)
            y1 = st.number_input("y1", min_value=0, max_value=H - 1, value=int(H * 0.12), step=1)
        with c2:
            x2 = st.number_input("x2", min_value=1, max_value=W, value=int(W * 0.80), step=1)
            y2 = st.number_input("y2", min_value=1, max_value=H, value=int(H * 0.84), step=1)
        if x2 <= x1 or y2 <= y1:
            st.error("Manual crop coordinates must satisfy x2 > x1 and y2 > y1.")
        manual_vals = (int(x1), int(y1), int(x2), int(y2))

    st.subheader("Displayed maximum dose for each image")
    st.write("Enter the value shown at the TOP of the grayscale dose scale for each PNG.")

    displayed_max_inputs = {}
    input_cols = st.columns(2)
    for i, file in enumerate(uploaded_files):
        with input_cols[i % 2]:
            displayed_max_inputs[file.name] = st.number_input(
                f"Displayed max dose for {file.name} (Gy)",
                min_value=0.0,
                value=1.0,
                step=0.001,
                format="%.3f",
                key=f"maxdose_{file.name}",
            )

    if st.button("Build reconstructed cumulative map", type="primary"):
        try:
            preview_records = []
            reconstructed = []
            raw_shapes = []

            for file in uploaded_files:
                img_gray = Image.open(file).convert("L")
                gray = np.array(img_gray, dtype=np.uint8)

                if crop_mode == "Auto-detect map panel":
                    x1, y1, x2, y2 = auto_find_map_bbox(gray)
                elif crop_mode == "Use full image":
                    x1, y1, x2, y2 = 0, 0, gray.shape[1], gray.shape[0]
                else:
                    x1, y1, x2, y2 = manual_vals

                crop = gray[y1:y2, x1:x2]
                raw_shapes.append(crop.shape)

                displayed_max = float(displayed_max_inputs[file.name])
                if displayed_max <= 0:
                    raise ValueError(f"Displayed max dose for {file.name} must be greater than 0.")

                dose_map = reconstruct_displayed_dose(crop, displayed_max, mask_threshold=mask_threshold)
                reconstructed.append((file.name, dose_map, displayed_max, (x1, y1, x2, y2), crop))

                preview_records.append(
                    {
                        "file": file.name,
                        "crop_box": f"({x1}, {y1}) to ({x2}, {y2})",
                        "cropped_shape": str(crop.shape),
                        "displayed_max_gy": displayed_max,
                        "reconstructed_peak_gy": float(np.max(dose_map)),
                        "nonzero_pixels": int(np.count_nonzero(dose_map > 0)),
                    }
                )

            shapes = [r[1].shape for r in reconstructed]
            heights = [s[0] for s in shapes]
            widths = [s[1] for s in shapes]

            if size_mode == "Strict (all reconstructed maps must match exactly)":
                if len(set(shapes)) != 1:
                    raise ValueError(f"Reconstructed map size mismatch: found shapes {sorted(set(shapes))}")
                target_h, target_w = shapes[0]
            elif size_mode == "Crop to smallest common size":
                target_h, target_w = min(heights), min(widths)
            else:
                target_h, target_w = max(heights), max(widths)

            processed = []
            for name, dose_map, displayed_max, bbox, crop in reconstructed:
                if size_mode == "Crop to smallest common size":
                    dose_proc = center_crop(dose_map, target_h, target_w)
                elif size_mode == "Pad to largest common size":
                    dose_proc = center_pad(dose_map, target_h, target_w, pad_value=0.0)
                else:
                    dose_proc = dose_map
                processed.append((name, dose_proc, displayed_max, bbox, crop))

            cumulative_dose = np.sum([p[1] for p in processed], axis=0)
            peak_cumulative_dose = float(np.max(cumulative_dose))
            peak_idx = np.unravel_index(np.argmax(cumulative_dose), cumulative_dose.shape)

            thr2 = int(np.sum(cumulative_dose >= 2.0))
            thr5 = int(np.sum(cumulative_dose >= 5.0))
            thr10 = int(np.sum(cumulative_dose >= 10.0))

            st.success(f"Cumulative reconstructed map created. Final size: {cumulative_dose.shape}")

            st.subheader("Crop preview and reconstruction summary")
            st.dataframe(preview_records, use_container_width=True)

            st.subheader("Individual reconstructed session maps")
            session_vmax = max(p[2] for p in processed) if use_global_scale else None
            session_cols = st.columns(2)
            session_pngs = []

            for i, (name, dose_map, displayed_max, bbox, crop) in enumerate(processed):
                with session_cols[i % 2]:
                    st.markdown(f"**{name}**")
                    st.caption(f"Crop box: {bbox}")
                    fig = render_dose_figure(
                        dose_map,
                        title=f"{name} | reconstructed peak {np.max(dose_map):.3f} Gy",
                        vmax=session_vmax if use_global_scale else displayed_max,
                    )
                    st.pyplot(fig)
                    session_pngs.append((name.replace('.png', '_reconstructed.png'), fig_to_png_bytes(fig).getvalue()))
                    plt.close(fig)

            st.subheader("Cumulative reconstructed dose map")
            c1, c2 = st.columns([1.3, 1])
            with c1:
                fig_cum = render_dose_figure(
                    cumulative_dose,
                    title=f"Cumulative reconstructed dose | peak {peak_cumulative_dose:.3f} Gy",
                    vmax=peak_cumulative_dose if peak_cumulative_dose > 0 else 1.0,
                )
                st.pyplot(fig_cum)
                cum_png = fig_to_png_bytes(fig_cum).getvalue()
                plt.close(fig_cum)

            with c2:
                st.metric("Number of sessions", len(processed))
                st.metric("Peak cumulative dose", f"{peak_cumulative_dose:.3f} Gy")
                st.metric("Peak location (row, col)", str(peak_idx))
                st.write("Pixels above thresholds")
                st.write(f"≥ 2 Gy: {thr2}")
                st.write(f"≥ 5 Gy: {thr5}")
                st.write(f"≥ 10 Gy: {thr10}")

            csv_lines = [
                "file,crop_box,cropped_shape,displayed_max_gy,reconstructed_peak_gy,nonzero_pixels"
            ]
            for row in preview_records:
                csv_lines.append(
                    f"{row['file']},{row['crop_box']},{row['cropped_shape']},{row['displayed_max_gy']},{row['reconstructed_peak_gy']},{row['nonzero_pixels']}"
                )
            csv_lines.append(f"TOTAL,,,,{peak_cumulative_dose},")
            csv_bytes = "\n".join(csv_lines).encode("utf-8")

            npy_buf = io.BytesIO()
            np.save(npy_buf, cumulative_dose)
            npy_buf.seek(0)

            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("cumulative_reconstructed_dose.npy", npy_buf.getvalue())
                zf.writestr("cumulative_reconstructed_dose.png", cum_png)
                zf.writestr("reconstruction_summary.csv", csv_bytes)
                for fname, content in session_pngs:
                    zf.writestr(fname, content)
            zip_buf.seek(0)

            d1, d2 = st.columns(2)
            with d1:
                st.download_button(
                    "Download cumulative array (.npy)",
                    data=npy_buf.getvalue(),
                    file_name="cumulative_reconstructed_dose.npy",
                    mime="application/octet-stream",
                )
            with d2:
                st.download_button(
                    "Download all results (.zip)",
                    data=zip_buf.getvalue(),
                    file_name="reconstructed_dose_results.zip",
                    mime="application/zip",
                )

        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.caption("Upload one or more OpenREM PNG images to begin.")
