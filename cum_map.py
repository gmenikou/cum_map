import io
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import streamlit as st
from PIL import Image


st.set_page_config(page_title="Cumulative Dose Map GUI", layout="wide")
st.title("Cumulative Dose Map GUI")

st.write(
    "Upload OpenREM skin dose map PNG images. For each image, enter the displayed maximum dose from the grayscale scale. "
    "The app crops the dose-map panel, converts grayscale display values back into approximate dose values, "
    "shows each reconstructed map in colour with a Gy colourbar, and builds a cumulative map."
)

st.warning(
    "This reconstructs dose from the displayed 2D grayscale map, not from raw openSkin/OpenREM numeric data. "
    "Use for visualisation, QA, and exploratory review. Do not treat it as validated clinical dosimetry."
)


# -----------------------------
# Helpers
# -----------------------------
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
    return buf.getvalue()


def render_interactive_dose_map(dose_map, title, vmax=None, roi_used=None, roi_list=None, peak_xy=None):
    use_vmax = vmax if vmax is not None and vmax > 0 else max(float(np.max(dose_map)), 1e-6)

    fig = px.imshow(
        dose_map,
        color_continuous_scale="Jet",
        zmin=0,
        zmax=use_vmax,
        aspect="equal",
        origin="upper",
        labels={"x": "X", "y": "Y", "color": "Dose (Gy)"},
    )

    fig.update_traces(
        hovertemplate="X: %{x}<br>Y: %{y}<br>Dose: %{z:.3f} Gy<extra></extra>"
    )

    fig.update_layout(
        title=title,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="X",
        yaxis_title="Y",
    )

    if roi_used is not None:
        fig.add_shape(
            type="rect",
            x0=roi_used[0], y0=roi_used[1], x1=roi_used[2], y1=roi_used[3],
            line=dict(color="white", width=2),
            fillcolor="rgba(0,0,0,0)"
        )
        fig.add_annotation(
            x=roi_used[0],
            y=roi_used[1],
            text=f"ROI {roi_used[2]-roi_used[0]}×{roi_used[3]-roi_used[1]} px",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font=dict(color="white"),
            bgcolor="rgba(0,0,0,0.45)"
        )

    if roi_list:
        for i, item in enumerate(roi_list, start=1):
            if isinstance(item, dict):
                roi = item.get("roi")
                label = item.get("label", f"R{i}")
            else:
                roi = item
                label = f"R{i}"

            if roi is None or len(roi) != 4:
                continue

            fig.add_shape(
                type="rect",
                x0=roi[0], y0=roi[1], x1=roi[2], y1=roi[3],
                line=dict(color="cyan", width=1),
                fillcolor="rgba(0,255,255,0.0)"
            )
            fig.add_annotation(
                x=roi[0],
                y=roi[1],
                text=label,
                showarrow=False,
                xanchor="left",
                yanchor="bottom",
                font=dict(color="cyan"),
                bgcolor="rgba(0,0,0,0.35)"
            )

    if peak_xy is not None:
        fig.add_scatter(
            x=[peak_xy[0]],
            y=[peak_xy[1]],
            mode="markers+text",
            text=["Peak"],
            textposition="top center",
            marker=dict(size=10, symbol="x", color="white"),
            name="Peak",
            hovertemplate="Peak X: %{x}<br>Peak Y: %{y}<extra></extra>"
        )

    return fig


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
    h, w = gray.shape

    panel_mask = gray < 248
    rows = np.where(panel_mask.sum(axis=1) > max(20, w * 0.10))[0]
    cols = np.where(panel_mask.sum(axis=0) > max(20, h * 0.10))[0]

    if len(rows) == 0 or len(cols) == 0:
        return 0, 0, w, h

    y1 = max(0, int(rows[0]))
    y2 = min(h, int(rows[-1] + 1))
    x1 = max(0, int(cols[0]))
    x2 = min(w, int(cols[-1] + 1))

    crop = gray[y1:y2, x1:x2]
    _, cw = crop.shape

    right_start = int(cw * 0.78)
    if right_start < cw - 5:
        region = crop[:, right_start:]
        darkness = 255.0 - region.mean(axis=0)
        if np.max(darkness) > 25:
            idx = int(np.argmax(darkness))
            bar_x = right_start + idx
            x2 = min(w, x1 + max(bar_x - 10, int(cw * 0.70)))

    pad_x = max(0, int((x2 - x1) * 0.01))
    pad_y = max(0, int((y2 - y1) * 0.01))
    x1 = min(max(x1 + pad_x, 0), w)
    x2 = min(max(x2 - pad_x, x1 + 1), w)
    y1 = min(max(y1 + pad_y, 0), h)
    y2 = min(max(y2 - pad_y, y1 + 1), h)

    return x1, y1, x2, y2


def reconstruct_displayed_dose(gray_crop, displayed_max_dose, mask_threshold=245):
    arr = gray_crop.astype(np.float32)
    dose = (1.0 - arr / 255.0) * float(displayed_max_dose)
    dose[arr >= mask_threshold] = 0.0
    dose = np.clip(dose, 0.0, None)
    return dose


def clamp_roi(roi, width, height):
    x1, y1, x2, y2 = roi
    x1 = max(0, min(int(x1), width - 1))
    x2 = max(x1 + 1, min(int(x2), width))
    y1 = max(0, min(int(y1), height - 1))
    y2 = max(y1 + 1, min(int(y2), height))
    return (x1, y1, x2, y2)


def roi_stats(cumulative_dose, roi=None):
    h, w = cumulative_dose.shape

    if roi is not None:
        roi = clamp_roi(roi, w, h)
        x1, y1, x2, y2 = roi
        stats_map = cumulative_dose[y1:y2, x1:x2]
    else:
        x1, y1, x2, y2 = 0, 0, w, h
        stats_map = cumulative_dose
        roi = None

    if stats_map.size == 0:
        peak_val = 0.0
        peak_x, peak_y = x1, y1
    else:
        peak_val = float(np.max(stats_map))
        peak_idx_local = np.unravel_index(np.argmax(stats_map), stats_map.shape)
        peak_x = int(peak_idx_local[1] + x1)
        peak_y = int(peak_idx_local[0] + y1)

    nonzero = stats_map[stats_map > 0]
    mean_dose = float(np.mean(nonzero)) if nonzero.size else 0.0
    median_dose = float(np.median(nonzero)) if nonzero.size else 0.0
    std_dose = float(np.std(nonzero)) if nonzero.size else 0.0
    min_nonzero = float(np.min(nonzero)) if nonzero.size else 0.0
    p95 = float(np.percentile(nonzero, 95)) if nonzero.size else 0.0
    dose_sum = float(np.sum(stats_map)) if stats_map.size else 0.0
    nonzero_pixels = int(np.count_nonzero(stats_map > 0))
    thr2 = int(np.sum(stats_map >= 2.0))
    thr5 = int(np.sum(stats_map >= 5.0))
    thr10 = int(np.sum(stats_map >= 10.0))

    return {
        "roi": roi,
        "peak_cumulative_dose": peak_val,
        "peak_x": peak_x,
        "peak_y": peak_y,
        "mean_dose": mean_dose,
        "median_dose": median_dose,
        "std_dose": std_dose,
        "min_nonzero_dose": min_nonzero,
        "p95_dose": p95,
        "dose_sum": dose_sum,
        "nonzero_pixels": nonzero_pixels,
        "thr2": thr2,
        "thr5": thr5,
        "thr10": thr10,
        "stats_map": stats_map,
    }


def build_csv_bytes(session_records, overall_stats, multi_roi_stats):
    lines = []
    lines.append("file,crop_box,cropped_shape,displayed_max_gy,reconstructed_peak_gy,nonzero_pixels")

    for row in session_records:
        lines.append(
            f"{row['file']},{row['crop_box']},{row['cropped_shape']},{row['displayed_max_gy']},{row['reconstructed_peak_gy']},{row['nonzero_pixels']}"
        )

    lines.append("")
    lines.append("cumulative_statistic,value")
    lines.append(f"stats_roi,{overall_stats['roi'] if overall_stats['roi'] is not None else 'full_map'}")
    lines.append(f"peak_cumulative_dose_gy,{overall_stats['peak_cumulative_dose']}")
    lines.append(f"peak_location_x,{overall_stats['peak_x']}")
    lines.append(f"peak_location_y,{overall_stats['peak_y']}")
    lines.append(f"mean_dose_gy,{overall_stats['mean_dose']}")
    lines.append(f"median_dose_gy,{overall_stats['median_dose']}")
    lines.append(f"std_dose_gy,{overall_stats['std_dose']}")
    lines.append(f"minimum_nonzero_dose_gy,{overall_stats['min_nonzero_dose']}")
    lines.append(f"dose_p95_gy,{overall_stats['p95_dose']}")
    lines.append(f"dose_sum,{overall_stats['dose_sum']}")
    lines.append(f"nonzero_pixels,{overall_stats['nonzero_pixels']}")
    lines.append(f"pixels_ge_2gy,{overall_stats['thr2']}")
    lines.append(f"pixels_ge_5gy,{overall_stats['thr5']}")
    lines.append(f"pixels_ge_10gy,{overall_stats['thr10']}")

    if multi_roi_stats:
        lines.append("")
        lines.append(
            "roi_label,roi,peak_cumulative_dose_gy,peak_x,peak_y,mean_dose_gy,median_dose_gy,std_dose_gy,min_nonzero_dose_gy,p95_dose_gy,dose_sum,nonzero_pixels,pixels_ge_2gy,pixels_ge_5gy,pixels_ge_10gy"
        )
        for label, stats in multi_roi_stats:
            lines.append(
                f"{label},{stats['roi']},{stats['peak_cumulative_dose']},{stats['peak_x']},{stats['peak_y']},{stats['mean_dose']},{stats['median_dose']},{stats['std_dose']},{stats['min_nonzero_dose']},{stats['p95_dose']},{stats['dose_sum']},{stats['nonzero_pixels']},{stats['thr2']},{stats['thr5']},{stats['thr10']}"
            )

    return "\n".join(lines).encode("utf-8")


# -----------------------------
# Session state
# -----------------------------
if "roi_list" not in st.session_state:
    st.session_state.roi_list = []
else:
    cleaned = []
    for item in st.session_state.roi_list:
        if isinstance(item, dict) and "roi" in item:
            cleaned.append({"label": item.get("label", f"ROI {len(cleaned)+1}"), "roi": tuple(item["roi"])})
        elif isinstance(item, (list, tuple)) and len(item) == 4:
            cleaned.append({"label": f"ROI {len(cleaned)+1}", "roi": tuple(item)})
    st.session_state.roi_list = cleaned


# -----------------------------
# UI
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload OpenREM PNG images",
    type=["png"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.subheader("Reconstruction settings")

    stats_from_roi_only = st.checkbox(
        "Calculate statistics only inside a manually selected analysis area",
        value=True,
        help="Use this to exclude annotations and calculate cumulative-map statistics only inside the real map area.",
    )
    st.caption("The analysis ROI is a free rectangle defined by independent X1, Y1, X2, Y2 coordinates. There is no fixed aspect ratio.")

    crop_mode = st.selectbox(
        "Map area selection",
        ["Auto-detect map panel", "Use full image", "Manual crop for all images"],
        index=0,
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

    use_global_scale = st.checkbox("Use the same colour scale for all individual maps", value=True)
    mask_threshold = st.slider("Background threshold", min_value=220, max_value=255, value=245, step=1)

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
        manual_vals = (int(x1), int(y1), int(x2), int(y2))

    analysis_roi = None
    if stats_from_roi_only:
        st.subheader("Manual analysis area for statistics")
        st.write("Set a free rectangular ROI using independent corner coordinates. The box is not constrained to any fixed aspect ratio.")
        a1, a2 = st.columns(2)
        with a1:
            roi_x1 = st.number_input("ROI x1 (left)", min_value=0, max_value=W - 1, value=int(W * 0.12), step=1)
            roi_y1 = st.number_input("ROI y1 (top)", min_value=0, max_value=H - 1, value=int(H * 0.18), step=1)
        with a2:
            roi_x2 = st.number_input("ROI x2 (right)", min_value=1, max_value=W, value=int(W * 0.70), step=1)
            roi_y2 = st.number_input("ROI y2 (bottom)", min_value=1, max_value=H, value=int(H * 0.80), step=1)
        analysis_roi = (int(roi_x1), int(roi_y1), int(roi_x2), int(roi_y2))
        st.write(f"ROI size: width = {analysis_roi[2] - analysis_roi[0]} px, height = {analysis_roi[3] - analysis_roi[1]} px")
        st.caption("Changing ROI values will automatically refresh the statistics and ROI overlay.")

    st.subheader("Displayed maximum dose for each image")
    st.write("Enter the value shown at the top of the grayscale dose scale for each PNG.")

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

    build_clicked = st.button("Build reconstructed cumulative map", type="primary")

    if build_clicked or "cumulative_dose" not in st.session_state:
        try:
            preview_records = []
            reconstructed = []

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
                displayed_max = float(displayed_max_inputs[file.name])
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

            st.session_state.processed = processed
            st.session_state.preview_records = preview_records
            st.session_state.cumulative_dose = cumulative_dose
            st.session_state.session_vmax = max(p[2] for p in processed) if use_global_scale else None
            st.session_state.use_global_scale = use_global_scale

        except Exception as e:
            st.error(f"Error: {e}")

    if "cumulative_dose" in st.session_state:
        cumulative_dose = st.session_state.cumulative_dose
        processed = st.session_state.processed
        preview_records = st.session_state.preview_records
        session_vmax = st.session_state.session_vmax if st.session_state.use_global_scale else None

        roi_used = clamp_roi(analysis_roi, cumulative_dose.shape[1], cumulative_dose.shape[0]) if (stats_from_roi_only and analysis_roi is not None) else None
        overall_stats = roi_stats(cumulative_dose, roi_used)

        st.success(f"Cumulative reconstructed map ready. Final size: {cumulative_dose.shape}")
        if roi_used is not None:
            st.info(f"Statistics calculated only inside ROI: x={roi_used[0]} to {roi_used[2]}, y={roi_used[1]} to {roi_used[3]}")

        st.subheader("Crop preview and reconstruction summary")
        st.dataframe(preview_records, use_container_width=True)

        st.subheader("Individual reconstructed session maps")
        session_cols = st.columns(2)
        session_pngs = []

        for i, (name, dose_map, displayed_max, bbox, crop) in enumerate(processed):
            with session_cols[i % 2]:
                st.markdown(f"**{name}**")
                st.caption(f"Crop box: {bbox}")
                plotly_fig = render_interactive_dose_map(
                    dose_map,
                    title=f"{name} | reconstructed peak {np.max(dose_map):.3f} Gy",
                    vmax=session_vmax if session_vmax is not None else displayed_max,
                )
                st.plotly_chart(plotly_fig, use_container_width=True)

                fig = render_dose_figure(
                    dose_map,
                    title=f"{name} | reconstructed peak {np.max(dose_map):.3f} Gy",
                    vmax=session_vmax if session_vmax is not None else displayed_max,
                )
                session_pngs.append((name.replace('.png', '_reconstructed.png'), fig_to_png_bytes(fig)))
                plt.close(fig)

        st.subheader("Cumulative reconstructed dose map")
        c1, c2 = st.columns([1.4, 1])

        with c1:
            plotly_cum = render_interactive_dose_map(
                cumulative_dose,
                title=f"Cumulative reconstructed dose | peak {overall_stats['peak_cumulative_dose']:.3f} Gy",
                vmax=max(float(np.max(cumulative_dose)), 1.0),
                roi_used=roi_used,
                roi_list=st.session_state.roi_list,
                peak_xy=(overall_stats['peak_x'], overall_stats['peak_y'])
            )
            st.plotly_chart(plotly_cum, use_container_width=True)

            fig_cum = render_dose_figure(
                cumulative_dose,
                title=f"Cumulative reconstructed dose | peak {overall_stats['peak_cumulative_dose']:.3f} Gy",
                vmax=max(float(np.max(cumulative_dose)), 1.0),
            )
            cum_png = fig_to_png_bytes(fig_cum)
            plt.close(fig_cum)

        with c2:
            st.metric("Number of sessions", len(processed))
            st.metric("Peak cumulative dose", f"{overall_stats['peak_cumulative_dose']:.3f} Gy")
            st.metric("Peak location (X, Y)", f"({overall_stats['peak_x']}, {overall_stats['peak_y']})")
            st.caption("Coordinates are reported as cursor-style X,Y positions, where X is horizontal and Y is vertical.")
            st.write("Cumulative map statistics")
            st.write(f"Mean dose: {overall_stats['mean_dose']:.3f} Gy")
            st.write(f"Median dose: {overall_stats['median_dose']:.3f} Gy")
            st.write(f"Std deviation: {overall_stats['std_dose']:.3f} Gy")
            st.write(f"Minimum non-zero dose: {overall_stats['min_nonzero_dose']:.3f} Gy")
            st.write(f"95th percentile: {overall_stats['p95_dose']:.3f} Gy")
            st.write(f"Dose sum over selected area: {overall_stats['dose_sum']:.3f}")
            st.write(f"Non-zero pixels: {overall_stats['nonzero_pixels']}")
            st.write("Pixels above thresholds")
            st.write(f"≥ 2 Gy: {overall_stats['thr2']}")
            st.write(f"≥ 5 Gy: {overall_stats['thr5']}")
            st.write(f"≥ 10 Gy: {overall_stats['thr10']}")

        st.subheader("Dose histogram inside selected area")
        nz = overall_stats["stats_map"][overall_stats["stats_map"] > 0]
        if nz.size > 0:
            fig_hist, ax_hist = plt.subplots(figsize=(7, 4))
            ax_hist.hist(nz.ravel(), bins=40)
            ax_hist.set_xlabel("Dose (Gy)")
            ax_hist.set_ylabel("Pixel count")
            ax_hist.set_title("Histogram of non-zero dose values in selected area")
            st.pyplot(fig_hist)
            plt.close(fig_hist)
        else:
            st.info("No non-zero pixels in the selected area, so the histogram is empty.")

        st.subheader("Additional ROIs")
        st.write("You can save the current free ROI as an additional region and compare multiple regions.")
        r1, r2 = st.columns([1, 1])

        with r1:
            roi_label = st.text_input("Label for current ROI", value=f"ROI {len(st.session_state.roi_list) + 1}")
        with r2:
            add_roi_clicked = st.button("Add current ROI to comparison table")

        if add_roi_clicked and roi_used is not None:
            st.session_state.roi_list.append({
                "label": roi_label,
                "roi": tuple(roi_used),
            })

        if st.button("Clear saved ROIs"):
            st.session_state.roi_list = []

        multi_roi_stats = []
        if st.session_state.roi_list:
            for item in st.session_state.roi_list:
                stats = roi_stats(cumulative_dose, item["roi"])
                multi_roi_stats.append((item["label"], stats))

            roi_rows = []
            for label, stats in multi_roi_stats:
                roi_rows.append(
                    {
                        "label": label,
                        "roi": str(stats["roi"]),
                        "peak_gy": round(stats["peak_cumulative_dose"], 3),
                        "peak_xy": f"({stats['peak_x']}, {stats['peak_y']})",
                        "mean_gy": round(stats["mean_dose"], 3),
                        "median_gy": round(stats["median_dose"], 3),
                        "p95_gy": round(stats["p95_dose"], 3),
                        "dose_sum": round(stats["dose_sum"], 3),
                        "nonzero_pixels": stats["nonzero_pixels"],
                        ">=2Gy": stats["thr2"],
                        ">=5Gy": stats["thr5"],
                        ">=10Gy": stats["thr10"],
                    }
                )
            st.dataframe(roi_rows, use_container_width=True)

        st.subheader("Hotspot helper")
        st.write("The main hotspot is marked on the cumulative map. Use the peak X,Y values to position a tighter ROI around it if you want local statistics.")

        csv_bytes = build_csv_bytes(preview_records, overall_stats, multi_roi_stats)

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
else:
    st.caption("Upload one or more OpenREM PNG images to begin.")
