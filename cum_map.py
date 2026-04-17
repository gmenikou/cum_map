import io
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image


st.set_page_config(page_title="Cumulative Dose Map", layout="wide")
st.title("Cumulative reconstructed dose map")

st.write(
    "Upload OpenREM PNG dose maps, enter the displayed maximum dose for each session, "
    "build the cumulative reconstructed map, then draw one inclusion ROI with the mouse "
    "using box select on the cumulative map."
)

st.warning(
    "This reconstructs dose from the displayed grayscale map, not from raw OpenREM/openSkin numeric data. "
    "Use for visualisation, QA, and exploratory analysis."
)


# =========================================================
# Helpers
# =========================================================
def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def render_matplotlib_map(dose_map, title, vmax=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    use_vmax = vmax if vmax is not None and vmax > 0 else max(float(np.max(dose_map)), 1e-6)
    im = ax.imshow(dose_map, cmap="jet", vmin=0, vmax=use_vmax)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Dose (Gy)")
    fig.tight_layout()
    return fig


def adaptive_grid_step(width, height, target_points=12000):
    area = max(width * height, 1)
    step = int(np.ceil(np.sqrt(area / target_points)))
    return max(step, 2)


def build_selectable_grid(width, height, step=None):
    if step is None:
        step = adaptive_grid_step(width, height)
    xs = np.arange(0, width, step, dtype=int)
    ys = np.arange(0, height, step, dtype=int)
    xx, yy = np.meshgrid(xs, ys)
    return xx.ravel(), yy.ravel(), step


def render_interactive_map(
    dose_map,
    title,
    vmax=None,
    inclusion_roi=None,
    peak_xy=None,
    selectable=True,
):
    use_vmax = vmax if vmax is not None and vmax > 0 else max(float(np.max(dose_map)), 1e-6)
    h, w = dose_map.shape

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=dose_map,
            colorscale="Jet",
            zmin=0,
            zmax=use_vmax,
            colorbar=dict(title="Dose (Gy)"),
            hovertemplate="X: %{x}<br>Y: %{y}<br>Dose: %{z:.3f} Gy<extra></extra>",
        )
    )

    selection_step = None
    if selectable:
        gx, gy, selection_step = build_selectable_grid(w, h)
        fig.add_trace(
            go.Scatter(
                x=gx,
                y=gy,
                mode="markers",
                marker=dict(size=6, color="rgba(0,0,0,0)"),
                hoverinfo="skip",
                showlegend=False,
                selected=dict(marker=dict(opacity=0)),
                unselected=dict(marker=dict(opacity=0)),
            )
        )

    if inclusion_roi is not None:
        x1, y1, x2, y2 = inclusion_roi
        fig.add_shape(
            type="rect",
            x0=x1, y0=y1, x1=x2, y1=y2,
            line=dict(color="white", width=2),
            fillcolor="rgba(0,0,0,0)"
        )
        fig.add_annotation(
            x=x1,
            y=y1,
            text=f"ROI {x2-x1}×{y2-y1} px",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font=dict(color="white"),
            bgcolor="rgba(0,0,0,0.45)"
        )

    if peak_xy is not None:
        fig.add_trace(
            go.Scatter(
                x=[peak_xy[0]],
                y=[peak_xy[1]],
                mode="markers+text",
                text=["Peak"],
                textposition="top center",
                marker=dict(size=10, symbol="x", color="white"),
                name="Peak",
                hovertemplate="Peak X: %{x}<br>Peak Y: %{y}<extra></extra>",
                showlegend=False,
            )
        )

    fig.update_layout(
        title=title,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="X",
        yaxis_title="Y",
        dragmode="select",
    )
    fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1)

    return fig, selection_step


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


def reconstruct_displayed_dose(gray_crop, displayed_max_dose):
    arr = gray_crop.astype(np.float32)
    dose = (1.0 - arr / 255.0) * float(displayed_max_dose)
    dose[arr >= 245] = 0.0
    dose = np.clip(dose, 0.0, None)
    return dose


def clamp_roi(roi, width, height):
    x1, y1, x2, y2 = roi
    x1 = max(0, min(int(x1), width - 1))
    x2 = max(x1 + 1, min(int(x2), width))
    y1 = max(0, min(int(y1), height - 1))
    y2 = max(y1 + 1, min(int(y2), height))
    return (x1, y1, x2, y2)


def masked_to_inclusion_roi(arr, roi):
    if roi is None:
        return arr
    x1, y1, x2, y2 = roi
    out = np.zeros_like(arr)
    out[y1:y2, x1:x2] = arr[y1:y2, x1:x2]
    return out


def compute_stats(dose_map, roi=None):
    h, w = dose_map.shape

    if roi is None:
        roi = (0, 0, w, h)

    roi = clamp_roi(roi, w, h)
    x1, y1, x2, y2 = roi
    stats_map = dose_map[y1:y2, x1:x2]

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

    total_pixels = int(stats_map.size)
    nonzero_pixels = int(np.count_nonzero(stats_map > 0))

    thr1 = int(np.sum(stats_map >= 1.0))
    thr2 = int(np.sum(stats_map >= 2.0))
    thr5 = int(np.sum(stats_map >= 5.0))
    thr10 = int(np.sum(stats_map >= 10.0))

    def pct(count, denom):
        return (100.0 * count / denom) if denom > 0 else 0.0

    pct_roi_ge_1 = pct(thr1, total_pixels)
    pct_roi_ge_2 = pct(thr2, total_pixels)
    pct_roi_ge_5 = pct(thr5, total_pixels)

    pct_nz_ge_1 = pct(thr1, nonzero_pixels)
    pct_nz_ge_2 = pct(thr2, nonzero_pixels)
    pct_nz_ge_5 = pct(thr5, nonzero_pixels)

    def significance_label(percent_nonzero):
        if percent_nonzero < 1.0:
            return "Not significant"
        elif percent_nonzero < 5.0:
            return "Borderline"
        elif percent_nonzero < 10.0:
            return "Significant"
        else:
            return "Large significant region"

    return {
        "roi": roi,
        "peak_dose": peak_val,
        "peak_x": peak_x,
        "peak_y": peak_y,
        "mean_dose": mean_dose,
        "median_dose": median_dose,
        "std_dose": std_dose,
        "min_nonzero_dose": min_nonzero,
        "p95_dose": p95,
        "dose_sum": dose_sum,
        "total_pixels": total_pixels,
        "nonzero_pixels": nonzero_pixels,
        "thr1": thr1,
        "thr2": thr2,
        "thr5": thr5,
        "thr10": thr10,
        "pct_roi_ge_1": pct_roi_ge_1,
        "pct_roi_ge_2": pct_roi_ge_2,
        "pct_roi_ge_5": pct_roi_ge_5,
        "pct_nz_ge_1": pct_nz_ge_1,
        "pct_nz_ge_2": pct_nz_ge_2,
        "pct_nz_ge_5": pct_nz_ge_5,
        "sig_1gy": significance_label(pct_nz_ge_1),
        "sig_2gy": significance_label(pct_nz_ge_2),
        "sig_5gy": significance_label(pct_nz_ge_5),
        "stats_map": stats_map,
    }


def build_csv_bytes(session_rows, roi_stats):
    lines = []
    lines.append("file,displayed_max_gy,crop_box,reconstructed_peak_gy,nonzero_pixels")
    for row in session_rows:
        lines.append(
            f"{row['file']},{row['displayed_max_gy']},{row['crop_box']},{row['reconstructed_peak_gy']},{row['nonzero_pixels']}"
        )

    lines.append("")
    lines.append(
        "summary,roi,peak_dose_gy,peak_x,peak_y,mean_dose_gy,median_dose_gy,std_dose_gy,min_nonzero_dose_gy,p95_dose_gy,dose_sum,total_pixels,nonzero_pixels,pixels_ge_1gy,pixels_ge_2gy,pixels_ge_5gy,pixels_ge_10gy,pct_roi_ge_1gy,pct_roi_ge_2gy,pct_roi_ge_5gy,pct_nonzero_ge_1gy,pct_nonzero_ge_2gy,pct_nonzero_ge_5gy,sig_1gy,sig_2gy,sig_5gy"
    )
    lines.append(
        f"roi,{roi_stats['roi']},{roi_stats['peak_dose']},{roi_stats['peak_x']},{roi_stats['peak_y']},"
        f"{roi_stats['mean_dose']},{roi_stats['median_dose']},{roi_stats['std_dose']},{roi_stats['min_nonzero_dose']},"
        f"{roi_stats['p95_dose']},{roi_stats['dose_sum']},{roi_stats['total_pixels']},{roi_stats['nonzero_pixels']},"
        f"{roi_stats['thr1']},{roi_stats['thr2']},{roi_stats['thr5']},{roi_stats['thr10']},"
        f"{roi_stats['pct_roi_ge_1']},{roi_stats['pct_roi_ge_2']},{roi_stats['pct_roi_ge_5']},"
        f"{roi_stats['pct_nz_ge_1']},{roi_stats['pct_nz_ge_2']},{roi_stats['pct_nz_ge_5']},"
        f"{roi_stats['sig_1gy']},{roi_stats['sig_2gy']},{roi_stats['sig_5gy']}"
    )

    return "\n".join(lines).encode("utf-8")


def roi_from_selection(selection_state, width, height):
    if selection_state is None:
        return None

    sel = selection_state.get("selection", selection_state)

    points = sel.get("points", []) if isinstance(sel, dict) else []
    if not points:
        return None

    xs = []
    ys = []
    for p in points:
        x = p.get("x")
        y = p.get("y")
        if x is None or y is None:
            continue
        xs.append(float(x))
        ys.append(float(y))

    if not xs or not ys:
        return None

    x1 = int(np.floor(min(xs)))
    x2 = int(np.ceil(max(xs))) + 1
    y1 = int(np.floor(min(ys)))
    y2 = int(np.ceil(max(ys))) + 1
    return clamp_roi((x1, y1, x2, y2), width, height)


# =========================================================
# Session state
# =========================================================
if "reconstruction_signature" not in st.session_state:
    st.session_state.reconstruction_signature = None
if "cumulative_dose" not in st.session_state:
    st.session_state.cumulative_dose = None
if "processed_maps" not in st.session_state:
    st.session_state.processed_maps = None
if "session_rows" not in st.session_state:
    st.session_state.session_rows = None
if "inclusion_roi" not in st.session_state:
    st.session_state.inclusion_roi = None


# =========================================================
# Main UI
# =========================================================
uploaded_files = st.file_uploader(
    "Upload OpenREM PNG images",
    type=["png"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.subheader("Session dose calibration")
    st.write("Enter the displayed maximum dose shown at the top of the grayscale scale for each uploaded session.")

    displayed_max_inputs = {}
    dose_cols = st.columns(2)
    for i, file in enumerate(uploaded_files):
        file_key = f"{i}_{file.name}"
        with dose_cols[i % 2]:
            displayed_max_inputs[file_key] = st.number_input(
                f"{file.name} displayed max dose (Gy)",
                min_value=0.0,
                value=1.0,
                step=0.001,
                format="%.3f",
                key=f"maxdose_{file_key}",
            )

    with st.expander("Advanced settings"):
        crop_mode = st.selectbox(
            "Map panel crop",
            ["Auto-detect map panel", "Use full image", "Manual crop for all images"],
            index=1,
        )

        size_mode = st.selectbox(
            "Size harmonisation",
            [
                "Strict (all reconstructed maps must match exactly)",
                "Crop to smallest common size",
                "Pad to largest common size",
            ],
            index=2,
        )

        show_session_previews = st.checkbox("Show reconstructed session previews", value=False)
        show_histogram = st.checkbox("Show histogram for ROI", value=False)
        show_downloads = st.checkbox("Show downloads", value=True)

        sample_img = Image.open(uploaded_files[0]).convert("L")
        sample_arr = np.array(sample_img)
        H, W = sample_arr.shape

        manual_vals = None
        if crop_mode == "Manual crop for all images":
            st.write("Manual crop is applied to all sessions before reconstruction.")
            c1, c2 = st.columns(2)
            with c1:
                crop_x1 = st.number_input("Crop x1", min_value=0, max_value=W - 1, value=int(W * 0.08), step=1)
                crop_y1 = st.number_input("Crop y1", min_value=0, max_value=H - 1, value=int(H * 0.12), step=1)
            with c2:
                crop_x2 = st.number_input("Crop x2", min_value=1, max_value=W, value=int(W * 0.80), step=1)
                crop_y2 = st.number_input("Crop y2", min_value=1, max_value=H, value=int(H * 0.84), step=1)
            manual_vals = (int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2))
    if "crop_mode" not in locals():
        crop_mode = "Use full image"
        size_mode = "Pad to largest common size"
        manual_vals = None
        show_session_previews = False
        show_histogram = False
        show_downloads = True

    build_clicked = st.button("Build cumulative map", type="primary", width="stretch")

    current_signature = {
        "files": tuple((i, f.name, getattr(f, "size", None)) for i, f in enumerate(uploaded_files)),
        "displayed_max": tuple(
            (f"{i}_{f.name}", float(displayed_max_inputs[f"{i}_{f.name}"]))
            for i, f in enumerate(uploaded_files)
        ),
        "crop_mode": crop_mode,
        "size_mode": size_mode,
        "manual_vals": manual_vals,
    }

    needs_rebuild = (
        build_clicked
        or st.session_state.cumulative_dose is None
        or st.session_state.reconstruction_signature != current_signature
    )

    if needs_rebuild:
        try:
            session_rows = []
            reconstructed = []

            for i, file in enumerate(uploaded_files):
                file_key = f"{i}_{file.name}"
                displayed_max = float(displayed_max_inputs[file_key])

                img_gray = Image.open(file).convert("L")
                gray = np.array(img_gray, dtype=np.uint8)

                if crop_mode == "Auto-detect map panel":
                    x1, y1, x2, y2 = auto_find_map_bbox(gray)
                elif crop_mode == "Use full image":
                    x1, y1, x2, y2 = 0, 0, gray.shape[1], gray.shape[0]
                else:
                    x1, y1, x2, y2 = manual_vals

                crop = gray[y1:y2, x1:x2]
                dose_map = reconstruct_displayed_dose(crop, displayed_max_dose=displayed_max)

                reconstructed.append((file.name, dose_map, displayed_max, (x1, y1, x2, y2)))

                session_rows.append(
                    {
                        "file": file.name,
                        "displayed_max_gy": displayed_max,
                        "crop_box": f"({x1}, {y1}) to ({x2}, {y2})",
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

            processed_maps = []
            for name, dose_map, displayed_max, bbox in reconstructed:
                if size_mode == "Crop to smallest common size":
                    dose_proc = center_crop(dose_map, target_h, target_w)
                elif size_mode == "Pad to largest common size":
                    dose_proc = center_pad(dose_map, target_h, target_w, pad_value=0.0)
                else:
                    dose_proc = dose_map
                processed_maps.append((name, dose_proc, displayed_max, bbox))

            cumulative_dose = np.sum([m[1] for m in processed_maps], axis=0)

            st.session_state.cumulative_dose = cumulative_dose
            st.session_state.processed_maps = processed_maps
            st.session_state.session_rows = session_rows
            st.session_state.reconstruction_signature = current_signature
            st.session_state.inclusion_roi = None

        except Exception as e:
            st.error(f"Error: {e}")

    if st.session_state.cumulative_dose is not None:
        cumulative_dose = st.session_state.cumulative_dose
        processed_maps = st.session_state.processed_maps
        session_rows = st.session_state.session_rows

        st.subheader("Inclusion ROI")
        st.caption(
            "Use box select on the cumulative map. The ROI stats and previews appear after a selection. "
            "Selection uses a lightweight invisible grid to reduce stall time."
        )

        map_h, map_w = cumulative_dose.shape

        current_peak = None
        if st.session_state.inclusion_roi is not None:
            tmp_stats = compute_stats(cumulative_dose, st.session_state.inclusion_roi)
            current_peak = (tmp_stats["peak_x"], tmp_stats["peak_y"])

        selector_fig, grid_step = render_interactive_map(
            cumulative_dose,
            title="Draw inclusion ROI with box select",
            vmax=max(float(np.max(cumulative_dose)), 1.0),
            inclusion_roi=st.session_state.inclusion_roi,
            peak_xy=current_peak,
            selectable=True,
        )

        selection_event = st.plotly_chart(
            selector_fig,
            width="stretch",
            on_select="rerun",
            selection_mode=("box",),
            config={
                "modeBarButtonsToRemove": [
                    "lasso2d",
                    "zoom2d",
                    "pan2d",
                    "autoScale2d",
                ],
                "displaylogo": False,
            },
            key="roi_selector_chart",
        )

        new_roi = roi_from_selection(selection_event, map_w, map_h)
        if new_roi is not None:
            st.session_state.inclusion_roi = new_roi

        controls = st.columns([1, 1, 2])
        with controls[0]:
            if st.button("Clear inclusion ROI", width="stretch"):
                st.session_state.inclusion_roi = None
                st.rerun()
        with controls[1]:
            if st.button("Use full map as ROI", width="stretch"):
                st.session_state.inclusion_roi = (0, 0, map_w, map_h)
                st.rerun()

        st.caption(f"Selection grid step: {grid_step} px")

        if st.session_state.inclusion_roi is None:
            st.info("No ROI selected yet. Draw a rectangle on the map with box select, or use the full-map button.")
        else:
            inclusion_roi = st.session_state.inclusion_roi
            roi_stats = compute_stats(cumulative_dose, roi=inclusion_roi)
            display_map = masked_to_inclusion_roi(cumulative_dose, inclusion_roi)

            left, right = st.columns([1.4, 1])

            with left:
                result_fig, _ = render_interactive_map(
                    display_map,
                    title=f"Cumulative reconstructed dose | ROI peak {roi_stats['peak_dose']:.3f} Gy",
                    vmax=max(float(np.max(cumulative_dose)), 1.0),
                    inclusion_roi=inclusion_roi,
                    peak_xy=(roi_stats["peak_x"], roi_stats["peak_y"]),
                    selectable=False,
                )
                st.plotly_chart(
                    result_fig,
                    width="stretch",
                    config={"displaylogo": False},
                    key="roi_result_chart",
                )

            with right:
                st.metric("Number of sessions", len(processed_maps))
                st.metric("ROI peak dose", f"{roi_stats['peak_dose']:.3f} Gy")
                st.metric("ROI peak location", f"({roi_stats['peak_x']}, {roi_stats['peak_y']})")
                st.caption("Coordinates are cursor-style X,Y positions.")

                st.write("Inclusion ROI statistics")
                st.write(f"Mean dose: {roi_stats['mean_dose']:.3f} Gy")
                st.write(f"Median dose: {roi_stats['median_dose']:.3f} Gy")
                st.write(f"Std deviation: {roi_stats['std_dose']:.3f} Gy")
                st.write(f"Minimum non-zero dose: {roi_stats['min_nonzero_dose']:.3f} Gy")
                st.write(f"95th percentile: {roi_stats['p95_dose']:.3f} Gy")
                st.write(f"Dose sum over ROI: {roi_stats['dose_sum']:.3f}")
                st.write(f"Non-zero pixels: {roi_stats['nonzero_pixels']} of {roi_stats['total_pixels']}")

                st.write("Area above dose thresholds")
                st.write(
                    f"≥ 1 Gy: {roi_stats['thr1']} pixels | "
                    f"{roi_stats['pct_roi_ge_1']:.1f}% of ROI | "
                    f"{roi_stats['pct_nz_ge_1']:.1f}% of non-zero | "
                    f"{roi_stats['sig_1gy']}"
                )
                st.write(
                    f"≥ 2 Gy: {roi_stats['thr2']} pixels | "
                    f"{roi_stats['pct_roi_ge_2']:.1f}% of ROI | "
                    f"{roi_stats['pct_nz_ge_2']:.1f}% of non-zero | "
                    f"{roi_stats['sig_2gy']}"
                )
                st.write(
                    f"≥ 5 Gy: {roi_stats['thr5']} pixels | "
                    f"{roi_stats['pct_roi_ge_5']:.1f}% of ROI | "
                    f"{roi_stats['pct_nz_ge_5']:.1f}% of non-zero | "
                    f"{roi_stats['sig_5gy']}"
                )
                st.write(f"≥ 10 Gy: {roi_stats['thr10']} pixels")

            if show_session_previews:
                st.subheader("Reconstructed session previews")
                preview_cols = st.columns(2)
                for i, (name, dose_map, displayed_max, bbox) in enumerate(processed_maps):
                    with preview_cols[i % 2]:
                        st.markdown(f"**{name}**")
                        st.caption(f"Crop box: {bbox}")
                        session_display = masked_to_inclusion_roi(dose_map, inclusion_roi)
                        session_stats = compute_stats(dose_map, inclusion_roi)
                        session_fig, _ = render_interactive_map(
                            session_display,
                            title=f"{name} | ROI peak {session_stats['peak_dose']:.3f} Gy",
                            vmax=max(float(np.max(cumulative_dose)), 1.0),
                            inclusion_roi=inclusion_roi,
                            peak_xy=(session_stats["peak_x"], session_stats["peak_y"]),
                            selectable=False,
                        )
                        st.plotly_chart(
                            session_fig,
                            width="stretch",
                            config={"displaylogo": False},
                            key=f"session_chart_{i}",
                        )

            if show_histogram:
                st.subheader("Dose histogram inside ROI")
                st.caption(
                    "Significance labels use the percentage of non-zero ROI pixels above each threshold: "
                    "<1% Not significant, 1–5% Borderline, 5–10% Significant, >10% Large significant region."
                )
                nz = roi_stats["stats_map"][roi_stats["stats_map"] > 0]
                if nz.size > 0:
                    fig_hist, ax_hist = plt.subplots(figsize=(7, 4))
                    ax_hist.hist(nz.ravel(), bins=40)
                    ax_hist.set_xlabel("Dose (Gy)")
                    ax_hist.set_ylabel("Pixel count")
                    ax_hist.set_title("Histogram of non-zero dose values inside ROI")
                    st.pyplot(fig_hist, width="stretch")
                    plt.close(fig_hist)
                else:
                    st.info("No non-zero pixels inside the ROI.")

            if show_downloads:
                st.subheader("Downloads")

                fig_cum = render_matplotlib_map(
                    display_map,
                    title=f"Cumulative reconstructed dose | ROI peak {roi_stats['peak_dose']:.3f} Gy",
                    vmax=max(float(np.max(cumulative_dose)), 1.0),
                )
                cum_png = fig_to_png_bytes(fig_cum)
                plt.close(fig_cum)

                csv_bytes = build_csv_bytes(session_rows, roi_stats)

                npy_buf = io.BytesIO()
                np.save(npy_buf, cumulative_dose)
                npy_buf.seek(0)

                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    zf.writestr("cumulative_reconstructed_dose.npy", npy_buf.getvalue())
                    zf.writestr("cumulative_reconstructed_dose_roi_view.png", cum_png)
                    zf.writestr("reconstruction_summary.csv", csv_bytes)
                zip_buf.seek(0)

                d1, d2 = st.columns(2)
                with d1:
                    st.download_button(
                        "Download cumulative array (.npy)",
                        data=npy_buf.getvalue(),
                        file_name="cumulative_reconstructed_dose.npy",
                        mime="application/octet-stream",
                        width="stretch",
                    )
                with d2:
                    st.download_button(
                        "Download results (.zip)",
                        data=zip_buf.getvalue(),
                        file_name="reconstructed_dose_results.zip",
                        mime="application/zip",
                        width="stretch",
                    )
else:
    st.caption("Upload one or more OpenREM PNG images to begin.")
