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


def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def get_window_range(level, window, data_max=None):
    window = max(float(window), 1e-6)
    level = float(level)

    vmin = level - window / 2.0
    vmax = level + window / 2.0

    vmin = max(0.0, vmin)
    if data_max is not None:
        vmax = min(float(data_max), vmax)

    if vmax <= vmin:
        vmax = vmin + 1e-6

    return vmin, vmax


def init_window_state(view_key, data_max, default_auto=True):
    data_max = max(float(data_max), 1.0)

    auto_key = f"{view_key}_auto_window"
    level_key = f"{view_key}_level"
    window_key = f"{view_key}_window"

    if auto_key not in st.session_state:
        st.session_state[auto_key] = default_auto
    if level_key not in st.session_state:
        st.session_state[level_key] = data_max / 2.0
    if window_key not in st.session_state:
        st.session_state[window_key] = data_max


def get_window_state(view_key, data_max):
    data_max = max(float(data_max), 1.0)

    auto_key = f"{view_key}_auto_window"
    level_key = f"{view_key}_level"
    window_key = f"{view_key}_window"

    init_window_state(view_key, data_max)

    auto_window = st.session_state[auto_key]
    if auto_window:
        vmin, vmax = 0.0, data_max
    else:
        level = float(np.clip(st.session_state[level_key], 0.0, data_max))
        min_window = max(data_max / 1000.0, 0.001)
        window = float(np.clip(st.session_state[window_key], min_window, data_max))
        vmin, vmax = get_window_range(level, window, data_max=data_max)

    return vmin, vmax


def render_window_controls(view_key, data_max):
    data_max = max(float(data_max), 1.0)

    auto_key = f"{view_key}_auto_window"
    level_key = f"{view_key}_level"
    window_key = f"{view_key}_window"

    init_window_state(view_key, data_max)

    c1, c2, c3 = st.columns([1, 1, 1])

    with c1:
        st.checkbox("Auto WL/WW", key=auto_key)

    if st.session_state[auto_key]:
        st.session_state[level_key] = data_max / 2.0
        st.session_state[window_key] = data_max
        with c2:
            st.caption(f"WL: {st.session_state[level_key]:.3f} Gy")
        with c3:
            st.caption(f"WW: {st.session_state[window_key]:.3f} Gy")
    else:
        min_window = max(data_max / 1000.0, 0.001)
        step = max(data_max / 500.0, 0.001)

        st.session_state[level_key] = float(np.clip(st.session_state[level_key], 0.0, data_max))
        st.session_state[window_key] = float(np.clip(st.session_state[window_key], min_window, data_max))

        with c2:
            st.slider(
                "WL (Gy)",
                min_value=0.0,
                max_value=float(data_max),
                step=step,
                key=level_key,
            )
        with c3:
            st.slider(
                "WW (Gy)",
                min_value=float(min_window),
                max_value=float(data_max),
                step=step,
                key=window_key,
            )

    vmin, vmax = get_window_state(view_key, data_max)
    st.caption(f"Display range: {vmin:.3f} to {vmax:.3f} Gy")


def render_matplotlib_map(dose_map, title, vmin=None, vmax=None, show_contours=False, min_cluster=10, cluster_rules=None):
    fig, ax = plt.subplots(figsize=(7, 5))

    if vmin is None:
        vmin = 0.0
    if vmax is None or vmax <= vmin:
        vmax = max(float(np.max(dose_map)), vmin + 1e-6)

    im = ax.imshow(dose_map, cmap="jet", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Dose (Gy)")

    if show_contours:
        for thr in (1.0, 2.0, 5.0, 10.0):
            mask = dose_map >= thr
            thr_min_cluster = int(cluster_rules.get(thr, min_cluster)) if cluster_rules else int(min_cluster)
            cluster_masks = get_cluster_masks(mask, min_cluster=thr_min_cluster, connectivity=8)
            for cluster_mask in cluster_masks:
                ax.contour(cluster_mask.astype(float), levels=[0.5], linewidths=1.2)

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


def largest_connected_component_size(mask, connectivity=8):
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    largest = 0

    if connectivity == 8:
        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),            (0, 1),
            (1, -1),  (1, 0),   (1, 1),
        ]
    else:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue

            stack = [(y, x)]
            visited[y, x] = True
            size = 0

            while stack:
                cy, cx = stack.pop()
                size += 1

                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))

            largest = max(largest, size)

    return int(largest)


def get_cluster_masks(mask, min_cluster=10, connectivity=8):
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    clusters = []

    if connectivity == 8:
        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),            (0, 1),
            (1, -1),  (1, 0),   (1, 1),
        ]
    else:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue

            stack = [(y, x)]
            visited[y, x] = True
            component = []

            while stack:
                cy, cx = stack.pop()
                component.append((cy, cx))

                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))

            if len(component) >= int(min_cluster):
                comp_mask = np.zeros_like(mask, dtype=bool)
                for cy, cx in component:
                    comp_mask[cy, cx] = True
                clusters.append(comp_mask)

    return clusters


def get_component_sizes(mask, connectivity=8):
    clusters = get_cluster_masks(mask, min_cluster=1, connectivity=connectivity)
    return [int(np.sum(c)) for c in clusters]


def estimate_dynamic_cluster_rules_from_cumulative(
    cumulative_dose,
    thresholds=(1.0, 2.0, 5.0, 10.0),
    connectivity=8,
    lower_portion_percentile=60,
    nuisance_percentile=95,
    floor_px=4,
):
    rules = {}

    for thr in thresholds:
        mask = cumulative_dose >= float(thr)
        sizes = get_component_sizes(mask, connectivity=connectivity)

        if not sizes:
            rules[float(thr)] = int(floor_px)
            continue

        sizes = np.array(sorted(sizes), dtype=np.float32)

        if len(sizes) == 1:
            rules[float(thr)] = max(int(floor_px), int(np.ceil(sizes[0])))
            continue

        size_cut = np.percentile(sizes, lower_portion_percentile)
        small_sizes = sizes[sizes <= size_cut]

        if len(small_sizes) == 0:
            small_sizes = sizes

        thr_min_cluster = int(np.ceil(np.percentile(small_sizes, nuisance_percentile)))
        rules[float(thr)] = max(int(floor_px), thr_min_cluster)

    return rules


def cluster_label(largest_cluster, min_cluster=10):
    if largest_cluster < min_cluster:
        return "No meaningful cluster"
    elif largest_cluster < 20:
        return "Small meaningful cluster"
    elif largest_cluster < 50:
        return "Moderate meaningful cluster"
    else:
        return "Large meaningful cluster"


def add_isocontours(fig, dose_map, thresholds=(1, 2, 5, 10), min_cluster=10, cluster_rules=None):
    colors = {
        1: "#FF00FF",
        2: "#00FF66",
        5: "#FFA500",
        10: "#FF2D2D",
    }

    for thr in thresholds:
        mask = dose_map >= float(thr)
        if not np.any(mask):
            continue

        thr_min_cluster = int(cluster_rules.get(float(thr), min_cluster)) if cluster_rules else int(min_cluster)
        cluster_masks = get_cluster_masks(mask, min_cluster=thr_min_cluster, connectivity=8)
        if not cluster_masks:
            continue

        for cluster_mask in cluster_masks:
            m = cluster_mask.astype(np.uint8)
            h, w = m.shape

            segments_x = []
            segments_y = []

            for y in range(h):
                for x in range(w):
                    if m[y, x] != 1:
                        continue

                    if y == 0 or m[y - 1, x] == 0:
                        segments_x += [x - 0.5, x + 0.5, None]
                        segments_y += [y - 0.5, y - 0.5, None]

                    if y == h - 1 or m[y + 1, x] == 0:
                        segments_x += [x - 0.5, x + 0.5, None]
                        segments_y += [y + 0.5, y + 0.5, None]

                    if x == 0 or m[y, x - 1] == 0:
                        segments_x += [x - 0.5, x - 0.5, None]
                        segments_y += [y - 0.5, y + 0.5, None]

                    if x == w - 1 or m[y, x + 1] == 0:
                        segments_x += [x + 0.5, x + 0.5, None]
                        segments_y += [y - 0.5, y + 0.5, None]

            if not segments_x:
                continue

            fig.add_trace(
                go.Scatter(
                    x=segments_x,
                    y=segments_y,
                    mode="lines",
                    line=dict(color=colors.get(thr, "#FFFFFF"), width=5),
                    hoverinfo="none",
                    hovertemplate=None,
                    showlegend=False,
                )
            )


def compute_stats(dose_map, roi=None, cluster_rules=None, default_min_cluster=10):
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

    cl1 = largest_connected_component_size(stats_map >= 1.0)
    cl2 = largest_connected_component_size(stats_map >= 2.0)
    cl5 = largest_connected_component_size(stats_map >= 5.0)
    cl10 = largest_connected_component_size(stats_map >= 10.0)

    min1 = int(cluster_rules.get(1.0, default_min_cluster)) if cluster_rules else int(default_min_cluster)
    min2 = int(cluster_rules.get(2.0, default_min_cluster)) if cluster_rules else int(default_min_cluster)
    min5 = int(cluster_rules.get(5.0, default_min_cluster)) if cluster_rules else int(default_min_cluster)
    min10 = int(cluster_rules.get(10.0, default_min_cluster)) if cluster_rules else int(default_min_cluster)

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
        "cl1": cl1,
        "cl2": cl2,
        "cl5": cl5,
        "cl10": cl10,
        "min1": min1,
        "min2": min2,
        "min5": min5,
        "min10": min10,
        "flag1": cl1 >= min1,
        "flag2": cl2 >= min2,
        "flag5": cl5 >= min5,
        "flag10": cl10 >= min10,
        "label1": cluster_label(cl1, min_cluster=min1),
        "label2": cluster_label(cl2, min_cluster=min2),
        "label5": cluster_label(cl5, min_cluster=min5),
        "label10": cluster_label(cl10, min_cluster=min10),
        "stats_map": stats_map,
    }


def build_csv_bytes(session_rows, roi_stats, cluster_rules):
    lines = []
    lines.append("file,displayed_max_gy,crop_box,reconstructed_peak_gy,nonzero_pixels")
    for row in session_rows:
        lines.append(
            f"{row['file']},{row['displayed_max_gy']},{row['crop_box']},{row['reconstructed_peak_gy']},{row['nonzero_pixels']}"
        )

    lines.append("")
    lines.append("adaptive_cluster_rule,derived_from,cumulative_map_small_component_distribution")
    lines.append("dose_threshold_gy,min_connected_pixels")
    for thr in [1.0, 2.0, 5.0, 10.0]:
        lines.append(f"{thr},{int(cluster_rules.get(thr, 0))}")

    lines.append("")
    lines.append(
        "summary,roi,peak_dose_gy,peak_x,peak_y,mean_dose_gy,median_dose_gy,std_dose_gy,min_nonzero_dose_gy,p95_dose_gy,dose_sum,total_pixels,nonzero_pixels,"
        "pixels_ge_1gy,largest_cluster_ge_1gy,min_cluster_ge_1gy,meaningful_cluster_ge_1gy,label_ge_1gy,"
        "pixels_ge_2gy,largest_cluster_ge_2gy,min_cluster_ge_2gy,meaningful_cluster_ge_2gy,label_ge_2gy,"
        "pixels_ge_5gy,largest_cluster_ge_5gy,min_cluster_ge_5gy,meaningful_cluster_ge_5gy,label_ge_5gy,"
        "pixels_ge_10gy,largest_cluster_ge_10gy,min_cluster_ge_10gy,meaningful_cluster_ge_10gy,label_ge_10gy"
    )
    lines.append(
        f"roi,{roi_stats['roi']},{roi_stats['peak_dose']},{roi_stats['peak_x']},{roi_stats['peak_y']},"
        f"{roi_stats['mean_dose']},{roi_stats['median_dose']},{roi_stats['std_dose']},{roi_stats['min_nonzero_dose']},"
        f"{roi_stats['p95_dose']},{roi_stats['dose_sum']},{roi_stats['total_pixels']},{roi_stats['nonzero_pixels']},"
        f"{roi_stats['thr1']},{roi_stats['cl1']},{roi_stats['min1']},{roi_stats['flag1']},{roi_stats['label1']},"
        f"{roi_stats['thr2']},{roi_stats['cl2']},{roi_stats['min2']},{roi_stats['flag2']},{roi_stats['label2']},"
        f"{roi_stats['thr5']},{roi_stats['cl5']},{roi_stats['min5']},{roi_stats['flag5']},{roi_stats['label5']},"
        f"{roi_stats['thr10']},{roi_stats['cl10']},{roi_stats['min10']},{roi_stats['flag10']},{roi_stats['label10']}"
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


def make_edit_figure(dose_map, title, vmin, vmax):
    h, w = dose_map.shape
    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=dose_map,
            colorscale="Jet",
            zmin=vmin,
            zmax=vmax,
            colorbar=dict(title="Dose (Gy)"),
            hoverongaps=False,
            hovertemplate="X: %{x}<br>Y: %{y}<br>Dose: %{z:.3f} Gy<extra></extra>",
        )
    )

    gx, gy, step = build_selectable_grid(w, h)
    fig.add_trace(
        go.Scatter(
            x=gx,
            y=gy,
            mode="markers",
            marker=dict(size=2, color="rgba(0,0,0,0)"),
            hoverinfo="none",
            hovertemplate=None,
            showlegend=False,
            selected=dict(marker=dict(opacity=0)),
            unselected=dict(marker=dict(opacity=0)),
        )
    )

    fig.update_layout(
        title=title,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="X",
        yaxis_title="Y",
        dragmode="select",
        hovermode="closest",
    )
    fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1)
    return fig, step


def make_clean_hover_figure(dose_map, title, vmin, vmax, inclusion_roi=None):
    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=dose_map,
            colorscale="Jet",
            zmin=vmin,
            zmax=vmax,
            colorbar=dict(title="Dose (Gy)"),
            hoverongaps=False,
            hovertemplate="X: %{x}<br>Y: %{y}<br>Dose: %{z:.3f} Gy<extra></extra>",
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

    fig.update_layout(
        title=title,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="X",
        yaxis_title="Y",
        dragmode="pan",
        hovermode="closest",
    )
    fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1)

    return fig


def make_contour_figure(dose_map, title, vmin, vmax, inclusion_roi=None, min_cluster=10, cluster_rules=None):
    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=dose_map,
            colorscale="Jet",
            zmin=vmin,
            zmax=vmax,
            colorbar=dict(title="Dose (Gy)"),
            hoverinfo="skip",
        )
    )

    fig.update_traces(opacity=0.85, selector=dict(type="heatmap"))

    if inclusion_roi is not None:
        x1, y1, x2, y2 = inclusion_roi
        fig.add_shape(
            type="rect",
            x0=x1, y0=y1, x1=x2, y1=y2,
            line=dict(color="white", width=2),
            fillcolor="rgba(0,0,0,0)"
        )

    add_isocontours(
        fig,
        dose_map,
        thresholds=(1, 2, 5, 10),
        min_cluster=min_cluster,
        cluster_rules=cluster_rules,
    )

    legend_items = [
        ("≥ 1 Gy", "#FF00FF"),
        ("≥ 2 Gy", "#00FF66"),
        ("≥ 5 Gy", "#FFA500"),
        ("≥ 10 Gy", "#FF2D2D"),
    ]

    for name, col in legend_items:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color=col, width=4),
                name=name,
                showlegend=True,
            )
        )

    if inclusion_roi is not None:
        x1, y1, x2, y2 = inclusion_roi
        pad_x = max(10, int(round((x2 - x1) * 0.08)))
        pad_y = max(10, int(round((y2 - y1) * 0.08)))

        x_min = max(0, x1 - pad_x)
        x_max = min(dose_map.shape[1], x2 + pad_x)
        y_min = max(0, y1 - pad_y)
        y_max = min(dose_map.shape[0], y2 + pad_y)

        fig.update_xaxes(range=[x_min, x_max])
        fig.update_yaxes(range=[y_max, y_min])
    else:
        fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1)

    fig.update_layout(
        title=title,
        margin=dict(l=10, r=180, t=40, b=10),
        xaxis_title="X",
        yaxis_title="Y",
        legend=dict(
            title="Isocontours",
            x=1.18,
            y=1,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="rgba(0,0,0,0.15)",
            borderwidth=1,
        ),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    return fig


if "reconstruction_signature" not in st.session_state:
    st.session_state.reconstruction_signature = None
if "cumulative_dose" not in st.session_state:
    st.session_state.cumulative_dose = None
if "processed_maps" not in st.session_state:
    st.session_state.processed_maps = None
if "session_rows" not in st.session_state:
    st.session_state.session_rows = None
if "dynamic_cluster_rules" not in st.session_state:
    st.session_state.dynamic_cluster_rules = None
if "inclusion_roi" not in st.session_state:
    st.session_state.inclusion_roi = None
if "roi_edit_mode" not in st.session_state:
    st.session_state.roi_edit_mode = True


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

        fallback_min_cluster_pixels = st.number_input(
            "Fallback minimum connected cluster (pixels)",
            min_value=1,
            value=4,
            step=1,
        )

        adaptive_lower_portion_percentile = st.slider(
            "Adaptive rule: lower portion of component sizes (%)",
            min_value=20,
            max_value=90,
            value=60,
            step=5,
        )

        adaptive_nuisance_percentile = st.slider(
            "Adaptive rule: nuisance percentile (%)",
            min_value=70,
            max_value=99,
            value=95,
            step=1,
        )

        show_isocontours = st.checkbox("Show thin color-coded isocontours", value=True)
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
        fallback_min_cluster_pixels = 4
        adaptive_lower_portion_percentile = 60
        adaptive_nuisance_percentile = 95
        show_isocontours = True
        show_session_previews = False
        show_histogram = False
        show_downloads = True
        manual_vals = None

    build_clicked = st.button("Build cumulative map", type="primary", width="stretch")

    current_signature = {
        "files": tuple((i, f.name, getattr(f, "size", None)) for i, f in enumerate(uploaded_files)),
        "displayed_max": tuple(
            (f"{i}_{f.name}", float(displayed_max_inputs[f'{i}_{f.name}']))
            for i, f in enumerate(uploaded_files)
        ),
        "crop_mode": crop_mode,
        "size_mode": size_mode,
        "manual_vals": manual_vals,
        "fallback_min_cluster_pixels": int(fallback_min_cluster_pixels),
        "adaptive_lower_portion_percentile": int(adaptive_lower_portion_percentile),
        "adaptive_nuisance_percentile": int(adaptive_nuisance_percentile),
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

            dynamic_cluster_rules = estimate_dynamic_cluster_rules_from_cumulative(
                cumulative_dose,
                thresholds=(1.0, 2.0, 5.0, 10.0),
                connectivity=8,
                lower_portion_percentile=adaptive_lower_portion_percentile,
                nuisance_percentile=adaptive_nuisance_percentile,
                floor_px=fallback_min_cluster_pixels,
            )

            st.session_state.cumulative_dose = cumulative_dose
            st.session_state.processed_maps = processed_maps
            st.session_state.session_rows = session_rows
            st.session_state.dynamic_cluster_rules = dynamic_cluster_rules
            st.session_state.reconstruction_signature = current_signature
            st.session_state.inclusion_roi = None
            st.session_state.roi_edit_mode = True

        except Exception as e:
            st.error(f"Error: {e}")

    if st.session_state.cumulative_dose is not None:
        cumulative_dose = st.session_state.cumulative_dose
        processed_maps = st.session_state.processed_maps
        session_rows = st.session_state.session_rows
        dynamic_cluster_rules = st.session_state.dynamic_cluster_rules

        st.subheader("Inclusion ROI")
        map_h, map_w = cumulative_dose.shape

        controls = st.columns([1, 1, 1, 2])
        with controls[0]:
            if st.button("Clear inclusion ROI", width="stretch"):
                st.session_state.inclusion_roi = None
                st.session_state.roi_edit_mode = True
                st.rerun()
        with controls[1]:
            if st.button("Use full map as ROI", width="stretch"):
                st.session_state.inclusion_roi = (0, 0, map_w, map_h)
                st.session_state.roi_edit_mode = False
                st.rerun()
        with controls[2]:
            if st.button("Redraw ROI", width="stretch"):
                st.session_state.roi_edit_mode = True
                st.rerun()

        selector_max = max(float(np.max(cumulative_dose)), 1.0)
        selector_vmin, selector_vmax = get_window_state("selector_view", selector_max)

        if st.session_state.roi_edit_mode:
            edit_fig, grid_step = make_edit_figure(
                cumulative_dose,
                "ROI edit mode: draw box selection",
                selector_vmin,
                selector_vmax,
            )

            selection_event = st.plotly_chart(
                edit_fig,
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
                key="edit_chart_unique",
            )

            render_window_controls("selector_view", selector_max)
            st.caption(f"Selection grid step: {grid_step} px")

            new_roi = roi_from_selection(selection_event, map_w, map_h)
            if new_roi is not None:
                st.session_state.inclusion_roi = new_roi
                st.session_state.roi_edit_mode = False
                st.rerun()

            if st.session_state.inclusion_roi is None:
                st.info("Draw a rectangle on the map.")
        else:
            hover_fig = make_clean_hover_figure(
                cumulative_dose,
                "Hover view: cumulative map with ROI",
                selector_vmin,
                selector_vmax,
                inclusion_roi=st.session_state.inclusion_roi,
            )

            st.plotly_chart(
                hover_fig,
                width="stretch",
                config={
                    "modeBarButtonsToRemove": [
                        "select2d",
                        "lasso2d",
                        "zoom2d",
                        "autoScale2d",
                    ],
                    "displaylogo": False,
                },
                key="hover_chart_unique",
            )

            render_window_controls("selector_view", selector_max)

        if st.session_state.inclusion_roi is None:
            st.info("No ROI selected yet.")
        else:
            inclusion_roi = st.session_state.inclusion_roi
            roi_stats = compute_stats(
                cumulative_dose,
                roi=inclusion_roi,
                cluster_rules=dynamic_cluster_rules,
                default_min_cluster=fallback_min_cluster_pixels,
            )
            display_map = masked_to_inclusion_roi(cumulative_dose, inclusion_roi)

            left, right = st.columns([1.4, 1])

            with left:
                result_max = max(float(np.max(cumulative_dose)), 1.0)
                result_vmin, result_vmax = get_window_state("roi_result_view", result_max)

                result_fig = make_clean_hover_figure(
                    display_map,
                    f"Cumulative reconstructed dose | ROI peak {roi_stats['peak_dose']:.3f} Gy",
                    result_vmin,
                    result_vmax,
                    inclusion_roi=inclusion_roi,
                )

                st.plotly_chart(
                    result_fig,
                    width="stretch",
                    config={
                        "modeBarButtonsToRemove": [
                            "select2d",
                            "lasso2d",
                            "zoom2d",
                            "autoScale2d",
                        ],
                        "displaylogo": False,
                    },
                    key="roi_result_chart_clean",
                )

                render_window_controls("roi_result_view", result_max)

                if show_isocontours:
                    st.subheader("Isocontour view (visual only)")

                    contour_fig = make_contour_figure(
                        display_map,
                        "Cumulative dose with isocontours",
                        result_vmin,
                        result_vmax,
                        inclusion_roi=inclusion_roi,
                        min_cluster=fallback_min_cluster_pixels,
                        cluster_rules=dynamic_cluster_rules,
                    )

                    st.plotly_chart(
                        contour_fig,
                        width="stretch",
                        config={"displaylogo": False},
                        key="contour_chart",
                    )
                    st.caption("Contour lines show connected regions meeting each fixed dose threshold and its adaptive connected-pixel rule.")

            with right:
                st.metric("Number of sessions", len(processed_maps))
                st.metric("ROI peak dose", f"{roi_stats['peak_dose']:.3f} Gy")
                st.metric("ROI peak location", f"({roi_stats['peak_x']}, {roi_stats['peak_y']})")
                st.caption("Coordinates are cursor-style X,Y positions.")

                st.write("Adaptive contour pixel thresholds")
                st.write(f"≥ 1 Gy: {roi_stats['min1']} px")
                st.write(f"≥ 2 Gy: {roi_stats['min2']} px")
                st.write(f"≥ 5 Gy: {roi_stats['min5']} px")
                st.write(f"≥ 10 Gy: {roi_stats['min10']} px")

                st.write("Inclusion ROI statistics")
                st.write(f"Mean dose: {roi_stats['mean_dose']:.3f} Gy")
                st.write(f"Median dose: {roi_stats['median_dose']:.3f} Gy")
                st.write(f"Std deviation: {roi_stats['std_dose']:.3f} Gy")
                st.write(f"Minimum non-zero dose: {roi_stats['min_nonzero_dose']:.3f} Gy")
                st.write(f"95th percentile: {roi_stats['p95_dose']:.3f} Gy")
                st.write(f"Dose sum over ROI: {roi_stats['dose_sum']:.3f}")
                st.write(f"Non-zero pixels: {roi_stats['nonzero_pixels']} of {roi_stats['total_pixels']}")

                st.write("Threshold extent by connected cluster")
                st.write(
                    f"≥ 1 Gy: {roi_stats['thr1']} pixels | "
                    f"largest cluster {roi_stats['cl1']} px | "
                    f"min cluster {roi_stats['min1']} px | "
                    f"{roi_stats['label1']} | "
                    f"{'YES' if roi_stats['flag1'] else 'NO'}"
                )
                st.write(
                    f"≥ 2 Gy: {roi_stats['thr2']} pixels | "
                    f"largest cluster {roi_stats['cl2']} px | "
                    f"min cluster {roi_stats['min2']} px | "
                    f"{roi_stats['label2']} | "
                    f"{'YES' if roi_stats['flag2'] else 'NO'}"
                )
                st.write(
                    f"≥ 5 Gy: {roi_stats['thr5']} pixels | "
                    f"largest cluster {roi_stats['cl5']} px | "
                    f"min cluster {roi_stats['min5']} px | "
                    f"{roi_stats['label5']} | "
                    f"{'YES' if roi_stats['flag5'] else 'NO'}"
                )
                st.write(
                    f"≥ 10 Gy: {roi_stats['thr10']} pixels | "
                    f"largest cluster {roi_stats['cl10']} px | "
                    f"min cluster {roi_stats['min10']} px | "
                    f"{roi_stats['label10']} | "
                    f"{'YES' if roi_stats['flag10'] else 'NO'}"
                )

            if show_session_previews:
                st.subheader("Reconstructed session previews")
                preview_cols = st.columns(2)
                for i, (name, dose_map, displayed_max, bbox) in enumerate(processed_maps):
                    with preview_cols[i % 2]:
                        st.markdown(f"**{name}**")
                        st.caption(f"Crop box: {bbox}")

                        session_display = masked_to_inclusion_roi(dose_map, inclusion_roi)
                        session_stats = compute_stats(
                            dose_map,
                            inclusion_roi,
                            cluster_rules=dynamic_cluster_rules,
                            default_min_cluster=fallback_min_cluster_pixels,
                        )

                        session_max = max(float(np.max(dose_map)), 1.0)
                        session_vmin, session_vmax = get_window_state(f"session_view_{i}", session_max)

                        session_fig = make_clean_hover_figure(
                            session_display,
                            f"{name} | ROI peak {session_stats['peak_dose']:.3f} Gy",
                            session_vmin,
                            session_vmax,
                            inclusion_roi=inclusion_roi,
                        )

                        st.plotly_chart(
                            session_fig,
                            width="stretch",
                            config={
                                "modeBarButtonsToRemove": [
                                    "select2d",
                                    "lasso2d",
                                    "zoom2d",
                                    "autoScale2d",
                                ],
                                "displaylogo": False,
                            },
                            key=f"session_chart_{i}",
                        )

                        render_window_controls(f"session_view_{i}", session_max)

                        if show_isocontours:
                            session_contour_fig = make_contour_figure(
                                session_display,
                                f"{name} | Isocontours",
                                session_vmin,
                                session_vmax,
                                inclusion_roi=inclusion_roi,
                                min_cluster=fallback_min_cluster_pixels,
                                cluster_rules=dynamic_cluster_rules,
                            )
                            st.plotly_chart(
                                session_contour_fig,
                                width="stretch",
                                config={"displaylogo": False},
                                key=f"session_contour_chart_{i}",
                            )

            if show_histogram:
                st.subheader("Dose histogram inside ROI")
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
                    vmin=result_vmin,
                    vmax=result_vmax,
                    show_contours=show_isocontours,
                    min_cluster=fallback_min_cluster_pixels,
                    cluster_rules=dynamic_cluster_rules,
                )
                cum_png = fig_to_png_bytes(fig_cum)
                plt.close(fig_cum)

                csv_bytes = build_csv_bytes(session_rows, roi_stats, dynamic_cluster_rules)

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
