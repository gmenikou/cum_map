import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import zipfile

st.set_page_config(page_title="Cumulative Dose Map GUI", layout="wide")
st.title("Cumulative Dose Map GUI")

st.write(
    "Upload any number of grayscale PNG dose maps. For each image, enter that session's max skin dose in Gy. "
    "The app calibrates each 0 to 255 map into absolute dose, shows every session in color with a dose scale, "
    "and builds a cumulative colored dose map."
)

st.info(
    "Assumptions: each PNG is an 8-bit grayscale map where 0 means zero dose and 255 means that session's max skin dose. "
    "Maps should ideally have the same dimensions and already be anatomically aligned."
)


def render_dose_figure(dose_map, title, vmax=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(dose_map, cmap="jet", vmin=0, vmax=vmax)
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
    start_y = max((h - target_h) // 2, 0)
    start_x = max((w - target_w) // 2, 0)
    return arr[start_y:start_y + target_h, start_x:start_x + target_w]


def center_pad(arr, target_h, target_w, pad_value=0):
    h, w = arr.shape
    out = np.full((target_h, target_w), pad_value, dtype=arr.dtype)
    start_y = (target_h - h) // 2
    start_x = (target_w - w) // 2
    out[start_y:start_y + h, start_x:start_x + w] = arr
    return out


uploaded_files = st.file_uploader(
    "Upload dose map PNG files",
    type=["png"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.subheader("Session calibration")
    st.write("Enter the max skin dose for each uploaded map.")

    dose_entries = {}
    input_cols = st.columns(2)
    for i, file in enumerate(uploaded_files):
        with input_cols[i % 2]:
            dose_entries[file.name] = st.number_input(
                f"Max skin dose for {file.name} (Gy)",
                min_value=0.0,
                value=1.0,
                step=0.1,
                format="%.3f",
                key=f"dose_{file.name}",
            )

    size_mode = st.selectbox(
        "How should mismatched image sizes be handled?",
        [
            "Strict (all images must match exactly)",
            "Crop to smallest common size",
            "Pad to largest common size",
        ],
        index=1,
    )

    use_global_scale = st.checkbox(
        "Use the same color scale for all session maps",
        value=True,
    )

    if st.button("Build cumulative map", type="primary"):
        try:
            raw_images = []
            shapes = []

            for file in uploaded_files:
                img = Image.open(file).convert("L")
                arr = np.array(img, dtype=np.float32)
                raw_images.append((file.name, arr))
                shapes.append(arr.shape)

            heights = [s[0] for s in shapes]
            widths = [s[1] for s in shapes]

            if size_mode == "Strict (all images must match exactly)":
                if len(set(shapes)) != 1:
                    raise ValueError(f"Image size mismatch: found shapes {sorted(set(shapes))}")
                target_h, target_w = shapes[0]

            elif size_mode == "Crop to smallest common size":
                target_h = min(heights)
                target_w = min(widths)

            elif size_mode == "Pad to largest common size":
                target_h = max(heights)
                target_w = max(widths)

            processed_images = []
            for name, arr in raw_images:
                if size_mode == "Crop to smallest common size":
                    arr2 = center_crop(arr, target_h, target_w)
                elif size_mode == "Pad to largest common size":
                    arr2 = center_pad(arr, target_h, target_w, pad_value=0)
                else:
                    arr2 = arr
                processed_images.append((name, arr2))

            abs_dose_maps = []
            session_summaries = []

            for name, arr in processed_images:
                max_skin_dose = dose_entries[name]
                if max_skin_dose <= 0:
                    raise ValueError(f"Max skin dose for {name} must be greater than 0.")

                abs_dose = (arr / 255.0) * max_skin_dose
                abs_dose_maps.append((name, abs_dose, max_skin_dose))

                session_summaries.append(
                    {
                        "file": name,
                        "shape_after_processing": str(arr.shape),
                        "entered_max_skin_dose_gy": float(max_skin_dose),
                        "reconstructed_peak_gy": float(abs_dose.max()),
                        "mean_dose_gy": float(abs_dose.mean()),
                    }
                )

            cumulative_dose = np.sum([m[1] for m in abs_dose_maps], axis=0)
            peak_cumulative_dose = float(cumulative_dose.max())
            peak_idx = np.unravel_index(np.argmax(cumulative_dose), cumulative_dose.shape)

            thr2 = int(np.sum(cumulative_dose >= 2.0))
            thr5 = int(np.sum(cumulative_dose >= 5.0))
            thr10 = int(np.sum(cumulative_dose >= 10.0))

            st.success(f"Cumulative map created. Final map size: {cumulative_dose.shape}")

            st.subheader("Individual calibrated session maps")
            global_session_vmax = max(m[2] for m in abs_dose_maps) if use_global_scale else None
            map_cols = st.columns(2)

            for i, (name, abs_dose, max_skin_dose) in enumerate(abs_dose_maps):
                vmax = global_session_vmax if use_global_scale else max_skin_dose
                with map_cols[i % 2]:
                    st.markdown(f"**{name}**")
                    fig = render_dose_figure(
                        abs_dose,
                        title=f"{name} | peak {abs_dose.max():.3f} Gy",
                        vmax=vmax,
                    )
                    st.pyplot(fig)
                    plt.close(fig)

            st.subheader("Cumulative dose map")
            c1, c2 = st.columns([1.3, 1])

            with c1:
                fig_cum = render_dose_figure(
                    cumulative_dose,
                    title=f"Cumulative Dose Map | peak {peak_cumulative_dose:.3f} Gy",
                    vmax=peak_cumulative_dose if peak_cumulative_dose > 0 else 1,
                )
                st.pyplot(fig_cum)
                plt.close(fig_cum)

            with c2:
                st.metric("Number of sessions", len(uploaded_files))
                st.metric("Peak cumulative dose", f"{peak_cumulative_dose:.3f} Gy")
                st.metric("Peak location (row, col)", f"{peak_idx}")
                st.write("Pixels above thresholds")
                st.write(f"≥ 2 Gy: {thr2}")
                st.write(f"≥ 5 Gy: {thr5}")
                st.write(f"≥ 10 Gy: {thr10}")

            st.subheader("Per-session summary")
            st.dataframe(session_summaries, use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.caption("Upload one or more PNG dose maps to begin.")
