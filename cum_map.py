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
    "All maps must have the same dimensions and already be anatomically aligned."
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

    use_global_scale = st.checkbox(
        "Use the same color scale for all session maps",
        value=True,
        help="When enabled, all individual maps use the same Gy range so colors can be compared directly.",
    )

    if st.button("Build cumulative map", type="primary"):
        try:
            abs_dose_maps = []
            session_summaries = []
            shape_ref = None

            for file in uploaded_files:
                img = Image.open(file).convert("L")
                arr = np.array(img, dtype=np.float32)

                if shape_ref is None:
                    shape_ref = arr.shape
                elif arr.shape != shape_ref:
                    raise ValueError(
                        f"Image size mismatch: {file.name} has shape {arr.shape}, expected {shape_ref}."
                    )

                max_skin_dose = dose_entries[file.name]
                if max_skin_dose <= 0:
                    raise ValueError(f"Max skin dose for {file.name} must be greater than 0.")

                abs_dose = (arr / 255.0) * max_skin_dose
                abs_dose_maps.append((file.name, abs_dose, max_skin_dose))

                session_summaries.append(
                    {
                        "file": file.name,
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

            st.success("Cumulative map created.")

            st.subheader("Individual calibrated session maps")
            global_session_vmax = max(m[2] for m in abs_dose_maps) if use_global_scale else None
            map_cols = st.columns(2)
            session_fig_buffers = []

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
                    fig_buf = fig_to_png_bytes(fig)
                    session_fig_buffers.append((name.replace('.png', '_colored.png'), fig_buf.getvalue()))
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
                cum_fig_buf = fig_to_png_bytes(fig_cum)
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

            npy_buf = io.BytesIO()
            np.save(npy_buf, cumulative_dose)
            npy_buf.seek(0)

            csv_lines = ["file,entered_max_skin_dose_gy,reconstructed_peak_gy,mean_dose_gy"]
            for row in session_summaries:
                csv_lines.append(
                    f"{row['file']},{row['entered_max_skin_dose_gy']},{row['reconstructed_peak_gy']},{row['mean_dose_gy']}"
                )
            csv_lines.append(f"TOTAL_PEAK,,{peak_cumulative_dose},")
            csv_bytes = "\n".join(csv_lines).encode("utf-8")

            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("cumulative_dose_map.npy", npy_buf.getvalue())
                zf.writestr("cumulative_dose_map_colored.png", cum_fig_buf.getvalue())
                zf.writestr("session_summary.csv", csv_bytes)
                for fname, content in session_fig_buffers:
                    zf.writestr(fname, content)
            zip_buf.seek(0)

            d1, d2 = st.columns(2)
            with d1:
                st.download_button(
                    "Download cumulative array (.npy)",
                    data=npy_buf.getvalue(),
                    file_name="cumulative_dose_map.npy",
                    mime="application/octet-stream",
                )
            with d2:
                st.download_button(
                    "Download all results (.zip)",
                    data=zip_buf.getvalue(),
                    file_name="dose_map_results.zip",
                    mime="application/zip",
                )

        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.caption("Upload one or more PNG dose maps to begin.")
