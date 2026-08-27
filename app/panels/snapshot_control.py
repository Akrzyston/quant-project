"""Snapshot capture and replay control."""

from __future__ import annotations

import streamlit as st

from app import data, state
from app.registry import Slot, panel
from voltk.snapshots import SnapshotError, rebuild_universe


@panel(
    key="snapshot_control",
    title="Snapshots",
    slot=Slot.LEFT_RAIL,
    order=50,
    milestone="M0",
)
def render() -> None:
    s = state.get()
    store = data.store()

    note = st.text_input("Note", key="snapshot_note", placeholder="optional label")
    disabled = not s.currencies
    if st.button("Capture snapshot", width="stretch", disabled=disabled):
        try:
            snapshot_id = data.capture_and_save(s.currencies, note=note)
        except Exception as exc:
            data.show_error(exc)
        else:
            st.success(f"Captured {snapshot_id}")
            st.rerun()
    if disabled:
        st.caption("Select a currency first.")

    saved = store.list()
    if not saved:
        st.caption("No snapshots yet.")
        return

    mode_is_replay = st.toggle("Replay mode", value=not s.is_live, key="replay_toggle")
    s.data_mode = "snapshot" if mode_is_replay else "live"

    options = [meta.snapshot_id for meta in saved]
    labels = {meta.snapshot_id: meta.display_name() for meta in saved}
    index = options.index(s.active_snapshot_id) if s.active_snapshot_id in options else 0
    s.active_snapshot_id = st.selectbox(
        "Snapshot",
        options,
        index=index,
        format_func=lambda k: labels[k],
        key="snapshot_select",
        disabled=not mode_is_replay,
    )

    meta = next(m for m in saved if m.snapshot_id == s.active_snapshot_id)
    st.caption(f"{meta.quality.summary()} · schema v{meta.schema_version}")
    if meta.degraded:
        for reason in meta.quality.reasons:
            st.warning(reason)

    with st.expander("Provenance"):
        st.caption(f"Source: {meta.source_label}")
        st.caption(f"Captured: {meta.started_at:%Y-%m-%d %H:%M:%S}Z")
        st.caption(f"As of: {meta.as_of:%Y-%m-%d %H:%M:%S}Z")
        st.caption(f"Content hash: {meta.content_hash[:16]}")
        if st.button("Verify replay", key="verify_replay"):
            try:
                snapshot = store.load(meta.snapshot_id)
                first = data.summarise(rebuild_universe(snapshot))["digest"]
                second = data.summarise(rebuild_universe(store.load(meta.snapshot_id)))["digest"]
            except SnapshotError as exc:
                st.error(str(exc))
            else:
                if first == second:
                    st.success(f"Reproducible · digest {first[:16]}")
                else:
                    st.error("Reload produced a different result.")
