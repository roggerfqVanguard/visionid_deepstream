#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import pyds

# ---------------------------
# User-config (mirrors your .txt)
# ---------------------------
SOURCES = [
    ("file:///home/vanguardvision/DeepStream-Yolo_face/video_faces.mp4", 4),
    ("file:///home/vanguardvision/DeepStream-Yolo_face/video_faces.mp4", 4),
    ("file:///home/vanguardvision/DeepStream-Yolo_face/video_faces.mp4", 4),
    ("file:///home/vanguardvision/DeepStream-Yolo_face/video_faces.mp4", 4),
    ("file:///home/vanguardvision/DeepStream-Yolo_face/video_faces.mp4", 4),
]
STREAMMUX = dict(
    gpu_id=0, live_source=0, batch_size=5, batched_push_timeout=40000,
    width=1920, height=1080, enable_padding=0, nvbuf_memory_type=0,
)
TILED_DISPLAY = dict(enable=1, rows=2, columns=3, width=1920, height=1080, gpu_id=0, nvbuf_memory_type=0)
SINK0 = dict(enable=1, type=2, sync=0, gpu_id=0, nvbuf_memory_type=0)

PGIE = dict(
    enable=1, gpu_id=0, gie_unique_id=1, nvbuf_memory_type=0,
    config_file="/home/vanguardvision/DeepStream-Yolo/config_infer_primary_yolo11.txt",
)
TRACKER = dict(
    enable=1, tracker_width=960, tracker_height=544,
    ll_lib_file="/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
    ll_config_file="/home/vanguardvision/DeepStream-Yolo/config_tracker_NvDeepSORT.yml",
    gpu_id=0,
)
SGIE0 = dict(
    enable=1, gpu_id=0, gie_unique_id=2, nvbuf_memory_type=0,
    process_mode=2,  # IMPORTANT: also set operate-on-gie-id=1 in the SGIE config file
    config_file="/home/vanguardvision/DeepStream-Yolo_face/config_infer_primary_yolo11.txt",
)

# Logging throttle (console preview only)
PRINT_EVERY_N_FRAMES = 30
ENABLE_CONSOLE_LOG = True

# ---------------------------
# Boilerplate / helpers
# ---------------------------
def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, dbg = message.parse_error()
        sys.stderr.write(f"Error: {err}, {dbg}\n")
        loop.quit()
    return True

def make_element(factory, name=None):
    el = Gst.ElementFactory.make(factory, name if name else factory)
    if not el:
        raise RuntimeError(f"Failed to create GStreamer element: {factory}")
    return el

def create_source_bin(index, uri, drop_frame_interval):
    """Use nvurisrcbin so we can set drop-frame-interval and keep NVMM fast path."""
    bin_name = f"source-bin-{index:02d}"
    nbin = Gst.Bin.new(bin_name)

    srcbin = make_element("nvurisrcbin", f"uri-decode-bin-{index}")
    srcbin.set_property("uri", uri)
    srcbin.set_property("cudadec-memtype", 0)
    if drop_frame_interval and drop_frame_interval > 0:
        srcbin.set_property("drop-frame-interval", drop_frame_interval)

    nbin.add(srcbin)
    ghost = Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC)
    nbin.add_pad(ghost)

    def on_pad_added(element, pad):
        ghost.set_target(pad)

    srcbin.connect("pad-added", on_pad_added)
    return nbin

def add_nvtracker_props(tracker):
    tracker.set_property("tracker-width", TRACKER["tracker_width"])
    tracker.set_property("tracker-height", TRACKER["tracker_height"])
    tracker.set_property("gpu_id", TRACKER["gpu_id"])
    tracker.set_property("ll-lib-file", TRACKER["ll_lib_file"])
    tracker.set_property("ll-config-file", TRACKER["ll_config_file"])

def add_nvinfer_common_props(infer, cfg):
    infer.set_property("gpu-id", cfg["gpu_id"])
    infer.set_property("unique-id", cfg["gie_unique_id"])
    infer.set_property("config-file-path", cfg["config_file"])

# ---------------------------
# Unified probe (tracked parents + face objects) w/o child_obj_list
# ---------------------------
PGIE_UID = PGIE["gie_unique_id"]
SGIE_UID = SGIE0["gie_unique_id"]

def _set_box_style(obj_meta, rgba=(0.0, 1.0, 0.0, 1.0), border=3):
    """Ensure nvdsosd draws the rectangle by setting border width & color."""
    rp = obj_meta.rect_params
    rp.border_width = border
    color = pyds.NvOSD_ColorParams()
    color.red, color.green, color.blue, color.alpha = rgba
    rp.border_color = color
    rp.has_bg_color = 0  # no filled rect

def unified_src_pad_buffer_probe(pad, info, u_data):
    """
    Single probe attached AFTER SGIE:
      - Collect top-level/tracked objects (producer ~ PGIE/Tracker).
      - Collect face objects (producer ~ SGIE), link to parent via obj_meta.parent.
      - Force bbox draw by setting rect_params.border_width & border_color.
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    # We’ll gather all objects first, then separate by producer id.
    all_parents = []  # tracked parents
    all_faces = []    # faces detected by SGIE

    l_frame = batch_meta.frame_meta_list
    while l_frame:
        frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        l_obj = frame_meta.obj_meta_list
        while l_obj:
            obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)

            # Producer that last modified this object (PGIE or SGIE)
            comp_id = int(obj_meta.unique_component_id)

            # Force drawing style for everything we touch
            if comp_id == SGIE_UID:
                _set_box_style(obj_meta, rgba=(1.0, 0.3, 0.3, 1.0), border=3)  # faces reddish
                all_faces.append((frame_meta, obj_meta))
            else:
                # treat as parent/tracked (PGIE/Tracker produced)
                _set_box_style(obj_meta, rgba=(0.1, 0.8, 1.0, 1.0), border=3)  # cyan-ish
                all_parents.append((frame_meta, obj_meta))

            l_obj = l_obj.next
        l_frame = l_frame.next

    # Build output records (parents first)
    out_records = []
    for frame_meta, obj_meta in all_parents:
        rp = obj_meta.rect_params
        out_records.append({
            "role": "track",
            "stream_id": int(frame_meta.pad_index),
            "frame_num": int(frame_meta.frame_num),
            "track_id": int(obj_meta.object_id),
            "class_id": int(obj_meta.class_id),
            "confidence": float(obj_meta.confidence),
            "bbox": {"left": float(rp.left), "top": float(rp.top),
                     "width": float(rp.width), "height": float(rp.height)},
        })

    # Faces (find parent via obj_meta.parent when available)
    for frame_meta, face_meta in all_faces:
        parent_id = -1
        if face_meta.parent is not None:
            try:
                parent_id = int(pyds.NvDsObjectMeta.cast(face_meta.parent).object_id)
            except Exception:
                parent_id = -1
        rc = face_meta.rect_params
        out_records.append({
            "role": "face",
            "stream_id": int(frame_meta.pad_index),
            "frame_num": int(frame_meta.frame_num),
            "parent_track_id": parent_id,
            "class_id": int(face_meta.class_id),
            "confidence": float(face_meta.confidence),
            "bbox": {"left": float(rc.left), "top": float(rc.top),
                     "width": float(rc.width), "height": float(rc.height)},
        })

    # Your hook: forward to sink (Kafka/DB/file/etc.). Here: preview every N frames.
    if out_records and ENABLE_CONSOLE_LOG:
        if (out_records[0]["frame_num"] % PRINT_EVERY_N_FRAMES) == 0:
            print(f"[unified] {len(out_records)} recs; sample: {json.dumps(out_records[:4], ensure_ascii=False)}")

    return Gst.PadProbeReturn.OK

# ---------------------------
# Main
# ---------------------------
def main():
    Gst.init(None)

    pipeline = make_element("pipeline", "deepstream-pipeline")

    # Streammux
    streammux = make_element("nvstreammux", "stream-muxer")
    streammux.set_property("gpu-id", STREAMMUX["gpu_id"])
    streammux.set_property("live-source", STREAMMUX["live_source"])
    streammux.set_property("batch-size", STREAMMUX["batch_size"])
    streammux.set_property("batched-push-timeout", STREAMMUX["batched_push_timeout"])
    streammux.set_property("width", STREAMMUX["width"])
    streammux.set_property("height", STREAMMUX["height"])
    streammux.set_property("enable-padding", STREAMMUX["enable_padding"])
    streammux.set_property("nvbuf-memory-type", STREAMMUX["nvbuf_memory_type"])
    pipeline.add(streammux)

    # Sources
    for i, (uri, dfi) in enumerate(SOURCES):
        srcbin = create_source_bin(i, uri, dfi)
        pipeline.add(srcbin)
        sinkpad = streammux.request_pad_simple(f"sink_{i}")
        if not sinkpad:
            raise RuntimeError("Failed to request sink pad on streammux")
        srcpad = srcbin.get_static_pad("src")
        if not srcpad:
            raise RuntimeError("Failed to get src pad from source bin")
        if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
            raise RuntimeError("Failed to link source bin to streammux")

    # Queues
    q1 = make_element("queue", "q1")
    q2 = make_element("queue", "q2")
    q3 = make_element("queue", "q3")
    q4 = make_element("queue", "q4")
    q5 = make_element("queue", "q5")
    for q in (q1, q2, q3, q4, q5):
        pipeline.add(q)

    # Primary GIE (YOLO)
    pgie = make_element("nvinfer", "primary-gie")
    add_nvinfer_common_props(pgie, PGIE)
    pipeline.add(pgie)

    # Tracker
    tracker = make_element("nvtracker", "tracker")
    add_nvtracker_props(tracker)
    pipeline.add(tracker)

    # Secondary GIE (faces) — make sure SGIE config has operate-on-gie-id=1
    sgie0 = make_element("nvinfer", "secondary-gie0")
    add_nvinfer_common_props(sgie0, SGIE0)
    sgie0.set_property("process-mode", SGIE0["process_mode"])
    pipeline.add(sgie0)

    # Tiler
    tiler = make_element("nvmultistreamtiler", "tiler")
    tiler.set_property("rows", TILED_DISPLAY["rows"])
    tiler.set_property("columns", TILED_DISPLAY["columns"])
    tiler.set_property("width", TILED_DISPLAY["width"])
    tiler.set_property("height", TILED_DISPLAY["height"])
    pipeline.add(tiler)

    # Convert → OSD → Sink
    nvvidconv = make_element("nvvideoconvert", "nvvidconv")
    nvosd = make_element("nvdsosd", "nvosd")
    nvosd.set_property("process-mode", 1)  # GPU OSD
    nvosd.set_property("display-text", 1)
    if nvosd.find_property("display-bbox") is not None:
        nvosd.set_property("display-bbox", 1)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)

    sink = make_element("nveglglessink", "sink0")
    sink.set_property("sync", SINK0["sync"])
    sink.set_property("qos", 0)
    pipeline.add(sink)

    # Link pipeline
    assert Gst.Element.link(streammux, q1)
    assert Gst.Element.link(q1, pgie)
    assert Gst.Element.link(pgie, tracker)
    assert Gst.Element.link(tracker, sgie0)
    assert Gst.Element.link(sgie0, q2)
    assert Gst.Element.link(q2, tiler)
    assert Gst.Element.link(tiler, q3)
    assert Gst.Element.link(q3, nvvidconv)
    assert Gst.Element.link(nvvidconv, q4)
    assert Gst.Element.link(q4, nvosd)
    assert Gst.Element.link(nvosd, q5)
    assert Gst.Element.link(q5, sink)

    # Attach the unified probe AFTER SGIE (so we see faces + tracked parents)
    sgie_src_pad = sgie0.get_static_pad("src")
    if not sgie_src_pad:
        raise RuntimeError("Failed to get sgie0 src pad")
    sgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, unified_src_pad_buffer_probe, None)

    # Bus / loop
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    print("Starting pipeline…")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping pipeline…")
        pipeline.set_state(Gst.State.NULL)

if __name__ == "__main__":
    if STREAMMUX["batch_size"] != len(SOURCES):
        print(f"[WARN] streammux.batch-size={STREAMMUX['batch_size']} != num sources={len(SOURCES)}; fixing.")
        STREAMMUX["batch_size"] = len(SOURCES)
    Gst.init(None)
    main()
