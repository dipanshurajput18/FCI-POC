"""FastAPI app: REST for control, WebSocket for the live overlay video and results.

Run:  .venv\\Scripts\\python -m uvicorn server.app:app --port 8000
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from grain import calibration

from .camera import CvSource, ImageSource, SimSource, list_usb_cameras
from .dual import DualEngine, DualSimSource, PhonePairSource
from .engine import Engine

WEB = Path(__file__).resolve().parent.parent / "web"
app = FastAPI(title="FCI AI Grain Analyser", version="0.1.0")
engine = Engine()
dual = DualEngine(engine)
app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/config")
def config():
    return engine.config()


@app.get("/api/cameras")
def cameras():
    return {"usb": list_usb_cameras()}


@app.post("/api/source")
async def set_source(req: Request):
    d = await req.json()
    kind = d.get("kind", "sim")
    commodity = d.get("commodity") or engine.commodity
    try:
        if kind == "sim":
            preset = d.get("preset") or ("mixed" if commodity == "rice" else "M-30")
            engine.set_source(SimSource(commodity, preset, seed=int(d.get("seed", 7))), commodity, preset)
        elif kind in ("usb", "url", "file"):
            value = str(d.get("value", "0")).strip()
            src = await asyncio.to_thread(CvSource, value, float(d.get("marker_mm", 20)),
                                          still_url=(d.get("still_url") or None))
            engine.set_source(src, commodity)
        else:
            raise HTTPException(400, "unknown source kind")
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    return engine.config()


@app.post("/api/upload")
async def upload(req: Request):
    data = await req.body()
    if not data or len(data) > 40_000_000:
        raise HTTPException(400, "empty or too large")
    try:
        engine.set_source(ImageSource(data=data, marker_mm=float(req.query_params.get("marker_mm", 20))))
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    return engine.config()


@app.post("/api/settings")
async def settings(req: Request):
    engine.update_settings(await req.json())
    return engine.config()


@app.post("/api/capture")
def capture():
    engine.capture()
    return {"ok": True}


@app.post("/api/new-sample")
def new_sample():
    engine.new_sample()
    return {"ok": True}


@app.post("/api/shake")
def shake():
    return {"ok": engine.shake()}


@app.get("/api/result")
def result():
    return engine.result()


@app.get("/api/shot/{n}.jpg")
def shot(n: int, raw: int = 0):
    with engine.lock:
        for s in engine.shots:
            if s["n"] == n:
                return Response(s["raw" if raw else "jpeg"], media_type="image/jpeg",
                                headers={"Cache-Control": "no-store"})
    raise HTTPException(404)


@app.get("/api/calibration-sheet.png")
def calibration_sheet(marker_mm: float = 20.0, black: int = 1):
    img = calibration.calibration_sheet(dpi=300, marker_mm=marker_mm, black_tray=bool(black))
    ok, png = cv2.imencode(".png", img)
    return Response(png.tobytes(), media_type="image/png",
                    headers={"Content-Disposition": "inline; filename=calibration-sheet-A4.png"})


async def _stream(sock: WebSocket, eng) -> None:
    await sock.accept()
    eng.clients += 1
    live_seen, res_seen, err_seen = -1, -1, None
    try:
        while True:
            if eng.live_ver != live_seen and eng.live_jpeg is not None:
                live_seen = eng.live_ver
                await sock.send_bytes(eng.live_jpeg)
                await sock.send_text(json.dumps({"type": "live", **eng.live}))
            if eng.result_ver != res_seen:
                res_seen = eng.result_ver
                await sock.send_text(json.dumps({"type": "result", **eng.result()}))
            if eng.error != err_seen:
                err_seen = eng.error
                await sock.send_text(json.dumps({"type": "error", "error": eng.error}))
            await asyncio.sleep(0.04)
    except (WebSocketDisconnect, RuntimeError):
        return
    finally:
        eng.clients -= 1


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await _stream(sock, engine)


# ---------------------------------------------------------------- dual-camera glass rig
@app.get("/dual")
def dual_page():
    return FileResponse(WEB / "dual.html")


@app.get("/rig3d")
def rig3d():
    return FileResponse(WEB.parent / "docs" / "rig_3d_dual_camera.html")


@app.get("/api/dual/config")
def dual_config():
    return dual.config()


@app.post("/api/dual/source")
async def dual_source(req: Request):
    d = await req.json()
    try:
        if d.get("kind", "sim") == "sim":
            preset = d.get("preset") or "mixed"
            dual.set_source(DualSimSource(preset, seed=int(d.get("seed", 7))), preset)
        else:
            top, bottom = str(d.get("top", "")).strip(), str(d.get("bottom", "")).strip()
            if not top or not bottom:
                raise HTTPException(400, "Enter both the top and the bottom phone address")
            src = await asyncio.to_thread(PhonePairSource, top, bottom, float(d.get("marker_mm", 20)))
            dual.set_source(src)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    return dual.config()


@app.post("/api/dual/settings")
async def dual_settings(req: Request):
    d = await req.json()
    shared = {k: v for k, v in d.items() if k in ("lot_id", "officer", "grade", "sample_g", "classifier")}
    if shared:
        engine.update_settings(shared)
    dual.update(d)
    return dual.config()


@app.post("/api/dual/capture")
def dual_capture():
    dual.capture()
    return {"ok": True}


@app.post("/api/dual/new-sample")
def dual_new_sample():
    dual.new_sample()
    return {"ok": True}


@app.post("/api/dual/shake")
def dual_shake():
    dual.source.shake()
    return {"ok": True}


@app.get("/api/dual/result")
def dual_result():
    return dual.result()


@app.get("/api/dual/shot/{n}/{side}.jpg")
def dual_shot(n: int, side: str, raw: int = 0):
    if side not in ("top", "bottom"):
        raise HTTPException(404)
    key = ("raw_" if raw else "jpeg_") + side[0]
    with dual.lock:
        for s in dual.shots:
            if s["n"] == n:
                return Response(s[key], media_type="image/jpeg", headers={"Cache-Control": "no-store"})
    raise HTTPException(404)


@app.websocket("/ws/dual")
async def ws_dual(sock: WebSocket):
    await _stream(sock, dual)
