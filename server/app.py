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
from .engine import Engine

WEB = Path(__file__).resolve().parent.parent / "web"
app = FastAPI(title="FCI AI Grain Analyser", version="0.1.0")
engine = Engine()
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


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    live_seen, res_seen, err_seen = -1, -1, None
    try:
        while True:
            if engine.live_ver != live_seen and engine.live_jpeg is not None:
                live_seen = engine.live_ver
                await sock.send_bytes(engine.live_jpeg)
                await sock.send_text(json.dumps({"type": "live", **engine.live}))
            if engine.result_ver != res_seen:
                res_seen = engine.result_ver
                await sock.send_text(json.dumps({"type": "result", **engine.result()}))
            if engine.error != err_seen:
                err_seen = engine.error
                await sock.send_text(json.dumps({"type": "error", "error": engine.error}))
            await asyncio.sleep(0.04)
    except (WebSocketDisconnect, RuntimeError):
        return
