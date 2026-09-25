"""API routes for LiDAR Sequence Replay and WebSocket streaming."""

import asyncio
import json
import logging
import shutil
import uuid
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
from fastapi import APIRouter, File, UploadFile, HTTPException, Query, status, WebSocket, WebSocketDisconnect

from ...config import settings
from ...models.replay_schemas import (
    PlaybackMode,
    ReplayUploadResponse,
    ReplaySessionStatus,
    ReplaySeekRequest,
    ReplayStartRequest,
    ReplayFrameStreamPayload,
    SetPlaybackModeRequest,
)
from ...services.sequence_replay import sequence_replay_service, SequenceIngestError
from ...services.replay_session import replay_session_manager
from ...services.precompute_service import precompute_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["LiDAR Sequence Replay"])


@router.post(
    "/upload-sequence",
    response_model=ReplayUploadResponse,
    summary="Upload and ingest a SemanticKITTI sequence ZIP package",
    description=(
        "Ingests a ZIP package containing `sequences/00/velodyne/*.bin` scans and optional "
        "`sequences/00/predictions/*.label` SalsaNext predictions. Validates binary point-cloud formats, "
        "sorts frames naturally, and initializes a replay session."
    ),
)
async def upload_sequence(
    file: UploadFile = File(..., description="ZIP archive of consecutive SemanticKITTI .bin scans and .label files"),
    session_id: Optional[str] = Query(None, description="Optional custom session identifier"),
    playback_mode: Optional[str] = Query(None, description="Playback mode: 'offline_precomputed_replay' or 'live_processing'"),
):
    """Handle sequence ZIP upload, validation, and session creation."""
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Please upload a ZIP archive (.zip) containing SemanticKITTI sequence data.",
        )

    # Stream upload directly to disk in chunks to avoid high RAM consumption
    temp_dir = settings.DATA_DIR / "temp_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_zip = temp_dir / f"upload_{uuid.uuid4().hex[:8]}.zip"
    try:
        with open(temp_zip, "wb") as f_out:
            shutil.copyfileobj(file.file, f_out)

        file_size = temp_zip.stat().st_size
        if file_size > settings.MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Uploaded ZIP exceeds maximum limit of {settings.MAX_UPLOAD_SIZE_BYTES / (1024 * 1024):.1f} MB.",
            )

        # Scoping session ID for manual uploads to prevent colliding with demo cache
        raw_sid = session_id or Path(file.filename or "uploaded_seq").stem.lower().replace("-", "_")
        if raw_sid.startswith("foveamap_sequence") or raw_sid in ("demo", "sample", "default"):
            effective_sid = f"upload_{raw_sid}"
        else:
            effective_sid = raw_sid

        session_data = sequence_replay_service.extract_and_validate_zip(temp_zip, session_id=effective_sid)
    finally:
        if temp_zip.exists():
            try:
                temp_zip.unlink()
            except Exception:
                pass

    try:
        sid = session_data["session_id"]
        # Manual ZIP uploads must NEVER use precomputed values unless they explicitly include GT labels
        session_data["is_manual_upload"] = True
        session_data["is_demo"] = False
        if session_data.get("has_predictions") or session_data.get("data_mode") == "precomputed_labels":
            session_data["semantic_source"] = "GROUND TRUTH"
            session_data["data_mode"] = "precomputed_labels"
            if not session_data.get("playback_mode"):
                session_data["playback_mode"] = playback_mode or "offline_precomputed_replay"
        else:
            session_data["playback_mode"] = playback_mode or "live_processing"
            session_data["semantic_source"] = "LIVE FAST-FRNET"
            precompute_service.clear_session(sid, delete_disk=True)

        replay_session_manager.delete_session(sid)
        session = replay_session_manager.create_session(session_data)

        return ReplayUploadResponse(
            session_id=session.session_id,
            sequence_name=session.sequence_name,
            total_frames=session.total_frames,
            has_predictions=session.has_predictions,
            has_poses=session.has_poses,
            has_calibration=session.has_calibration,
            data_mode=session.data_mode,
            semantic_source=session.semantic_source,
            playback_mode=session.playback_mode,
            is_manual_upload=session.is_manual_upload,
            frame_filenames=[f["filename"] for f in session.frames_meta],
            status=session.state,
            message=f"Sequence '{session.sequence_name}' ingested successfully with {session.total_frames} consecutive scans (Live Processing Mode).",
        )
    except SequenceIngestError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.error(f"Sequence upload error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to process sequence: {str(e)}")


@router.get(
    "/sessions",
    summary="List all available and discovered replay sessions",
    description="Returns metadata for all sessions that have been loaded, discovered from disk, or available as ZIP archives.",
)
async def list_sessions():
    """List all available replay sessions with metadata."""
    return replay_session_manager.list_available_sessions()


class IngestFolderRequest(BaseModel):
    folder_path: str
    session_id: Optional[str] = None
    playback_mode: Optional[str] = "offline_precomputed_replay"


@router.post(
    "/ingest-folder",
    response_model=ReplayUploadResponse,
    summary="Ingest a sequence directly from a local directory",
)
async def ingest_folder(req: IngestFolderRequest):
    """Ingest a sequence from a local directory path."""
    folder = Path(req.folder_path)
    if not folder.exists() or not folder.is_dir():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Directory '{req.folder_path}' does not exist or is not a directory.",
        )
    sid = req.session_id or folder.name
    try:
        session_data = sequence_replay_service._index_directory(folder, session_id=sid)
        mode = req.playback_mode or "offline_precomputed_replay"
        session_data["playback_mode"] = mode
        if mode == "live_processing":
            precompute_service.clear_session(session_data["session_id"], delete_disk=True)
        replay_session_manager.delete_session(session_data["session_id"])
        session = replay_session_manager.create_session(session_data)
        return ReplayUploadResponse(
            session_id=session.session_id,
            sequence_name=session.sequence_name,
            total_frames=session.total_frames,
            has_predictions=session.has_predictions,
            has_poses=session.has_poses,
            has_calibration=session.has_calibration,
            data_mode=session.data_mode,
            semantic_source=session.semantic_source,
            playback_mode=session.playback_mode,
            frame_filenames=[f["filename"] for f in session.frames_meta],
            status=session.state,
            message=f"Sequence '{session.sequence_name}' ingested successfully with {session.total_frames} consecutive scans.",
        )
    except SequenceIngestError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.error(f"Folder ingest error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to ingest folder: {str(e)}")


class IngestArchiveRequest(BaseModel):
    archive_path: str
    session_id: Optional[str] = None
    playback_mode: Optional[str] = "offline_precomputed_replay"


@router.post(
    "/ingest-archive",
    response_model=ReplayUploadResponse,
    summary="Ingest a sequence ZIP archive directly from local file path",
    description="Extracts and indexes all frames from a local ZIP archive path with zero HTTP upload overhead.",
)
async def ingest_archive(req: IngestArchiveRequest):
    """Ingest a sequence directly from a local archive file path."""
    archive_p = Path(req.archive_path)
    if not archive_p.exists() or not archive_p.is_file():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Archive file '{req.archive_path}' does not exist or is not a file.",
        )
    if not archive_p.name.lower().endswith(".zip"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File '{archive_p.name}' is not a ZIP archive (.zip).",
        )
    raw_sid = req.session_id or archive_p.stem.lower().replace("-", "_")
    if raw_sid.startswith("foveamap_sequence") or raw_sid in ("demo", "sample", "default"):
        effective_sid = f"upload_{raw_sid}"
    else:
        effective_sid = raw_sid
    try:
        session_data = sequence_replay_service.extract_and_validate_zip(archive_p, session_id=effective_sid)
        session_data["is_manual_upload"] = True
        session_data["is_demo"] = False
        session_data["playback_mode"] = "live_processing"
        session_data["semantic_source"] = "LIVE FAST-FRNET"
        precompute_service.clear_session(session_data["session_id"], delete_disk=True)
        replay_session_manager.delete_session(session_data["session_id"])
        session = replay_session_manager.create_session(session_data)
        return ReplayUploadResponse(
            session_id=session.session_id,
            sequence_name=session.sequence_name,
            total_frames=session.total_frames,
            has_predictions=session.has_predictions,
            has_poses=session.has_poses,
            has_calibration=session.has_calibration,
            data_mode=session.data_mode,
            semantic_source=session.semantic_source,
            playback_mode=session.playback_mode,
            is_manual_upload=session.is_manual_upload,
            frame_filenames=[f["filename"] for f in session.frames_meta],
            status=session.state,
            message=f"Sequence '{session.sequence_name}' ingested successfully with {session.total_frames} consecutive scans (Live Processing Mode).",
        )
    except SequenceIngestError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.error(f"Archive ingest error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to ingest archive: {str(e)}")


@router.get(
    "/{session_id}/status",
    response_model=ReplaySessionStatus,
    summary="Get playback status of an active replay session",
)
async def get_replay_status(session_id: str):
    session = replay_session_manager.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Replay session '{session_id}' not found.",
        )
    return session.get_status()


@router.post(
    "/{session_id}/start",
    response_model=ReplaySessionStatus,
    summary="Start or resume sequence replay",
)
async def start_replay(
    session_id: str,
    req: Optional[ReplayStartRequest] = None,
):
    session = replay_session_manager.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Replay session '{session_id}' not found.",
        )
    fps = req.fps if req else None
    playback_mode = req.playback_mode if req else None
    return session.start(fps=fps, playback_mode=playback_mode)


@router.get(
    "/{session_id}/precompute-status",
    summary="Get sequence precomputation status",
)
async def get_precompute_status(session_id: str):
    session = replay_session_manager.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Replay session '{session_id}' not found.",
        )
    prog = precompute_service.get_progress(session_id)
    if prog:
        return prog.model_dump()
    
    is_precomputed = precompute_service.is_sequence_precomputed(session_id, session.total_frames)
    completed_indices = precompute_service.get_completed_frame_indices(session_id)
    n_done = len(completed_indices)
    pct = round((n_done / session.total_frames * 100.0), 2) if session.total_frames > 0 else 0.0
    return {
        "session_id": session_id,
        "sequence_name": session.sequence_name,
        "processed_frames": n_done,
        "total_frames": session.total_frames,
        "percent_complete": pct,
        "is_complete": is_precomputed,
        "is_running": False,
        "elapsed_seconds": 0.0,
        "eta_seconds": 0.0,
        "current_stage": "Ready",
        "failed_count": 0,
    }


@router.post(
    "/{session_id}/cancel-precompute",
    summary="Cancel active sequence precomputation",
)
async def cancel_precompute_endpoint(session_id: str):
    session = replay_session_manager.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Replay session '{session_id}' not found.",
        )
    cancelled = precompute_service.cancel_precompute(session_id)
    return {"session_id": session_id, "cancelled": cancelled}


@router.post(
    "/{session_id}/set-mode",
    response_model=ReplaySessionStatus,
    summary="Set playback mode for session",
)
async def set_playback_mode_endpoint(session_id: str, req: SetPlaybackModeRequest):
    session = replay_session_manager.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Replay session '{session_id}' not found.",
        )
    session.set_playback_mode(req.playback_mode)
    return session.get_status()


@router.post(
    "/{session_id}/pause",
    response_model=ReplaySessionStatus,
    summary="Pause sequence replay at current frame",
)
async def pause_replay(session_id: str):
    session = replay_session_manager.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Replay session '{session_id}' not found.",
        )
    return session.pause()


@router.post(
    "/{session_id}/stop",
    response_model=ReplaySessionStatus,
    summary="Stop sequence replay and rewind to frame 0",
)
async def stop_replay(session_id: str):
    session = replay_session_manager.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Replay session '{session_id}' not found.",
        )
    return session.stop()


@router.delete(
    "/{session_id}",
    summary="Delete a replay session and all its cached data",
)
async def delete_session(session_id: str):
    """Delete a session and clean up all associated caches."""
    precompute_service.clear_session(session_id, delete_disk=True)
    deleted = replay_session_manager.delete_session(session_id)
    return {
        "session_id": session_id,
        "deleted": deleted,
        "message": f"Session '{session_id}' and all cached data have been cleaned up.",
    }


@router.post(
    "/{session_id}/seek",
    response_model=ReplaySessionStatus,
    summary="Seek to a target frame index",
)
async def seek_replay(session_id: str, req: ReplaySeekRequest):
    session = replay_session_manager.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Replay session '{session_id}' not found.",
        )
    return session.seek(req.frame_index)


@router.get(
    "/{session_id}/next-frame",
    response_model=ReplayFrameStreamPayload,
    summary="Advance to next frame and retrieve full perception payload",
)
async def get_next_frame(session_id: str):
    session = replay_session_manager.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Replay session '{session_id}' not found.",
        )
    payload = session.advance_and_get_frame()
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Reached end of sequence.",
        )
    return payload


@router.post(
    "/{session_id}/precompute",
    summary="Start background precomputation for sequence frames to enable 60 FPS playback",
)
async def start_precomputation(session_id: str):
    """Trigger background precomputation of all frames in sequence."""
    session = replay_session_manager.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Replay session '{session_id}' not found.",
        )
    return precompute_service.start_precompute_background(session).model_dump()


@router.post(
    "/{session_id}/cancel-precompute",
    summary="Cancel background precomputation for sequence frames",
)
async def cancel_precomputation(session_id: str):
    """Cancel in-progress background precomputation."""
    session = replay_session_manager.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Replay session '{session_id}' not found.",
        )
    cancelled = precompute_service.cancel_precompute(session_id)
    return {"session_id": session_id, "cancelled": cancelled}


@router.get(
    "/{session_id}/global-cells/{cell_key:path}",
    summary="Look up a cell by key from the session-global fused map",
    description=(
        "Returns the most recent matching cell from the per-session global fusion "
        "map.  Used by the frontend to keep a clicked cell inspectable even when "
        "the most recent frame's downsampled cells_sample has dropped it."
    ),
)
async def get_replay_global_cell(session_id: str, cell_key: str):
    session = replay_session_manager.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Replay session '{session_id}' not found.",
        )
    cell = session.get_global_cell(cell_key)
    if cell is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cell '{cell_key}' not in global map for session '{session_id}'.",
        )
    return cell


@router.get(
    "/{session_id}/global-cells",
    summary="List cells in the session-global fused map (optionally filtered)",
)
async def list_replay_global_cells(
    session_id: str,
    min_x: Optional[float] = None,
    max_x: Optional[float] = None,
    min_y: Optional[float] = None,
    max_y: Optional[float] = None,
    level: Optional[str] = None,
    limit: int = 5000,
):
    session = replay_session_manager.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Replay session '{session_id}' not found.",
        )
    return session.list_global_cells(
        min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y, level=level, limit=limit
    )


@router.websocket("/{session_id}/stream")
async def websocket_stream_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket streaming endpoint for continuous frame-by-playing replay with backpressure.

    On every (re)connect we immediately emit a ``replay_status`` event so the
    client can synchronise its UI without having to wait for the next frame.
    """
    session = replay_session_manager.get_session(session_id)
    if not session:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Session not found")
        return

    await websocket.accept()
    logger.info(f"WebSocket client connected to replay session {session_id}")

    # Send an initial status snapshot so the client can recover its UI state
    # immediately after a reconnect.
    try:
        await websocket.send_text(json.dumps({
            "event": "replay_status",
            "status": session.get_status().model_dump(),
            "resync": True,
        }))
    except Exception:
        pass

    # Ensure precomputed frames are loaded in RAM for maximum 60 FPS throughput
    try:
        precompute_service.preload_all_frames(session_id, session.total_frames)
    except Exception:
        pass

    stop_event = asyncio.Event()

    async def receive_control_loop():
        try:
            while not stop_event.is_set():
                msg = await websocket.receive_text()
                _handle_ws_control_message(session, msg)
                # Send status update upon control message
                try:
                    await websocket.send_text(json.dumps({
                        "event": "status",
                        "status": session.get_status().model_dump(),
                    }))
                except Exception:
                    pass
        except WebSocketDisconnect:
            stop_event.set()
        except Exception as e:
            logger.debug(f"WS receiver ended: {e}")
            stop_event.set()

    receiver_task = asyncio.create_task(receive_control_loop())

    try:
        while not stop_event.is_set():
            if session.state == "playing" and session.current_frame_index < session.total_frames:
                frame_start = asyncio.get_event_loop().time()

                # Precomputed zero-overhead streaming is STRICTLY restricted to built-in demo sequences.
                # Manual ZIP uploads must NEVER use precomputed values and must ALWAYS compute live.
                use_demo_precomputed = (
                    session.is_demo
                    and not getattr(session, "is_manual_upload", False)
                    and session.playback_mode == PlaybackMode.OFFLINE_PRECOMPUTED_REPLAY
                )
                raw_json = None
                if use_demo_precomputed:
                    raw_json = precompute_service.get_raw_frame_json(session.session_id, session.current_frame_index)

                # Fast zero-overhead 60 FPS streaming path: direct raw JSON transmission from RAM (Demo Only)
                if raw_json is not None:
                    session.current_frame_index += 1
                    is_done = session.current_frame_index >= session.total_frames
                    if is_done:
                        session.state = "completed"
                    await websocket.send_text(raw_json)
                    if is_done:
                        try:
                            await websocket.send_text(json.dumps({
                                "event": "status",
                                "status": session.get_status().model_dump(),
                            }))
                        except Exception:
                            pass
                else:
                    payload = await asyncio.to_thread(session.advance_and_get_frame)
                    if payload:
                        await websocket.send_text(payload.model_dump_json(exclude_none=True, exclude_defaults=True))
                        if session.current_frame_index >= session.total_frames:
                            try:
                                await websocket.send_text(json.dumps({
                                    "event": "status",
                                    "status": session.get_status().model_dump(),
                                }))
                            except Exception:
                                pass

                target_interval = 1.0 / max(0.5, session.fps)
                elapsed = asyncio.get_event_loop().time() - frame_start
                sleep_delay = max(0.0, target_interval - elapsed)
                if sleep_delay > 0:
                    await asyncio.sleep(sleep_delay)
            else:
                await asyncio.sleep(0.05)

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected from replay session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error in session {session_id}: {e}", exc_info=True)
    finally:
        stop_event.set()
        receiver_task.cancel()
        try:
            await receiver_task
        except (asyncio.CancelledError, Exception):
            pass


def _handle_ws_control_message(session, msg: str):
    """Parse and execute incoming client control JSON over WebSocket."""
    try:
        data = json.loads(msg)
        action = data.get("action")
        if action == "play":
            fps = data.get("fps")
            session.start(fps=fps)
        elif action == "pause":
            session.pause()
        elif action == "stop":
            session.stop()
        elif action == "seek":
            idx = data.get("frame_index", 0)
            session.seek(idx)
        elif action == "set_fps":
            fps = data.get("fps", 10.0)
            session.fps = max(0.2, min(60.0, float(fps)))
    except Exception as e:
        logger.warning(f"Error handling WebSocket message: {e}")
