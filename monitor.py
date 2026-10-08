"""
The camera monitor: owns the webcam and runs the capture loop in a background thread.

HOW IT WORKS
    Start pressed  ->  a background thread loads the face model, THEN opens the camera
                       (so a missing model never switches the camera on)
    loop:  read a frame -> find the face -> measure the eyes -> assess fatigue -> measure lighting -> draw outlines ->
           keep ONLY the newest frame (as a small JPEG in memory) -> wait for the next frame time
    Stop pressed   ->  the thread ends and the camera is released (its light goes off)

Phase 4 added face detection, Phase 5 eye measurements and calibration, Phase 6 the fatigue assessment,
Phase 7 posture (head position from the face, shoulders from an optional pose model on every 4th frame),
Phase 8 attention (face visible + head direction, from the same head measurements), and Phase 9 session
tracking (duration, active time, breaks -- see computer_vision/session.py).

PRIVACY SAFEGUARDS (project brief, section 19)
    * The camera never starts by itself. It starts only on an explicit Start request.
    * Frames are never written to disk. Only the single newest frame is kept in memory,
      for the live preview, and it is deleted when monitoring stops.
    * Only the account that pressed Start can see the preview or the numbers.
    * The camera switches off on: Stop, logout, server shutdown, camera errors, and when
      the dashboard has not contacted the server for CAMERA_IDLE_TIMEOUT_S seconds.

One computer has one webcam, so only ONE account can be monitored at a time.
"""

import threading
import time
from datetime import datetime, timezone

import cv2

from computer_vision import lighting
from computer_vision.calibrator import Calibrator
from computer_vision.camera import Camera, CameraError
from computer_vision.drawing import draw_face_overlay
from computer_vision.attention import NOBODY_VISIBLE, AttentionAssessor, baseline_supports_attention
from computer_vision.session import SessionTracker
from ml.alerts import AlertManager
from ml.dhri import DHRIAssessor
from computer_vision.eye_measurement import EyeMeasurer
from computer_vision.pose_landmarks import PoseDetector
from computer_vision.posture import PostureAssessor, measure_posture
from computer_vision.eye_tracker import EyeStateTracker
from computer_vision.face_landmarks import (
    FACE_MESSAGES,
    FACE_NONE,
    FaceDetector,
    ModelError,
    StatusDebouncer,
)
from ml.fatigue import FatigueAssessor
from utils import config

STOPPED = "stopped"
STARTING = "starting"
RUNNING = "running"
ERROR = "error"


class CameraBusyError(Exception):
    """Another account is already using the camera."""


class CameraMonitor:
    def __init__(self, camera_factory=None, idle_timeout_s=None, max_read_failures=None,
                 detector_factory=None, calibrator_factory=None, assessor_factory=None,
                 pose_detector_factory=None):
        # The factories let the tests plug in a fake camera / fake face detector / short calibration.
        self._camera_factory = camera_factory or Camera
        self._detector_factory = detector_factory or FaceDetector
        self._calibrator_factory = calibrator_factory or Calibrator
        self._assessor_factory = assessor_factory or FatigueAssessor.from_files
        self._pose_detector_factory = pose_detector_factory or PoseDetector
        self._idle_timeout_s = config.CAMERA_IDLE_TIMEOUT_S if idle_timeout_s is None else idle_timeout_s
        self._max_read_failures = (
            config.CAMERA_MAX_READ_FAILURES if max_read_failures is None else max_read_failures
        )
        self._lock = threading.Lock()          # protects everything below
        self._thread = None
        self._stop_event = threading.Event()   # a new one is made for every run
        # Unlike everything _reset() clears, the finished session result must SURVIVE the _reset() that
        # happens when the run's thread finishes, because the web route collects it right after Stop returns
        # (see take_session_result). It is set once, from _capture_loop's own finally block, before _run's
        # finally block calls _reset().
        self._session_result = None
        self._session_result_owner = None
        self._reset()

    def _reset(self):
        """Forget everything about the previous run (caller holds the lock, or __init__ is running)."""
        self._state = STOPPED
        self._message = ""
        self._owner_id = None
        self._latest_jpeg = None
        self._frame_id = 0
        self._frame_size = None
        self._brightness = None
        self._fps = 0.0
        self._face_status = FACE_NONE
        self._faces_found = 0
        self._processing_ms = 0.0
        self._baseline = None               # the user's eye calibration, or None if not calibrated
        self._baseline_version = 0          # increases whenever the baseline changes
        self._eye = None                    # newest eye features (a snapshot dict)
        self._fatigue = None                # newest fatigue assessment (a snapshot dict)
        self._posture = None                # newest posture assessment (a snapshot dict)
        self._attention = None              # newest attention assessment (a snapshot dict)
        self._session = None                # running totals for THIS session (a SessionSummary, live while it runs)
        self._dhri = None                   # newest DHRI assessment (a snapshot dict)
        self._alerts = None                 # newest alerts/recommendations snapshot dict
        self._posture_baseline = None       # the user's posture calibration, or None
        self._calibration_status = None     # newest calibration progress (a snapshot dict)
        self._calibration_result = None     # finished result waiting to be collected by the web route
        self._calibration_requested = False
        self._calibration_cancel = False
        self._started_at = None
        self._started_at_wall = None        # wall-clock time the camera started (for saving session rows)
        self._last_heartbeat = time.monotonic()

    # ------------------------------------------------------------------
    # Called by the web routes
    # ------------------------------------------------------------------
    def start(self, user_id, baseline=None, posture_baseline=None):
        """
        Switch the camera on for this user. Waits briefly so the caller gets a real result.
        baseline: the user's saved eye calibration (None = not calibrated yet).
        posture_baseline: the user's saved posture calibration (None = not calibrated for posture).
        """
        with self._lock:
            if self._state in (STARTING, RUNNING):
                if self._owner_id != user_id:
                    raise CameraBusyError()
                return self._status_locked(user_id)     # already running for this user
            self._reset()
            self._state = STARTING
            self._owner_id = user_id
            self._baseline = baseline
            self._baseline_version = 1
            self._posture_baseline = posture_baseline
            stop_event = self._stop_event = threading.Event()
            ready_event = threading.Event()
            self._thread = threading.Thread(
                target=self._run, args=(stop_event, ready_event), daemon=True, name="camera-monitor"
            )
            self._thread.start()

        ready_event.wait(timeout=config.CAMERA_OPEN_WAIT_S)
        return self.get_status(user_id)

    def stop(self):
        """Switch the camera off and wait until it has really been released."""
        with self._lock:
            thread = self._thread
            self._stop_event.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5)
        with self._lock:
            if self._state == ERROR:      # pressing Stop also clears an old error message
                self._reset()

    def stop_if_owner(self, user_id):
        """Used at logout: stop the camera only if this user is the one using it."""
        with self._lock:
            owns_camera = self._state in (STARTING, RUNNING) and self._owner_id == user_id
        if owns_camera:
            self.stop()
        return owns_camera

    def touch(self, user_id):
        """'The dashboard is still open.' Resets the idle timer (only for the owner)."""
        with self._lock:
            if self._owner_id == user_id:
                self._last_heartbeat = time.monotonic()

    # ---- calibration (the web routes call these; the capture thread does the work) ----
    def begin_calibration(self, user_id):
        """Start the calibration steps. Returns False if this user is not using the camera."""
        with self._lock:
            if self._state not in (STARTING, RUNNING) or self._owner_id != user_id:
                return False
            self._calibration_requested = True
            self._calibration_result = None
            self._calibration_status = self._calibrator_factory().status()   # "ready", 3 s left
            return True

    def cancel_calibration(self, user_id):
        with self._lock:
            status = self._calibration_status
            if self._owner_id == user_id and status and status["active"]:
                self._calibration_cancel = True

    def get_calibration_status(self, user_id):
        """Progress of the calibration, or None if there is none for this user."""
        with self._lock:
            if self._owner_id != user_id or self._state not in (STARTING, RUNNING):
                return None
            return dict(self._calibration_status) if self._calibration_status else None

    def take_calibration_result(self, user_id):
        """Return the finished calibration result ONCE (so the caller can save it), else None."""
        with self._lock:
            if self._owner_id != user_id:
                return None
            result, self._calibration_result = self._calibration_result, None
            return result

    def get_active_owner(self):
        """
        The user_id currently using the camera, or None if nobody is. Used by the server's shutdown
        handler to learn WHOSE session to save once stop() finishes (stop() itself does not know
        which caller is asking, since there is no HTTP request during a server shutdown).
        """
        with self._lock:
            return self._owner_id if self._state in (STARTING, RUNNING) else None

    def take_session_result(self, user_id):
        """
        Return the just-finished session's summary ONCE (so the caller can save it to the database), else None.
        Unlike the other take_*() methods this is checked against the OWNER OF THE FINISHED SESSION, not the
        current owner, because by the time a session ends (Stop, an error, or the idle timeout) the camera has
        already gone back to being unowned. Call this once, right after stop() returns.
        """
        with self._lock:
            if self._session_result_owner != user_id:
                return None
            result = self._session_result
            self._session_result = self._session_result_owner = None
            return result

    def get_status(self, user_id=None):
        with self._lock:
            return self._status_locked(user_id)

    def get_frame(self, user_id):
        """Return (frame_id, jpeg_bytes) of the newest frame, only for the account using the camera."""
        with self._lock:
            if self._state != RUNNING or self._owner_id != user_id or self._latest_jpeg is None:
                return None
            return self._frame_id, self._latest_jpeg

    def _status_locked(self, user_id):
        active = self._state in (STARTING, RUNNING)
        mine = active and user_id is not None and self._owner_id == user_id
        status = {
            "state": self._state,
            "camera_active": active,
            "owned_by_me": mine,
            "busy_other_user": active and not mine,   # deliberately does not say WHO
            "message": "" if (active and not mine) else self._message,
        }
        if mine and self._state == RUNNING:
            lighting_state, lighting_message = lighting.classify_lighting(self._brightness)
            width, height = self._frame_size if self._frame_size else (None, None)
            status.update(
                fps=round(self._fps, 1),
                frame_width=width,
                frame_height=height,
                brightness=None if self._brightness is None else round(self._brightness, 1),
                lighting=lighting_state,
                lighting_message=lighting_message,
                face_status=self._face_status,
                face_message=FACE_MESSAGES[self._face_status],
                faces_found=self._faces_found,
                processing_ms=round(self._processing_ms, 1),
                uptime_s=int(time.monotonic() - self._started_at),
                calibrated=self._baseline is not None,
                posture_calibrated=self._posture_baseline is not None,
                attention_calibrated=baseline_supports_attention(self._posture_baseline),
                calibrating=bool(self._calibration_status and self._calibration_status["active"]),
                eye=dict(self._eye) if self._eye else {},
                fatigue=dict(self._fatigue) if self._fatigue else {},
                posture=dict(self._posture) if self._posture else {},
                attention=dict(self._attention) if self._attention else {},
                session={} if self._session is None else {
                    "duration_s": self._session.duration_s, "active_duration_s": self._session.active_duration_s,
                    "break_count": self._session.break_count, "break_duration_s": self._session.break_duration_s,
                    "continuous_active_s": self._session.continuous_active_s,
                },
                dhri=dict(self._dhri) if self._dhri else {},
                alerts=dict(self._alerts) if self._alerts else {},
            )
            # Low frame rate makes fast blinks easy to miss (ignore the first seconds: warm-up)
            low_fps = status["uptime_s"] >= config.FPS_WARNING_WARMUP_S and 0 < self._fps < config.MIN_RELIABLE_FPS
            status["fps_warning"] = (
                "The frame rate is low, so fast blinks may be missed. Close other programs "
                "or improve the lighting." if low_fps else ""
            )
        return status

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------
    def _run(self, stop_event, ready_event):
        final_state, final_message = STOPPED, ""
        camera = None
        detector = None
        pose_detector = None
        try:
            detector = self._detector_factory()
            detector.load()                              # raises ModelError BEFORE the camera is touched
            assessor = self._assessor_factory()          # never fails: a bad model file falls back to the rules
            pose_detector, pose_message = self._load_pose_detector()      # optional: never stops the camera
            with self._lock:
                if self._stop_event is stop_event:
                    self._fatigue = assessor.idle_snapshot()
                    self._posture = PostureAssessor(self._posture_baseline).idle_snapshot()
                    self._attention = AttentionAssessor(self._posture_baseline).idle_snapshot()
                    self._dhri = DHRIAssessor().idle_snapshot()
                    self._alerts = AlertManager().idle_snapshot()
            camera = self._camera_factory()
            camera.open()                                # raises CameraError if there is no camera
            with self._lock:
                if self._stop_event is stop_event:
                    self._state = RUNNING
                    self._started_at = time.monotonic()
                    self._started_at_wall = datetime.now(timezone.utc)
            ready_event.set()
            final_state, final_message = self._capture_loop(camera, detector, assessor, pose_detector, pose_message, stop_event)
        except (CameraError, ModelError) as error:
            final_state, final_message = ERROR, str(error)
        except Exception as error:                       # never let the thread die silently
            final_state, final_message = ERROR, f"Unexpected camera problem: {error}"
        finally:
            if camera is not None:
                camera.release()                         # camera light goes off
            if detector is not None:
                detector.close()
            if pose_detector is not None:
                pose_detector.close()
            with self._lock:
                if self._stop_event is stop_event:       # ignore if a newer run has begun
                    self._reset()
                    self._state = final_state
                    self._message = final_message
            ready_event.set()

    def _load_pose_detector(self):
        """The shoulder model is optional. Returns (detector or None, message about it)."""
        detector = None
        try:
            detector = self._pose_detector_factory()
            detector.load()
            return detector, ""
        except ModelError as error:
            if detector is not None:
                detector.close()
            return None, f"{error} Posture is judged from your head position only."

    def _capture_loop(self, camera, detector, assessor, pose_detector, pose_message, stop_event):
        """Returns (final_state, message) when the loop ends."""
        frame_interval = 1.0 / config.TARGET_FPS
        failed_reads = 0
        smoothed_brightness = None
        fps = 0.0
        processing_ms = 0.0
        run_start = previous_frame_time = time.monotonic()
        face_debouncer = StatusDebouncer()
        calibrator = None
        measurer = EyeMeasurer()                       # decides whether the eyes can be trusted in each frame
        frame_index = 0
        last_pose, last_pose_time = None, 0.0          # newest shoulder result (the pose model runs every Nth frame)
        session_tracker = SessionTracker()             # Phase 9: session duration, active time, breaks
        last_t = 0.0                                   # updated every frame; used to finalize the session below
        with self._lock:
            tracker = EyeStateTracker(self._baseline)
            seen_baseline_version = self._baseline_version
            posture_assessor = PostureAssessor(self._posture_baseline)
            attention_assessor = AttentionAssessor(self._posture_baseline)
            dhri_assessor = DHRIAssessor()
            alert_manager = AlertManager()

        while not stop_event.is_set():
            loop_start = time.monotonic()

            if loop_start - self._last_heartbeat > self._idle_timeout_s:
                self._finish_session(session_tracker, last_t, stop_event)
                return STOPPED, "Monitoring stopped automatically because the dashboard was no longer open."

            frame = camera.read()
            if frame is None:
                failed_reads += 1
                if failed_reads >= self._max_read_failures:
                    self._finish_session(session_tracker, last_t, stop_event)
                    return ERROR, "Camera stopped sending images. Is it unplugged or in use by another app?"
                stop_event.wait(0.05)
                continue
            failed_reads = 0

            # ---- Phase 4: find the face and its landmarks ----
            detect_start = time.monotonic()
            face_result = detector.detect(frame, int((detect_start - run_start) * 1000))
            detect_ms = (time.monotonic() - detect_start) * 1000
            processing_ms = detect_ms if processing_ms == 0.0 else 0.9 * processing_ms + 0.1 * detect_ms
            shown_face_status = face_debouncer.update(face_result.status)

            # ---- Phase 5: eye measurements and calibration ----
            now_s = loop_start - run_start
            measurement = measurer.measure(now_s, frame, face_result)   # None values = eyes not trusted right now
            raw_brightness = measurement.brightness
            ear_values = measurement.values
            ear = None if ear_values is None else ear_values[2]        # average of both eyes
            last_t = now_s
            # ---- Phase 9: session/exposure tracking ----
            session_tracker.update(now_s, present=measurement.reason not in NOBODY_VISIBLE)

            # ---- Phase 7: posture metrics (only from a steady, trusted picture) ----
            frame_height, frame_width = frame.shape[:2]
            posture_metrics = None
            if measurement.values is not None:
                if pose_detector is not None and frame_index % config.POSE_EVERY_N_FRAMES == 0:
                    last_pose, last_pose_time = pose_detector.detect(frame, int(now_s * 1000)), now_s
                fresh = last_pose is not None and now_s - last_pose_time <= config.POSE_RESULT_MAX_AGE_S
                posture_metrics = measure_posture(face_result.landmarks, face_result.bbox,
                                                  last_pose if fresh else None, frame_width, frame_height)
            frame_index += 1

            with self._lock:                                           # pick up requests from the web routes
                if self._baseline_version != seen_baseline_version:
                    tracker = EyeStateTracker(self._baseline)
                    seen_baseline_version = self._baseline_version
                start_calibration = self._calibration_requested
                cancel_calibration = self._calibration_cancel
                self._calibration_requested = self._calibration_cancel = False
            if start_calibration:
                calibrator = self._calibrator_factory()
            if cancel_calibration and calibrator is not None:
                calibrator.cancel()

            calibration_status = None
            if calibrator is not None:
                # While calibrating, normal blink counting is paused so the "close your eyes"
                # step is not counted as a long closure.
                calibration_status = calibrator.update(now_s, ear, posture_metrics)
                eye_features = EyeStateTracker(None).update(now_s, ear)
                eye_features["calibrated"] = tracker.baseline is not None
                eye_features["paused"] = True
                if calibrator.finished:
                    result, calibrator = calibrator.result, None
                    with self._lock:
                        self._calibration_result = result
                        if result["ok"]:
                            self._baseline = result["baseline"]
                            self._baseline_version += 1
                            seen_baseline_version = self._baseline_version
                    if result["ok"]:
                        tracker = EyeStateTracker(result["baseline"])
                        with self._lock:
                            self._posture_baseline = result.get("posture_baseline")
                        posture_assessor = PostureAssessor(result.get("posture_baseline"))
                        attention_assessor = AttentionAssessor(result.get("posture_baseline"))
            else:
                eye_features = tracker.update(now_s, ear)
            eye_features["not_measured_reason"] = measurement.reason
            eye_features["ear_left"] = None if ear_values is None else round(ear_values[0], 4)
            eye_features["ear_right"] = None if ear_values is None else round(ear_values[1], 4)

            # ---- Phase 6: fatigue assessment from the eye features ----
            fatigue = assessor.update(now_s, eye_features)
            calibrating_now = calibration_status is not None and calibration_status["active"]
            posture = posture_assessor.update(now_s, posture_metrics, calibrating=calibrating_now)
            posture["shoulder_model_message"] = pose_message
            # ---- Phase 8: attention (is the face visible, does the head point at the screen) ----
            attention = attention_assessor.update(now_s, measurement.reason, posture_metrics, calibrating=calibrating_now)
            # ---- Phase 9/10: session totals so far, and the combined DHRI risk score ----
            session_summary = session_tracker.summary(now_s)
            dhri = dhri_assessor.update(now_s, fatigue, posture, attention, session_summary.continuous_active_s)
            # ---- Phase 11: alerts (cooldown-gated) and recommendations (always current) ----
            alerts = alert_manager.update(now_s, fatigue, posture, attention, dhri, session_summary.continuous_active_s)

            smoothed_brightness = lighting.smooth(smoothed_brightness, raw_brightness)

            now = time.monotonic()
            if now > previous_frame_time:
                instant_fps = 1.0 / (now - previous_frame_time)
                fps = instant_fps if fps == 0.0 else 0.9 * fps + 0.1 * instant_fps
            previous_frame_time = now

            draw_face_overlay(frame, face_result, eyes_closed=bool(eye_features.get("eyes_closed")))

            encoded, buffer = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, config.PREVIEW_JPEG_QUALITY]
            )
            if encoded:
                height, width = frame.shape[:2]
                with self._lock:
                    if self._stop_event is stop_event:
                        self._latest_jpeg = buffer.tobytes()
                        self._frame_id += 1
                        self._frame_size = (width, height)
                        self._brightness = smoothed_brightness
                        self._fps = fps
                        self._face_status = shown_face_status
                        self._faces_found = face_result.faces_found
                        self._processing_ms = processing_ms
                        self._eye = eye_features
                        self._fatigue = fatigue
                        self._posture = posture
                        self._attention = attention
                        self._session = session_summary
                        self._dhri = dhri
                        self._alerts = alerts
                        if calibration_status is not None:
                            self._calibration_status = calibration_status

            spare_time = frame_interval - (time.monotonic() - loop_start)
            if spare_time > 0:
                stop_event.wait(spare_time)              # returns immediately if Stop is pressed

        self._finish_session(session_tracker, last_t, stop_event)
        return STOPPED, ""

    def _finish_session(self, session_tracker, last_t, stop_event):
        """
        Called exactly once, from every exit point of _capture_loop (Stop, idle timeout, camera lost, or an
        unexpected error further up in _run). Closes out any break still in progress and stores the final
        summary where the web route can collect it with take_session_result() -- see that method's docstring
        for why this cannot simply reuse self._owner_id.
        """
        session_tracker.finalize(last_t)
        summary = session_tracker.summary(last_t)
        with self._lock:
            if self._stop_event is not stop_event:        # a newer run has already begun; this one is stale
                return
            self._session = summary
            self._session_result = {
                "start_time": self._started_at_wall,
                "duration_s": summary.duration_s,
                "active_duration_s": summary.active_duration_s,
                "break_count": summary.break_count,
                "break_duration_s": summary.break_duration_s,
                "fatigue_events": dict(self._fatigue["events"]) if self._fatigue else {"warning": 0, "fatigued": 0},
                "posture_events": dict(self._posture["events"]) if self._posture else {"warning": 0, "poor": 0},
                "attention_events": dict(self._attention["events"]) if self._attention else {"distracted": 0, "no_face": 0},
                "dhri_events": dict(self._dhri["events"]) if self._dhri else {"moderate": 0, "high": 0},
                "alert_count": self._alerts["total_count"] if self._alerts else 0,
            }
            self._session_result_owner = self._owner_id


# The one shared monitor used by the web app (one computer = one webcam).
monitor = CameraMonitor()
