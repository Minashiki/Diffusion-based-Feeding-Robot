"""Read-only rollout display, isolated from the physics process."""

import multiprocessing
import queue
import time

import mujoco
import numpy as np

from feedingrobot.sim.viewer import passive_viewer


def _display(model, initial_state, frames, messages, ready, stop):
    data = mujoco.MjData(model)
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    mujoco.mj_setState(model, data, initial_state, state_spec)
    mujoco.mj_forward(model, data)
    displayed = 0
    try:
        with passive_viewer(model, data) as viewer:
            plate = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "plate_frame")
            mouth = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "mouth_entry")
            viewer.cam.lookat[:] = (data.site_xpos[plate] + data.site_xpos[mouth]) / 2
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.1, 135., -30.
            viewer.sync(state_only=True)
            messages.put(dict(status="running"))
            ready.set()
            while not stop.is_set() and viewer.is_running():
                try:
                    state = frames.get(timeout=.05)
                except queue.Empty:
                    continue
                mujoco.mj_setState(model, data, state, state_spec)
                mujoco.mj_forward(model, data)
                viewer.sync(state_only=True)
                displayed += 1
            messages.put(dict(status="closed", reason="episode_finished" if stop.is_set() else "window_closed",
                              displayed_frames=displayed))
    except Exception as exc:
        messages.put(dict(status="unavailable", reason=f"{type(exc).__name__}: {exc}"))
    finally:
        ready.set()


class ObserverViewer:
    """A bounded queue drops display frames rather than delaying physical ticks."""

    def __init__(self, model, data):
        self.model, self.data = model, data
        self.state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self.state_size = mujoco.mj_stateSize(model, self.state_spec)
        self.process = None
        self.last_frame = -np.inf
        self.sent_frames = 0
        self.closed = False
        self.report = dict(status="starting", max_fps=30, physics_isolated=True)

    def _state(self):
        state = np.empty(self.state_size)
        mujoco.mj_getState(self.model, self.data, state, self.state_spec)
        return state

    def start(self):
        try:
            context = multiprocessing.get_context("spawn")
            self.frames, self.messages = context.Queue(maxsize=1), context.Queue()
            self.ready, self.stop = context.Event(), context.Event()
            self.process = context.Process(target=_display,
                args=(self.model, self._state(), self.frames, self.messages, self.ready, self.stop), daemon=True)
            self.process.start()
            if not self.ready.wait(timeout=10.):
                self.report.update(status="unavailable", reason="viewer_start_timeout")
                self.close()
            else:
                self._messages()
                self.last_frame = time.monotonic()
        except (OSError, RuntimeError) as exc:
            self.report.update(status="unavailable", reason=f"{type(exc).__name__}: {exc}")
            self.close()
        return self.report.copy()

    def _messages(self):
        while True:
            try:
                self.report.update(self.messages.get_nowait())
            except queue.Empty:
                break
        if self.process.exitcode is not None and self.report["status"] in ("starting", "running"):
            self.report.update(status="unavailable", reason=f"viewer_process_exit:{self.process.exitcode}")

    def update(self):
        if self.closed or self.process is None or self.process.pid is None:
            return
        self._messages()
        now = time.monotonic()
        if not self.process.is_alive() or self.report["status"] not in ("starting", "running"):
            return
        if now - self.last_frame < 1. / 30:
            return
        self.last_frame = now
        try:
            self.frames.put_nowait(self._state())
            self.sent_frames += 1
        except queue.Full:
            pass

    def close(self):
        if self.closed:
            return self.report.copy()
        if self.process is not None and self.process.pid is not None:
            self.stop.set()
            self.process.join(timeout=5.)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=1.)
                self.report.update(status="unavailable", reason="viewer_shutdown_timeout")
            self._messages()
        for name in ("frames", "messages"):
            channel = getattr(self, name, None)
            if channel is not None:
                channel.cancel_join_thread()
                channel.close()
        self.report["sent_frames"] = self.sent_frames
        self.closed = True
        return self.report.copy()
