"""Close the passive viewer before Python tears down GLFW."""

from contextlib import contextmanager
import threading


@contextmanager
def passive_viewer(model, data):
    import mujoco.viewer
    existing = set(threading.enumerate())
    handle = mujoco.viewer.launch_passive(model, data)
    # MuJoCo 3.14 starts a daemon UI thread; Handle.close only requests exit.
    viewer_threads = set(threading.enumerate()) - existing
    try:
        yield handle
    finally:
        handle.close()
        for thread in viewer_threads:
            thread.join(timeout=5.)
            if thread.is_alive():
                raise RuntimeError("Viewer did not finish shutting down")
