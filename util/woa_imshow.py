"""
Tkinter based fallback for cv2.imshow on headless OpenCV.

On Windows on ARM (WoA) only the headless build of OpenCV
(opencv-python-headless) is available, so cv2.imshow / cv2.waitKey and the
other highgui functions raise `cv2.error: The function is not implemented`.

This module provides a minimal drop-in replacement implemented with tkinter
(which ships with the standard CPython installer). Call `enable_if_needed()`
once at start-up (arg_utils does this automatically) and the affected cv2
highgui functions are transparently overwritten. Samples that use the common
`cv2.imshow` / `cv2.waitKey` / `cv2.getWindowProperty` / `cv2.destroyAllWindows`
pattern keep working without any change.
"""

import time
import base64

import numpy as np
import cv2

from logging import getLogger
logger = getLogger(__name__)


# =============================================================================
# headless detection
# =============================================================================
def is_headless():
    """Return True when the installed OpenCV has no GUI backend built in."""
    try:
        info = cv2.getBuildInformation()
    except Exception:
        return False

    for line in info.splitlines():
        stripped = line.strip()
        if stripped.startswith('GUI:'):
            # e.g. "GUI:                           NONE"
            value = stripped.split(':', 1)[1].strip().upper()
            return value in ('', 'NONE')

    # Older builds have no summary "GUI:" line; look at the backends instead.
    for backend in ('Win32 UI', 'GTK+', 'Cocoa', 'QT'):
        for line in info.splitlines():
            if line.strip().startswith(backend):
                if 'YES' in line.upper():
                    return False
    return True


# =============================================================================
# tkinter window manager
# =============================================================================
# keysym -> cv2 waitKey code for keys that have no printable char
_KEYSYM_MAP = {
    'Escape': 27,
    'Return': 13,
    'Tab': 9,
    'BackSpace': 8,
    'space': 32,
    'Delete': 127,
    'Left': 81,
    'Up': 82,
    'Right': 83,
    'Down': 84,
}


class _Window:
    def __init__(self, root, tk, name):
        self.name = name
        self.top = tk.Toplevel(root)
        self.top.title(name)
        self.label = tk.Label(self.top, borderwidth=0)
        self.label.pack()
        self.photo = None
        self.visible = True
        # Clicking the window's close button hides it (so getWindowProperty can
        # report it as no longer visible), matching how samples detect closing.
        self.top.protocol('WM_DELETE_WINDOW', self._on_close)

    def _on_close(self):
        self.visible = False
        try:
            self.top.withdraw()
        except Exception:
            pass


class _Manager:
    def __init__(self):
        self._tk = None
        self.root = None
        self.windows = {}
        self.last_key = -1

    def _ensure_root(self):
        if self.root is not None:
            return
        import tkinter as tk
        self._tk = tk
        self.root = tk.Tk()
        self.root.withdraw()  # we only use Toplevel windows
        self.root.bind_all('<Key>', self._on_key)

    def _on_key(self, event):
        ch = event.char
        if ch and len(ch) == 1 and 0 < ord(ch) < 256:
            self.last_key = ord(ch)
        elif event.keysym in _KEYSYM_MAP:
            self.last_key = _KEYSYM_MAP[event.keysym]

    def get_or_create(self, name):
        self._ensure_root()
        win = self.windows.get(name)
        if win is None or not win.top.winfo_exists():
            win = _Window(self.root, self._tk, name)
            self.windows[name] = win
        elif not win.visible:
            # re-show a window that had been closed by the user
            win.visible = True
            win.top.deiconify()
        return win

    def has_visible(self):
        return any(
            w.visible and w.top.winfo_exists() for w in self.windows.values()
        )

    def pump(self):
        if self.root is not None:
            try:
                self.root.update()
            except Exception:
                pass


_manager = _Manager()


# =============================================================================
# image conversion
# =============================================================================
def _to_photo(mat, tk):
    img = np.asarray(mat)

    # cv2.imshow scales floating point images (assumed range [0, 1]) by 255.
    if img.dtype in (np.float32, np.float64):
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    elif img.dtype == np.uint16:
        img = (img // 256).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    # cv2.imencode expects BGR and produces correct RGB in the PNG, so tkinter
    # decodes it with the right colors without a manual BGR->RGB conversion.
    ok, buf = cv2.imencode('.png', img)
    if not ok:
        raise RuntimeError('woa_imshow: failed to encode image')
    data = base64.b64encode(buf.tobytes())
    return tk.PhotoImage(data=data)


# =============================================================================
# cv2 highgui replacements
# =============================================================================
def imshow(winname, mat):
    win = _manager.get_or_create(winname)
    photo = _to_photo(mat, _manager._tk)
    win.photo = photo  # keep a reference so it is not garbage collected
    win.label.configure(image=photo)
    _manager.pump()


def waitKey(delay=0):
    if _manager.root is None:
        return -1

    _manager.last_key = -1
    if delay is None or delay <= 0:
        # block until a key is pressed or every window is closed
        while _manager.last_key == -1 and _manager.has_visible():
            _manager.pump()
            time.sleep(0.01)
    else:
        deadline = time.perf_counter() + delay / 1000.0
        while True:
            _manager.pump()
            if _manager.last_key != -1:
                break
            if time.perf_counter() >= deadline:
                break
            time.sleep(0.005)
    return _manager.last_key


# cv2.waitKeyEx behaves like waitKey for our purposes
def waitKeyEx(delay=0):
    return waitKey(delay)


def namedWindow(winname, flags=None):
    _manager.get_or_create(winname)


def destroyWindow(winname):
    win = _manager.windows.pop(winname, None)
    if win is not None:
        try:
            win.top.destroy()
        except Exception:
            pass
    _manager.pump()


def destroyAllWindows():
    for win in list(_manager.windows.values()):
        try:
            win.top.destroy()
        except Exception:
            pass
    _manager.windows.clear()
    _manager.pump()


def getWindowProperty(winname, prop_id):
    win = _manager.windows.get(winname)
    if win is None or not win.top.winfo_exists():
        return -1.0 if prop_id == cv2.WND_PROP_ASPECT_RATIO else 0.0
    if prop_id == cv2.WND_PROP_VISIBLE:
        return 1.0 if win.visible else 0.0
    if prop_id == cv2.WND_PROP_AUTOSIZE:
        return 1.0
    return 0.0


def setWindowProperty(winname, prop_id, prop_value):
    win = _manager.windows.get(winname)
    if win is None:
        return
    if prop_id == cv2.WND_PROP_FULLSCREEN:
        try:
            win.top.attributes('-fullscreen', prop_value == cv2.WINDOW_FULLSCREEN)
        except Exception:
            pass


def setWindowTitle(winname, title):
    win = _manager.windows.get(winname)
    if win is not None:
        try:
            win.top.title(title)
        except Exception:
            pass


def moveWindow(winname, x, y):
    win = _manager.get_or_create(winname)
    try:
        win.top.geometry('+%d+%d' % (int(x), int(y)))
    except Exception:
        pass


def resizeWindow(winname, width, height):
    # Windows auto-fit their image content, so this is intentionally a no-op.
    _manager.get_or_create(winname)


def startWindowThread():
    return 0


# =============================================================================
# activation
# =============================================================================
_PATCHED = False


def enable():
    """Overwrite cv2 highgui functions with the tkinter implementations."""
    global _PATCHED
    if _PATCHED:
        return
    try:
        import tkinter  # noqa: F401  (fail early if tkinter is unavailable)
    except Exception:
        logger.warning(
            'OpenCV is headless but tkinter is not available; '
            'cv2.imshow will not work.'
        )
        return

    for name in (
        'imshow', 'waitKey', 'waitKeyEx', 'namedWindow',
        'destroyWindow', 'destroyAllWindows',
        'getWindowProperty', 'setWindowProperty', 'setWindowTitle',
        'moveWindow', 'resizeWindow', 'startWindowThread',
    ):
        setattr(cv2, name, globals()[name])

    _PATCHED = True
    logger.info('headless OpenCV detected: cv2.imshow is now backed by tkinter.')


def enable_if_needed():
    """Enable the tkinter fallback only when OpenCV has no GUI backend."""
    if is_headless():
        enable()
