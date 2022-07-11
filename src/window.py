import mss
import ctypes
import numpy as np
import win32gui, win32ui, win32con

class Window:
    sct = None
    hwnd = None
    size = None
    position = None

    def __init__(self, window_name):
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        self.hwnd = win32gui.FindWindow(None, window_name)
        if not self.hwnd:
            raise Exception('window not found')
        self.update_size()
        self.sct = mss.mss()
        win32gui.SetWindowPos(
            self.hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)

    def __del__(self):
        win32gui.SetWindowPos(
            self.hwnd, win32con.HWND_NOTOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
            
    def update_size(self):
        window_rect = win32gui.GetClientRect(self.hwnd)
        self.size = (window_rect[2] - window_rect[0], window_rect[3] - window_rect[1])
        self.position = win32gui.ClientToScreen(self.hwnd, (0, 0))

    def capture(self):
        return np.array(
            self.sct.grab({
                "top": self.position[1], "left": self.position[0],
                "width": self.size[0], "height": self.size[1]
                }))