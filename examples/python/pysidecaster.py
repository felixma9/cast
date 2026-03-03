#!/usr/bin/env python

import ctypes
import os.path
import sys
from pathlib import Path
from typing import Final

import numpy as np

if sys.platform.startswith("linux"):
    libcast_handle = ctypes.CDLL("./libcast.so", ctypes.RTLD_GLOBAL)._handle  # load the libcast.so shared library
    pyclariuscast = ctypes.cdll.LoadLibrary("./pyclariuscast.so")  # load the pyclariuscast.so shared library

import pyclariuscast
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Slot

CMD_FREEZE: Final = 1
CMD_CAPTURE_IMAGE: Final = 2
CMD_CAPTURE_CINE: Final = 3
CMD_DEPTH_DEC: Final = 4
CMD_DEPTH_INC: Final = 5
CMD_GAIN_DEC: Final = 6
CMD_GAIN_INC: Final = 7
CMD_B_MODE: Final = 12
CMD_CFI_MODE: Final = 14


# custom event for handling change in freeze state
class FreezeEvent(QtCore.QEvent):
    def __init__(self, frozen):
        super().__init__(QtCore.QEvent.User)
        self.frozen = frozen


# custom event for handling button presses
class ButtonEvent(QtCore.QEvent):
    def __init__(self, btn, clicks):
        super().__init__(QtCore.QEvent.Type(QtCore.QEvent.User + 1))
        self.btn = btn
        self.clicks = clicks


# custom event for handling new images
class ImageEvent(QtCore.QEvent):
    def __init__(self):
        super().__init__(QtCore.QEvent.Type(QtCore.QEvent.User + 2))


# custom event for handling new RF heatmap images
class RawImageEvent(QtCore.QEvent):
    def __init__(self):
        super().__init__(QtCore.QEvent.Type(QtCore.QEvent.User + 3))


# manages custom events posted from callbacks, then relays as signals to the main widget
class Signaller(QtCore.QObject):
    freeze = QtCore.Signal(bool)
    button = QtCore.Signal(int, int)
    image = QtCore.Signal(QtGui.QImage)
    rf_image = QtCore.Signal(QtGui.QImage)

    def __init__(self):
        QtCore.QObject.__init__(self)
        self.usimage = QtGui.QImage()
        self.rf_qimage = QtGui.QImage()

    def event(self, evt):
        if evt.type() == QtCore.QEvent.User:
            self.freeze.emit(evt.frozen)
        elif evt.type() == QtCore.QEvent.Type(QtCore.QEvent.User + 1):
            self.button.emit(evt.btn, evt.clicks)
        elif evt.type() == QtCore.QEvent.Type(QtCore.QEvent.User + 2):
            self.image.emit(self.usimage)
        elif evt.type() == QtCore.QEvent.Type(QtCore.QEvent.User + 3):
            self.rf_image.emit(self.rf_qimage)
        return True


# global required for the cast api callbacks
signaller = Signaller()


# draws the ultrasound image
class ImageView(QtWidgets.QGraphicsView):
    def __init__(self, cast):
        QtWidgets.QGraphicsView.__init__(self)
        self.cast = cast
        self.setScene(QtWidgets.QGraphicsScene())

    # set the new image and redraw
    def updateImage(self, img):
        self.image = img
        self.scene().invalidate()

    # saves a local image
    def saveImage(self):
        self.image.save(str(Path.home() / "Pictures/clarius_image.png"))

    # resize the scan converter, image, and scene
    def resizeEvent(self, evt):
        w = evt.size().width()
        h = evt.size().height()
        self.cast.setOutputSize(w, h)
        self.image = QtGui.QImage(w, h, QtGui.QImage.Format_ARGB32)
        self.image.fill(QtCore.Qt.black)
        self.setSceneRect(0, 0, w, h)

    # black background
    def drawBackground(self, painter, rect):
        painter.fillRect(rect, QtCore.Qt.black)

    # draws the image
    def drawForeground(self, painter, rect):
        if not self.image.isNull():
            painter.drawImage(rect, self.image)


# draws the B/A proxy heatmap (no cast scan-converter dependency)
class HeatmapView(QtWidgets.QGraphicsView):
    def __init__(self):
        QtWidgets.QGraphicsView.__init__(self)
        self.setScene(QtWidgets.QGraphicsScene())
        self.image = QtGui.QImage()

    # set the new heatmap image and redraw
    def updateImage(self, img):
        self.image = img
        self.scene().invalidate()

    # resize the scene to match widget
    def resizeEvent(self, evt):
        w = evt.size().width()
        h = evt.size().height()
        self.setSceneRect(0, 0, w, h)

    # black background
    def drawBackground(self, painter, rect):
        painter.fillRect(rect, QtCore.Qt.black)

    # draws the heatmap scaled to fill the view
    def drawForeground(self, painter, rect):
        if not self.image.isNull():
            painter.drawImage(rect, self.image)


# main widget with controls and ui
class MainWidget(QtWidgets.QMainWindow):
    def __init__(self, cast, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)

        self.cast = cast
        self.setWindowTitle("Clarius Cast Demo")

        # create central widget within main window
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        ip = QtWidgets.QLineEdit("192.168.1.1")
        ip.setInputMask("000.000.000.000")
        port = QtWidgets.QLineEdit("5828")
        port.setInputMask("00000")

        conn = QtWidgets.QPushButton("Connect")
        self.run = QtWidgets.QPushButton("Run")
        quit = QtWidgets.QPushButton("Quit")
        depthUp = QtWidgets.QPushButton("< Depth")
        depthDown = QtWidgets.QPushButton("> Depth")
        gainInc = QtWidgets.QPushButton("> Gain")
        gainDec = QtWidgets.QPushButton("< Gain")
        captureImage = QtWidgets.QPushButton("Capture Image")
        captureCine = QtWidgets.QPushButton("Capture Movie")
        saveImage = QtWidgets.QPushButton("Save Local")
        bMode = QtWidgets.QPushButton("B Mode")
        cfiMode = QtWidgets.QPushButton("Color Mode")

        # try to connect/disconnect to/from the probe
        def tryConnect():
            if not cast.isConnected():
                if cast.connect(ip.text(), int(port.text()), "research"):
                    self.statusBar().showMessage("Connected")
                    conn.setText("Disconnect")
                else:
                    self.statusBar().showMessage(f"Failed to connect to {ip.text()}")
            elif cast.disconnect():
                self.statusBar().showMessage("Disconnected")
                conn.setText("Connect")
            else:
                self.statusBar().showMessage("Failed to disconnect")

        # try to freeze/unfreeze
        def tryFreeze():
            if cast.isConnected():
                cast.userFunction(CMD_FREEZE, 0)

        # try depth up
        def tryDepthUp():
            if cast.isConnected():
                cast.userFunction(CMD_DEPTH_DEC, 0)

        # try depth down
        def tryDepthDown():
            if cast.isConnected():
                cast.userFunction(CMD_DEPTH_INC, 0)

        # try gain down
        def tryGainDec():
            if cast.isConnected():
                cast.userFunction(CMD_GAIN_DEC, 0)

        # try gain up
        def tryGainInc():
            if cast.isConnected():
                cast.userFunction(CMD_GAIN_INC, 0)

        # try capture image
        def tryCaptureImage():
            if cast.isConnected():
                cast.userFunction(CMD_CAPTURE_IMAGE, 0)

        # try capture cine
        def tryCaptureCine():
            if cast.isConnected():
                cast.userFunction(CMD_CAPTURE_CINE, 0)

        # try to save a local image
        def trySaveImage():
            self.img.saveImage()

        # try b mode
        def tryBMode():
            if cast.isConnected():
                cast.userFunction(CMD_B_MODE, 0)

        # try cfi mode
        def tryCfiMode():
            if cast.isConnected():
                cast.userFunction(CMD_CFI_MODE, 0)

        conn.clicked.connect(tryConnect)
        self.run.clicked.connect(tryFreeze)
        quit.clicked.connect(self.shutdown)
        depthUp.clicked.connect(tryDepthUp)
        depthDown.clicked.connect(tryDepthDown)
        gainInc.clicked.connect(tryGainInc)
        gainDec.clicked.connect(tryGainDec)
        captureImage.clicked.connect(tryCaptureImage)
        captureCine.clicked.connect(tryCaptureCine)
        saveImage.clicked.connect(trySaveImage)
        bMode.clicked.connect(tryBMode)
        cfiMode.clicked.connect(tryCfiMode)

        # add widgets to layout
        self.img = ImageView(cast)

        # B/A Proxy right panel
        self.heatmap = HeatmapView()
        ba_label = QtWidgets.QLabel("B/A Proxy")
        ba_label.setAlignment(QtCore.Qt.AlignCenter)
        ba_panel = QtWidgets.QWidget()
        ba_vlayout = QtWidgets.QVBoxLayout(ba_panel)
        ba_vlayout.setContentsMargins(0, 0, 0, 0)
        ba_vlayout.addWidget(ba_label)
        ba_vlayout.addWidget(self.heatmap)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self.img)
        splitter.addWidget(ba_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([600, 600])

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(splitter)

        inplayout = QtWidgets.QHBoxLayout()
        layout.addLayout(inplayout)
        inplayout.addWidget(ip)
        inplayout.addWidget(port)

        connlayout = QtWidgets.QHBoxLayout()
        layout.addLayout(connlayout)
        connlayout.addWidget(conn)
        connlayout.addWidget(self.run)
        connlayout.addWidget(quit)
        central.setLayout(layout)

        prmlayout = QtWidgets.QHBoxLayout()
        layout.addLayout(prmlayout)
        prmlayout.addWidget(depthUp)
        prmlayout.addWidget(depthDown)
        prmlayout.addWidget(gainDec)
        prmlayout.addWidget(gainInc)

        caplayout = QtWidgets.QHBoxLayout()
        layout.addLayout(caplayout)
        caplayout.addWidget(captureImage)
        caplayout.addWidget(captureCine)
        caplayout.addWidget(saveImage)

        modelayout = QtWidgets.QHBoxLayout()
        layout.addLayout(modelayout)
        modelayout.addWidget(bMode)
        modelayout.addWidget(cfiMode)

        # connect signals
        signaller.freeze.connect(self.freeze)
        signaller.button.connect(self.button)
        signaller.image.connect(self.image)
        signaller.rf_image.connect(self.rfimage)

        # get home path
        path = os.path.expanduser("~/")
        if cast.init(path, 640, 480):
            self.statusBar().showMessage("Initialized")
        else:
            self.statusBar().showMessage("Failed to initialize")

    # handles freeze messages
    @Slot(bool)
    def freeze(self, frozen):
        if frozen:
            self.run.setText("Run")
            self.statusBar().showMessage("Image Stopped")
        else:
            self.run.setText("Freeze")
            self.statusBar().showMessage("Image Running (check firewall settings if no image seen)")

    # handles button messages
    @Slot(int, int)
    def button(self, btn, clicks):
        self.statusBar().showMessage(f"Button {btn} pressed w/ {clicks} clicks")

    # handles new images
    @Slot(QtGui.QImage)
    def image(self, img):
        self.img.updateImage(img)

    # handles new B/A proxy heatmap images
    @Slot(QtGui.QImage)
    def rfimage(self, img):
        self.heatmap.updateImage(img)

    # handles shutdown
    @Slot()
    def shutdown(self):
        if sys.platform.startswith("linux"):
            # unload the shared library before destroying the cast object
            ctypes.CDLL("libc.so.6").dlclose(libcast_handle)
        self.cast.destroy()
        QtWidgets.QApplication.quit()


## called when a new processed image is streamed
# @param image the scan-converted image data
# @param width width of the image in pixels
# @param height height of the image in pixels
# @param sz full size of image
# @param micronsPerPixel microns per pixel
# @param timestamp the image timestamp in nanoseconds
# @param angle acquisition angle for volumetric data
# @param imu inertial data tagged with the frame
def newProcessedImage(image, width, height, sz, micronsPerPixel, timestamp, angle, imu):
    bpp = sz / (width * height)
    if bpp == 4:
        img = QtGui.QImage(image, width, height, QtGui.QImage.Format_ARGB32)
    else:
        img = QtGui.QImage(image, width, height, QtGui.QImage.Format_Grayscale8)
    # a deep copy is important here, as the memory from 'image' won't be valid after the event posting
    signaller.usimage = img.copy()
    evt = ImageEvent()
    QtCore.QCoreApplication.postEvent(signaller, evt)


## called when a new raw image is streamed
# @param image the raw pre scan-converted image data, uncompressed 8-bit or jpeg compressed
# @param lines number of lines in the data
# @param samples number of samples in the data
# @param bps bits per sample
# @param axial microns per sample
# @param lateral microns per line
# @param timestamp the image timestamp in nanoseconds
# @param jpg jpeg compression size if the data is in jpeg format
# @param rf flag for if the image received is radiofrequency data
# @param angle acquisition angle for volumetric data
def newRawImage(image, lines, samples, bps, axial, lateral, timestamp, jpg, rf, angle):
    if rf != 1:
        return

    if lines <= 0 or samples <= 0:
        print(f"[BA Proxy] Warning: invalid dims lines={lines} samples={samples}")
        return

    data_len = len(image)
    expected_u16 = lines * samples * 2
    expected_u8 = lines * samples

    if data_len == expected_u16:
        arr = np.frombuffer(image, dtype='<u2').astype(np.float32)
    elif data_len == expected_u8:
        arr = np.frombuffer(image, dtype=np.uint8).astype(np.float32)
    else:
        print(f"[BA Proxy] Warning: unexpected byte length {data_len}, expected {expected_u8} (u8) or {expected_u16} (u16)")
        return

    try:
        arr = arr.reshape(lines, samples)
    except Exception as e:
        print(f"[BA Proxy] Warning: reshape failed: {e}")
        return

    if not np.isfinite(arr).any():
        print("[BA Proxy] Warning: no finite values in RF frame")
        return

    # B/A proxy parameters
    win_len = 64
    stride = 16
    eps = 1e-6

    # count only full-length windows; partial trailing windows have poor FFT resolution
    num_windows = max(1, (samples - win_len) // stride + 1)

    # --- Vectorized computation: no Python loops over lines or windows ---

    # Build start indices and extract all windows at once
    # window_indices shape: (num_windows, win_len)
    window_starts = np.arange(num_windows, dtype=np.int32) * stride
    window_indices = window_starts[:, None] + np.arange(win_len, dtype=np.int32)[None, :]

    # windows shape: (lines, num_windows, win_len)
    windows = arr[:, window_indices]

    # Remove DC per window
    windows = windows - windows.mean(axis=2, keepdims=True)

    # Power spectra for all windows at once: shape (lines, num_windows, nfft_bins)
    power = np.abs(np.fft.rfft(windows, axis=2)) ** 2
    nfft_bins = win_len // 2 + 1  # 33 for win_len=64
    last_bin = nfft_bins - 1      # 32

    # Find fundamental bin k1: strongest bin in [1 .. upper], shape (lines, num_windows)
    upper = min(max(2, win_len // 8), last_bin)  # 8 for win_len=64
    k1 = np.argmax(power[:, :, 1:upper + 1], axis=2) + 1

    # Second harmonic bin k2
    k2 = np.minimum(2 * k1, last_bin)

    # 3-bin neighbourhood energy; k1 in [1,8] and k2 in [2,16] for these params — no boundary issue
    L_idx = np.arange(lines)[:, None]        # (lines, 1)
    W_idx = np.arange(num_windows)[None, :]  # (1, num_windows)

    def gather_energy(kk):
        kk_lo = np.clip(kk - 1, 0, last_bin)
        kk_hi = np.clip(kk + 1, 0, last_bin)
        return power[L_idx, W_idx, kk_lo] + power[L_idx, W_idx, kk] + power[L_idx, W_idx, kk_hi]

    E1 = gather_energy(k1)
    E2 = gather_energy(k2)

    coarse_map = (E2 / (E1 + eps)).astype(np.float32)
    coarse_map = np.where(np.isfinite(coarse_map), coarse_map, 0.0)

    # Transpose so rows=depth (num_windows), cols=lateral (lines) — matches B-mode orientation
    coarse_map = coarse_map.T  # shape: (num_windows, lines)

    map_rows, map_cols = coarse_map.shape  # num_windows, lines

    # Upsample to display size via nearest-neighbor indexing
    disp_h, disp_w = 240, 320
    row_idx = np.clip((np.arange(disp_h) * map_rows / disp_h).astype(int), 0, map_rows - 1)
    col_idx = np.clip((np.arange(disp_w) * map_cols / disp_w).astype(int), 0, map_cols - 1)
    upsampled = coarse_map[np.ix_(row_idx, col_idx)]

    # Percentile scaling
    tiny_eps = 1e-9
    p5 = float(np.percentile(upsampled, 5))
    p95 = float(np.percentile(upsampled, 95))

    if p95 <= p5 + tiny_eps:
        print("[BA Proxy] Warning: collapsed percentile range, using zero fallback")
        normalized = np.zeros_like(upsampled)
    else:
        normalized = np.clip((upsampled - p5) / (p95 - p5), 0.0, 1.0)

    # Jet colormap via NumPy (no matplotlib)
    t = normalized.astype(np.float32)
    r = np.clip(1.5 - np.abs(4.0 * t - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * t - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * t - 1.0), 0.0, 1.0)

    rgb = np.stack([
        (r * 255).astype(np.uint8),
        (g * 255).astype(np.uint8),
        (b * 255).astype(np.uint8),
    ], axis=2)
    rgb = np.ascontiguousarray(rgb)

    try:
        qi = QtGui.QImage(rgb.data, disp_w, disp_h, disp_w * 3, QtGui.QImage.Format_RGB888)
        # deep copy: rgb buffer must not be referenced after this function returns
        signaller.rf_qimage = qi.copy()
        evt = RawImageEvent()
        QtCore.QCoreApplication.postEvent(signaller, evt)
    except Exception as e:
        print(f"[BA Proxy] Warning: image conversion/post failed: {e}")


## called when a new spectrum image is streamed
# @param image the spectral image
# @param lines number of lines in the spectrum
# @param samples number of samples per line
# @param bps bits per sample
# @param period line repetition period of spectrum
# @param micronsPerSample microns per sample for an m spectrum
# @param velocityPerSample velocity per sample for a pw spectrum
# @param pw flag that is true for a pw spectrum, false for an m spectrum
def newSpectrumImage(image, lines, samples, bps, period, micronsPerSample, velocityPerSample, pw):
    return


## called when a new imu data is streamed
# @param imu inertial data tagged with the frame
def newImuData(imu):
    return


## called when freeze state changes
# @param frozen the freeze state
def freezeFn(frozen):
    evt = FreezeEvent(frozen)
    QtCore.QCoreApplication.postEvent(signaller, evt)


## called when a button is pressed
# @param button the button that was pressed
# @param clicks number of clicks performed
def buttonsFn(button, clicks):
    evt = ButtonEvent(button, clicks)
    QtCore.QCoreApplication.postEvent(signaller, evt)


## main function
def main():
    cast = pyclariuscast.Caster(newProcessedImage, newRawImage, newSpectrumImage, newImuData, freezeFn, buttonsFn)
    app = QtWidgets.QApplication(sys.argv)
    widget = MainWidget(cast)
    widget.resize(1280, 600)
    widget.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
