#!/usr/bin/env python

import ctypes
import logging
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


class ProxyImageEvent(QtCore.QEvent):
    def __init__(self):
        super().__init__(QtCore.QEvent.Type(QtCore.QEvent.User + 3))


# manages custom events posted from callbacks, then relays as signals to the main widget
class Signaller(QtCore.QObject):
    freeze = QtCore.Signal(bool)
    button = QtCore.Signal(int, int)
    image = QtCore.Signal(QtGui.QImage)
    proxyimage = QtCore.Signal(QtGui.QImage)

    def __init__(self):
        QtCore.QObject.__init__(self)
        self.usimage = QtGui.QImage()
        self.proxy = QtGui.QImage()

    def event(self, evt):
        if evt.type() == QtCore.QEvent.User:
            self.freeze.emit(evt.frozen)
        elif evt.type() == QtCore.QEvent.Type(QtCore.QEvent.User + 1):
            self.button.emit(evt.btn, evt.clicks)
        elif evt.type() == QtCore.QEvent.Type(QtCore.QEvent.User + 2):
            self.image.emit(self.usimage)
        elif evt.type() == QtCore.QEvent.Type(QtCore.QEvent.User + 3):
            self.proxyimage.emit(self.proxy)
        return True


# global required for the cast api callbacks
signaller = Signaller()
logger = logging.getLogger(__name__)


# draws the ultrasound image
class ImageView(QtWidgets.QGraphicsView):
    def __init__(self, cast=None):
        QtWidgets.QGraphicsView.__init__(self)
        self.cast = cast
        self.setScene(QtWidgets.QGraphicsScene())
        self.image = QtGui.QImage()

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
        if self.cast is not None:
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
        self.proxy_img = ImageView()
        self.proxy_title = QtWidgets.QLabel("B/A Proxy")
        self.proxy_title.setAlignment(QtCore.Qt.AlignCenter)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.addWidget(self.img)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.proxy_title)
        right_layout.addWidget(self.proxy_img)
        right.setLayout(right_layout)

        split.addWidget(right)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        split.setSizes([1, 1])

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(split)

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
        signaller.proxyimage.connect(self.proxyImage)

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

    @Slot(QtGui.QImage)
    def proxyImage(self, img):
        self.proxy_img.updateImage(img)

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
        logger.warning("RF frame ignored: invalid dimensions lines=%s samples=%s", lines, samples)
        return

    expected_u16 = lines * samples * 2
    expected_u8 = lines * samples

    try:
        payload_len = len(image)
    except Exception:
        logger.warning("RF frame ignored: cannot determine payload length")
        return

    try:
        if payload_len == expected_u16:
            raw = np.frombuffer(image, dtype="<u2")
        elif payload_len == expected_u8:
            raw = np.frombuffer(image, dtype=np.uint8)
        else:
            logger.warning(
                "RF frame ignored: payload size mismatch actual=%s expected_u16=%s expected_u8=%s",
                payload_len,
                expected_u16,
                expected_u8,
            )
            return
    except Exception as exc:
        logger.warning("RF frame ignored: payload parsing failed: %s", exc)
        return

    try:
        rf_data = raw.reshape((lines, samples)).astype(np.float32, copy=False)
    except Exception as exc:
        logger.warning("RF frame ignored: reshape failed for (%s, %s): %s", lines, samples, exc)
        return

    if not np.isfinite(rf_data).all():
        logger.warning("RF frame contains non-finite values; replacing with finite defaults")
        rf_data = np.nan_to_num(rf_data, nan=0.0, posinf=0.0, neginf=0.0)

    window = 64
    stride = 64
    eps = 1e-6
    tiny_eps = 1e-12

    starts = list(range(0, samples, stride))
    if not starts:
        logger.warning("RF frame ignored: no axial windows available")
        return

    coarse = np.zeros((lines, len(starts)), dtype=np.float32)

    for line_idx in range(lines):
        line = rf_data[line_idx]
        for col_idx, start in enumerate(starts):
            seg = line[start : start + window]
            if seg.size < 2:
                coarse[line_idx, col_idx] = 0.0
                continue

            seg = seg.astype(np.float32, copy=False)
            seg = seg - np.mean(seg, dtype=np.float64)

            power = np.abs(np.fft.rfft(seg)) ** 2
            if power.size <= 1:
                coarse[line_idx, col_idx] = 0.0
                continue

            power = power.astype(np.float32, copy=False)
            last_bin = power.size - 1
            search_end = min(max(2, seg.size // 8), last_bin)
            if search_end < 1:
                coarse[line_idx, col_idx] = 0.0
                continue

            search_band = power[1 : search_end + 1]
            if search_band.size == 0 or not np.isfinite(search_band).any():
                coarse[line_idx, col_idx] = 0.0
                continue

            k1 = int(np.argmax(search_band)) + 1
            k2 = min(2 * k1, last_bin)

            e1_lo = max(1, k1 - 1)
            e1_hi = min(last_bin, k1 + 1)
            e2_lo = max(1, k2 - 1)
            e2_hi = min(last_bin, k2 + 1)

            e1 = float(np.sum(power[e1_lo : e1_hi + 1], dtype=np.float64))
            e2 = float(np.sum(power[e2_lo : e2_hi + 1], dtype=np.float64))
            coarse[line_idx, col_idx] = e2 / (e1 + eps)

    if not np.isfinite(coarse).all():
        logger.warning("Proxy map contains non-finite values; replacing with finite defaults")
        coarse = np.nan_to_num(coarse, nan=0.0, posinf=0.0, neginf=0.0)

    expanded = np.repeat(coarse, stride, axis=1)
    if expanded.shape[1] < samples:
        pad = samples - expanded.shape[1]
        expanded = np.pad(expanded, ((0, 0), (0, pad)), mode="edge")
    elif expanded.shape[1] > samples:
        expanded = expanded[:, :samples]

    p5, p95 = np.percentile(expanded, [5, 95])
    if not np.isfinite(p5) or not np.isfinite(p95):
        logger.warning("Proxy scaling percentiles are non-finite; applying safe fallback")
        p5, p95 = 0.0, 1.0

    if p95 <= p5 + tiny_eps:
        logger.warning("Proxy scaling collapsed (p5=%s p95=%s); applying safe fallback", p5, p95)
        normalized = np.zeros_like(expanded, dtype=np.float32)
    else:
        normalized = (expanded - p5) / (p95 - p5)
        normalized = np.clip(normalized, 0.0, 1.0)

    if not np.isfinite(normalized).all():
        logger.warning("Proxy normalized map contains non-finite values; replacing with finite defaults")
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)

    n = normalized
    r = np.clip(1.5 - np.abs(4.0 * n - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * n - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * n - 1.0), 0.0, 1.0)
    rgb = np.stack((r, g, b), axis=-1)
    rgb8 = (rgb * 255.0).astype(np.uint8)

    try:
        h, w = rgb8.shape[0], rgb8.shape[1]
        qimg = QtGui.QImage(rgb8.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        signaller.proxy = qimg.copy()
        evt = ProxyImageEvent()
        QtCore.QCoreApplication.postEvent(signaller, evt)
    except Exception as exc:
        logger.warning("Proxy image update failed: %s", exc)
        return


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
    widget.resize(640, 480)
    widget.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
