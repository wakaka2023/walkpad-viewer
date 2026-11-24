"""Positive evidence that a boxed figure lies behind a structured surface.

Slats outside a figure are not sufficient: their periodic image signal must
continue over two separated parts of the figure, with the same spatial phase.
This is an activity-area exclusion, not proof of the optical reflection type.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates


def _band(img, box, vertical_range):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    strips = ((-.39, -.13), (.24, .38), (.43, .57), (.62, .76), (1.12, 1.40))
    x = np.concatenate([np.linspace(x1 + a*w, x1 + b*w, 12) for a, b in strips])
    y = np.arange(np.ceil(y1 + vertical_range[0]*h), np.floor(y1 + vertical_range[1]*h))
    if len(y) < 22 or x.min() < 2 or x.max() >= img.shape[1] - 2:
        return None
    # Search projected slat pitch and tilt, not any particular pixel location.
    hz = np.fft.rfftfreq(len(y))
    band = (hz >= 4.5/len(y)) & (hz <= 1/3.4)
    if not band.any():
        return None
    center = (x1+x2)/2
    step = min(.012, 1 / max(np.ptp(x), 1.))
    tilts = np.arange(-.12, .12 + step, step)
    window = np.hanning(len(y))
    best = None
    for tilt in tilts:
        yy = y[:, None] + tilt*(x[None, :] - center)
        if yy.min() < 1 or yy.max() >= img.shape[0] - 1:
            continue
        pixels = map_coordinates(img, [yy, np.broadcast_to(x, yy.shape)], order=1, prefilter=False)
        detail = pixels - gaussian_filter1d(pixels, max(2., len(y)/21), axis=0)
        profile = np.median(detail.reshape(len(y), 5, 12), axis=2).T
        fft = np.fft.rfft(profile * window)[:, band]
        power = abs(fft) ** 2
        energy = np.sqrt(power) * (2 / window.sum())
        if not np.isfinite(energy).all():
            continue
        share = power / (power.sum(axis=1, keepdims=True) + 1e-9)
        unit = fft / np.maximum(abs(fft), 1e-9)
        alignment = np.real(unit * np.conj(unit.mean(axis=0)))
        # Broadband texture, phase jumps at the person, and alias-only patterns
        # cannot count as a signal continuing over the person.
        good = ((energy.min(axis=0) >= 1.6) & (share.min(axis=0) >= .28)
                & (alignment.min(axis=0) >= .76))
        if not good.any():
            continue
        score = np.where(good, share.min(axis=0) * alignment.min(axis=0), -1)
        j = int(np.argmax(score))
        if best is None or score[j] > best[0]:
            best = (float(score[j]), float(hz[band][j]), float(tilt), float(1/len(y)))
    return best


def foreground_slats(image, box):
    """Return minimum spectral evidence of a common foreground lattice, or None."""
    if not np.isfinite(box).all():
        return None
    for p, b in ((image, np.asarray(box)), (image.T, np.asarray(box)[[1,0,3,2]])):
        p = np.asarray(p, np.float32)
        first = _band(p, b, (.13, .43))
        if first is None:
            continue
        second = _band(p, b, (.44, .75))
        if second is None:
            continue
        u, f0, angle0, r0 = first
        v, f1, angle1, r1 = second
        if abs(f0-f1) <= max(r0, r1) and abs(angle0-angle1)*np.ptp(b[[0,2]]) <= .5 / max(f0,f1):
            return min(u,v)
    return None


def persistent_foreground_slats(images, box):
    if len(images) < 3:
        return None
    observations = [images[i] for i in np.linspace(0,len(images)-1,3).round().astype(int)]
    result = []
    for image in observations:
        score = foreground_slats(image, box)
        if score is None:
            return None
        result.append(score)
    return min(result)
