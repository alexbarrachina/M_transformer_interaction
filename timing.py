"""Causal delta-time features shared by training and live performance (seconds)."""

import math
import random
from collections import deque
from statistics import median

import torch


TIME_UNIT = 0.032
MAX_INTERVAL = 127 * TIME_UNIT
HISTORY_SIZE = 16
PHRASE_GAP = 10.0
_LOG_RANGE = math.log(128)


class TimingHistory:
    """The previous 16 onsets, including zero and unknown intervals."""

    def __init__(self):
        self.intervals = deque(maxlen=HISTORY_SIZE)

    @property
    def scale(self):
        positive = [d for d in self.intervals if d is not None and d > 0]
        return min(MAX_INTERVAL, max(TIME_UNIT, median(positive) if positive else 0.25))

    def push(self, interval):
        """Encode before updating history. None/nonfinite/negative means unknown."""
        valid = interval is not None and math.isfinite(interval) and interval >= 0
        if not valid:
            self.intervals.append(None)
            return [0.0] * 4, False
        d = min(float(interval), MAX_INTERVAL)
        scale = self.scale
        features = [
            math.log1p(d / TIME_UNIT) / _LOG_RANGE,
            max(-3.0, min(3.0, math.log2(d / scale))) / 3.0 if d else 0.0,
            math.log1p(scale / TIME_UNIT) / _LOG_RANGE,
            float(d == 0),
        ]
        self.intervals.append(d)
        return features, True


def timing_features(intervals):
    """Encode a sequence with the same operation used for each live onset."""
    history = TimingHistory()
    rows = [history.push(d) for d in intervals]
    return (
        torch.tensor([row[0] for row in rows], dtype=torch.float32).reshape(-1, 4),
        torch.tensor([row[1] for row in rows], dtype=torch.bool),
    )


def augment_timing(intervals, target_start, rng=None):
    """Transform prefix and target together, keeping pitch order unchanged."""
    rng = rng or random
    result = list(intervals)
    mode = rng.random()
    if mode < 0.50:
        return result

    def log_uniform(low, high):
        return math.exp(rng.uniform(math.log(low), math.log(high)))

    if mode < 0.75:
        factor = log_uniform(0.6, 1.6)
        for i, d in enumerate(result):
            if d is not None:
                result[i] = d * factor * rng.uniform(0.85, 1.15) if d > 0 else (
                    rng.uniform(0, 0.064) if rng.random() < 0.2 else 0.0
                )
    elif mode < 0.85:
        history = TimingHistory()
        for d in result[:target_start]:
            history.push(d)
        grid = min(1.0, max(TIME_UNIT, history.scale / rng.choice((1, 2))))
        onset = previous = 0.0
        for i, d in enumerate(result):
            onset += d if d is not None else 0.0
            snapped = round(onset / grid) * grid
            if d is not None:
                result[i] = max(0.0, snapped - previous)
            previous = snapped
    elif mode < 0.90:
        interval = log_uniform(0.08, 0.8)
        result = [interval if d is not None else None for d in result]
    else:
        candidates = [i for i in range(target_start + 1, len(result))
                      if result[i] is not None]
        if candidates:
            i = rng.choice(candidates)
            result[i] = max(result[i], rng.uniform(1, 6))
    return result


class LiveTimingContext:
    """Bounded generation history; the application's recording remains separate."""

    def __init__(self, primer_pitch, primer_button, primer_dtime, context_length):
        self.primer = list(zip(primer_pitch, primer_button, primer_dtime))[:context_length]
        self.context_length = context_length
        self.enabled = True
        self.reset()

    def reset(self):
        self.history = TimingHistory()
        self.last_onset = None
        self.pitches = deque(maxlen=self.context_length + 1)
        self.buttons = deque(maxlen=self.context_length + 1)
        self.features = deque(maxlen=self.context_length + 1)
        self.valid = deque(maxlen=self.context_length + 1)
        for i, (pitch, button, count) in enumerate(self.primer):
            features, valid = self.history.push(count * TIME_UNIT if i else None)
            self.pitches.append(pitch)
            self.buttons.append(button)
            self.features.append(features)
            self.valid.append(valid)

    def begin_note(self, timestamp, button):
        interval = timestamp - self.last_onset if self.last_onset is not None else None
        restarted = interval is not None and interval >= PHRASE_GAP
        if restarted:
            self.reset()
            interval = None
        features, valid = self.history.push(interval)
        self.last_onset = timestamp
        self.pitches.append(0)  # target placeholder, excluded from decoder pitch inputs
        self.buttons.append(button)
        self.features.append(features)
        self.valid.append(valid)
        context = {
            'pitch': torch.tensor([list(self.pitches)], dtype=torch.long),
            'button': torch.tensor([list(self.buttons)], dtype=torch.long),
            'timing_features': torch.tensor([list(self.features)], dtype=torch.float32),
            'timing_mask': torch.tensor([list(self.valid)], dtype=torch.bool),
        }
        if not self.enabled:
            context['timing_mask'].zero_()
        return context, restarted

    def finish_note(self, pitch):
        self.pitches[-1] = pitch
