"""A minimal energy-threshold voice-activity detector for streaming PCM16 audio."""

from typing import Literal

import numpy as np


Transition = Literal["speech_started", "speech_stopped"]


class VoiceActivityDetector:
    """Detects speech start/stop boundaries in a stream of 16-bit PCM chunks.

    A simple energy-threshold detector, good enough to bring up the realtime
    protocol end to end; replace with an OpenVINO-exported Silero VAD once the
    rest of the pipeline works (see the polish phase of the local-OpenVINO plan).
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        threshold: float = 0.02,
        silence_hold_s: float = 0.6,
    ) -> None:
        """Initialize the detector for a given sample rate and sensitivity."""
        self._silence_hold_samples = int(silence_hold_s * sample_rate)
        self._threshold = threshold
        self._speaking = False
        self._silence_run_samples = 0

    def push(self, pcm16_chunk: bytes) -> Transition | None:
        """Feed one chunk of 16-bit PCM audio; return a transition, if any."""
        samples = np.frombuffer(pcm16_chunk, dtype=np.int16)
        if samples.size == 0:
            return None

        rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float32) / 32768.0))))

        if rms >= self._threshold:
            self._silence_run_samples = 0
            if not self._speaking:
                self._speaking = True
                return "speech_started"
            return None

        if self._speaking:
            self._silence_run_samples += samples.size
            if self._silence_run_samples >= self._silence_hold_samples:
                self._speaking = False
                self._silence_run_samples = 0
                return "speech_stopped"
        return None
