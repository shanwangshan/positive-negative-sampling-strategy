import av
import numpy as np
from fractions import Fraction

def av_video_loader(path=None, container=None, rate=None, start_time=None, duration=None):
    # Extract video frames
    if container is None:
        container = av.open(path)
    video_stream = container.streams.video[0]

    # Parse metadata
    _rate = video_stream.average_rate
    _ss = video_stream.start_time * video_stream.time_base
    _dur = video_stream.duration * video_stream.time_base
    _ff = _ss + _dur

    if rate is None:
        rate = _rate
    if start_time is None:
        start_time = _ss
    if duration is None:
        duration = _ff - start_time
    duration = min(duration, _ff - start_time)
    end_time = start_time + duration

    # Figure out which frames to read in advance
    out_times = [t for t in np.arange(start_time, min(end_time, _ff) - 0.5 / _rate, 1. / rate)]
    out_frame_no = [int((t - _ss) * _rate) for t in out_times][:int(duration * rate)]
    start_time = out_frame_no[0] / float(_rate)

    import time

    # Read data
    n_read = 0
    video = [None] * len(out_frame_no)
    container.seek(int(start_time * av.time_base))

    ts = time.time()
    decode_time, cpu_time = 0., 0.
    for frame in container.decode(video=0):
        decode_time += time.time() - ts
        ts = time.time()
        if n_read == len(out_frame_no):
            break
        frame_no = frame.pts * frame.time_base * _rate
        if frame_no < out_frame_no[n_read]:
            continue
        pil_img = frame.to_image()
        while frame_no >= out_frame_no[n_read]:    # This 'while' takes care of the case where _rate < rate
            video[n_read] = pil_img
            n_read += 1
            if n_read == len(out_frame_no):
                break
        cpu_time += time.time() - ts
        ts = time.time()
    video = [v for v in video if v is not None]

    return (video, rate), (decode_time, cpu_time)

def av_audio_loader(path=None, container=None, rate=None, start_time=None, duration=None, layout="4.0"):
    if container is None:
        container = av.open(path)
    audio_stream = container.streams.audio[0]

    # Parse metadata
    _ss = audio_stream.start_time * audio_stream.time_base if audio_stream.start_time is not None else 0.
    _dur = audio_stream.duration * audio_stream.time_base
    _ff = _ss + _dur
    _rate = audio_stream.rate

    if rate is None:
        rate = _rate
    if start_time is None:
        start_time = _ss
    if duration is None:
        duration = _ff - start_time
    duration = min(duration, _ff - start_time)
    end_time = start_time + duration

    #resampler = av.audio.resampler.AudioResampler(format="s16p", layout=layout, rate=rate)
    resampler = None

    # Read data
    chunks = []
    container.seek(int(start_time * av.time_base))
    for frame in container.decode(audio=0):
        chunk_start_time = frame.pts * frame.time_base
        chunk_end_time = chunk_start_time + Fraction(frame.samples, frame.rate)
        if chunk_end_time < start_time:   # Skip until start time
            continue
        if chunk_start_time > end_time:       # Exit if clip has been extracted
            break

        try:
            frame.pts = None
            if resampler is not None:
                chunks.append((chunk_start_time, resampler.resample(frame).to_ndarray()))
            else:
                chunks.append((chunk_start_time, frame.to_ndarray()))
        except AttributeError:
            break

    # Trim for frame accuracy
    audio = np.concatenate([af[1] for af in chunks], 1)
    ss = int((start_time - chunks[0][0]) * rate)
    t = int(duration * rate)
    if ss < 0:
        audio = np.pad(audio, ((0, 0), (-ss, 0)), 'constant', constant_values=0)
        ss = 0
    audio = audio[:, ss: ss+t]

    # Normalize to [-1, 1]
    #audio = audio / np.max(audio)

    return audio, rate
