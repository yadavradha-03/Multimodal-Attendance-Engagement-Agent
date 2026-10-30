"""
Synthetic Classroom Multimodal Data Generator.
Creates a realistic test video (.mp4) and synchronized audio track (.wav)
simulating distinct classroom behaviors (attentive, distracted, drowsy, late arrival, passerby).
Enables instant 100% offline testing and demonstration.
"""
import math
from pathlib import Path
import cv2
import numpy as np
from scipy.io import wavfile


def create_avatar_face(
    size: int = 140,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    eye_open: bool = True,
    skin_tone: tuple = (185, 205, 235), # BGR
    expression: str = "neutral"
) -> np.ndarray:
    """Generate a realistic synthetic human-like avatar face image with controllable pose."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cx, cy = size // 2, size // 2
    
    # Face oval
    face_w = int(size * 0.38)
    face_h = int(size * 0.48)
    cv2.ellipse(img, (cx, cy), (face_w, face_h), 0, 0, 360, skin_tone, -1)
    cv2.ellipse(img, (cx, cy), (face_w, face_h), 0, 0, 360, (skin_tone[0]-30, skin_tone[1]-30, skin_tone[2]-30), 2)

    # Hair
    hair_color = (35, 45, 60)
    cv2.ellipse(img, (cx, cy - int(face_h * 0.45)), (face_w + 4, int(face_h * 0.4)), 0, 180, 360, hair_color, -1)

    # Pose offsets
    yaw_shift = int(np.clip(yaw_deg * 0.6, -18, 18))
    pitch_shift = int(np.clip(pitch_deg * 0.6, -18, 18))

    # Eyes
    eye_y = cy - int(face_h * 0.12) + pitch_shift
    eye_spacing = int(face_w * 0.48)
    left_x = cx - eye_spacing + yaw_shift
    right_x = cx + eye_spacing + yaw_shift

    eye_radius = int(face_w * 0.14)
    if eye_open:
        # Sclera
        cv2.circle(img, (left_x, eye_y), eye_radius, (245, 245, 245), -1)
        cv2.circle(img, (right_x, eye_y), eye_radius, (245, 245, 245), -1)
        # Iris
        iris_color = (60, 45, 30)
        iris_offset = int(yaw_deg * 0.15)
        cv2.circle(img, (left_x + iris_offset, eye_y), int(eye_radius * 0.55), iris_color, -1)
        cv2.circle(img, (right_x + iris_offset, eye_y), int(eye_radius * 0.55), iris_color, -1)
        # Pupils
        cv2.circle(img, (left_x + iris_offset, eye_y), int(eye_radius * 0.25), (10, 10, 10), -1)
        cv2.circle(img, (right_x + iris_offset, eye_y), int(eye_radius * 0.25), (10, 10, 10), -1)
    else:
        # Closed eyes (drowsy / sleeping)
        cv2.line(img, (left_x - eye_radius, eye_y), (left_x + eye_radius, eye_y), (50, 40, 30), 2)
        cv2.line(img, (right_x - eye_radius, eye_y), (right_x + eye_radius, eye_y), (50, 40, 30), 2)

    # Eyebrows
    brow_y = eye_y - eye_radius - 4
    cv2.line(img, (left_x - eye_radius, brow_y), (left_x + eye_radius, brow_y), hair_color, 2)
    cv2.line(img, (right_x - eye_radius, brow_y), (right_x + eye_radius, brow_y), hair_color, 2)

    # Nose
    nose_x = cx + yaw_shift
    nose_y = cy + int(face_h * 0.12) + pitch_shift
    cv2.line(img, (nose_x, nose_y - 8), (nose_x, nose_y + 4), (skin_tone[0]-40, skin_tone[1]-40, skin_tone[2]-40), 2)
    cv2.circle(img, (nose_x, nose_y + 4), 3, (skin_tone[0]-45, skin_tone[1]-45, skin_tone[2]-45), -1)

    # Mouth
    mouth_y = cy + int(face_h * 0.40) + pitch_shift
    mouth_w = int(face_w * 0.45)
    mouth_x1 = cx - mouth_w + yaw_shift
    mouth_x2 = cx + mouth_w + yaw_shift
    cv2.line(img, (mouth_x1, mouth_y), (mouth_x2, mouth_y), (70, 70, 170), 3)

    return img


def generate_classroom_demo(
    output_video_path: Path,
    output_audio_path: Path,
    duration_sec: int = 30,
    fps: int = 15,
    width: int = 800,
    height: int = 500
):
    """
    Generates a synchronized multimodal test classroom video and audio dataset.
    
    Student 1 (Front-Left, 'Alex'): Attentive, focused forward, continuous presence.
    Student 2 (Front-Right, 'Sam'): Starts attentive, looks sideways & distracted at t=10s-20s.
    Student 3 (Back-Left, 'Jordan'): Slumped & drowsy, stationary posture in the back.
    Student 4 (Back-Right, 'Taylor'): Late arrival (arrives at t=10s, attends till end).
    Transient Passerby: Walks past camera at t=4s to t=6s (should NOT be marked present).
    """
    output_video_path = Path(output_video_path)
    output_audio_path = Path(output_audio_path)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    total_frames = int(duration_sec * fps)

    # Desks and classroom layout
    # Alex: (140, 260) - Front-Left
    # Sam:  (540, 260) - Front-Right
    # Jordan: (200, 100) - Back-Left
    # Taylor: (500, 100) - Back-Right

    for frame_idx in range(total_frames):
        t = frame_idx / fps

        # Background: Modern classroom with whiteboard
        frame = np.full((height, width, 3), (240, 242, 245), dtype=np.uint8)

        # Draw classroom chalkboard / whiteboard
        cv2.rectangle(frame, (100, 15), (700, 75), (220, 225, 230), -1)
        cv2.rectangle(frame, (100, 15), (700, 75), (160, 170, 180), 2)
        cv2.putText(frame, "CLASSROOM INTELLIGENCE DEMO - SESSION IN PROGRESS", (130, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 70, 90), 2)

        # Draw desks
        # Back row desks
        cv2.rectangle(frame, (120, 180), (360, 205), (180, 160, 140), -1)
        cv2.rectangle(frame, (440, 180), (680, 205), (180, 160, 140), -1)
        # Front row desks
        cv2.rectangle(frame, (60, 360), (380, 390), (170, 150, 130), -1)
        cv2.rectangle(frame, (420, 360), (740, 390), (170, 150, 130), -1)

        # 1. Student 1 ('Alex') - Front-Left: Attentive, slight natural nodding
        alex_yaw = math.sin(t * 0.8) * 4.0
        alex_pitch = math.cos(t * 1.5) * 5.0
        alex_face = create_avatar_face(110, yaw_deg=alex_yaw, pitch_deg=alex_pitch, eye_open=True, skin_tone=(180, 205, 240))
        # Overlay
        ax, ay = 170, 250
        frame[ay:ay+110, ax:ax+110] = alex_face
        # Torso
        cv2.ellipse(frame, (ax + 55, ay + 140), (55, 35), 0, 0, 360, (80, 120, 180), -1)
        cv2.putText(frame, "Alex", (ax + 35, ay - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)

        # 2. Student 2 ('Sam') - Front-Right: Attentive early, then distracted (t=9s to 21s)
        if 9.0 <= t <= 21.0:
            sam_yaw = 32.0  # looking right / distracted
            sam_pitch = -24.0 # looking down at phone
        else:
            sam_yaw = math.sin(t * 0.5) * 6.0
            sam_pitch = 0.0
        sam_face = create_avatar_face(110, yaw_deg=sam_yaw, pitch_deg=sam_pitch, eye_open=True, skin_tone=(165, 190, 230))
        sx, sy = 530, 250
        frame[sy:sy+110, sx:sx+110] = sam_face
        cv2.ellipse(frame, (sx + 55, sy + 140), (55, 35), 0, 0, 360, (70, 160, 120), -1)
        cv2.putText(frame, "Sam", (sx + 35, sy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)

        # 3. Student 3 ('Jordan') - Back-Left: Drowsy / slumped posture
        jordan_eye = False if t > 5.0 else True
        jordan_pitch = -35.0  # Head slumped down
        jordan_face = create_avatar_face(90, yaw_deg=0.0, pitch_deg=jordan_pitch, eye_open=jordan_eye, skin_tone=(175, 195, 225))
        jx, jy = 200, 100
        frame[jy:jy+90, jx:jx+90] = jordan_face
        cv2.ellipse(frame, (jx + 45, jy + 110), (45, 25), 0, 0, 360, (140, 90, 150), -1)
        cv2.putText(frame, "Jordan", (jx + 25, jy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1)

        # 4. Student 4 ('Taylor') - Back-Right: Late arrival at t >= 10.0s
        if t >= 10.0:
            taylor_yaw = math.sin(t * 1.2) * 8.0
            taylor_face = create_avatar_face(90, yaw_deg=taylor_yaw, pitch_deg=-5.0, eye_open=True, skin_tone=(190, 215, 245))
            tx, ty = 500, 100
            frame[ty:ty+90, tx:tx+90] = taylor_face
            cv2.ellipse(frame, (tx + 45, ty + 110), (45, 25), 0, 0, 360, (200, 130, 60), -1)
            cv2.putText(frame, "Taylor (Late)", (tx + 5, ty - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1)

        # 5. Transient Passerby: walks across camera between t=3.5s and 5.5s
        if 3.5 <= t <= 5.5:
            walk_progress = (t - 3.5) / 2.0
            px = int(walk_progress * (width + 100) - 50)
            py = 340
            if 0 <= px <= width - 90:
                passer_face = create_avatar_face(85, yaw_deg=25.0, pitch_deg=0.0, eye_open=True, skin_tone=(170, 180, 210))
                frame[py:py+85, px:px+85] = passer_face
                cv2.ellipse(frame, (px + 42, py + 100), (42, 25), 0, 0, 360, (90, 90, 90), -1)
                cv2.putText(frame, "Passerby", (px + 10, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 180), 1)

        # Timestamp watermark
        cv2.putText(frame, f"Time: {t:05.1f}s | Frame {frame_idx}/{total_frames}", (20, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        writer.write(frame)

    writer.release()

    # Generate synchronized audio file (.wav)
    # Audio structure:
    # 0s - 10s: Teacher lecture monologue (harmonic tone with speech-like modulation)
    # 10s - 16s: Pause / silence (low ambient noise)
    # 16s - 24s: Student discussion / Q&A (bursty energy, pitch transitions)
    # 24s - 30s: Teacher summary + ambient room chatter
    sample_rate = 16000
    total_samples = int(duration_sec * sample_rate)
    audio = np.zeros(total_samples, dtype=np.float32)

    # Ambient room noise floor
    np.random.seed(42)
    noise = np.random.normal(0, 0.003, total_samples).astype(np.float32)
    audio += noise

    # Timeline modulation
    for i in range(total_samples):
        t_sec = i / sample_rate
        
        if 0.5 <= t_sec <= 9.5:
            # Teacher lecture: 160 Hz fundamental with harmonics and syllabic envelope
            envelope = 0.5 * (1 + math.sin(2 * math.pi * 3.5 * t_sec))  # speech rhythm ~3.5 Hz
            f0 = 160.0 + 15.0 * math.sin(2 * math.pi * 0.5 * t_sec)
            speech_wave = (
                0.25 * math.sin(2 * math.pi * f0 * t_sec) +
                0.15 * math.sin(2 * math.pi * 2 * f0 * t_sec) +
                0.08 * math.sin(2 * math.pi * 3 * f0 * t_sec)
            )
            audio[i] += speech_wave * envelope

        elif 10.0 <= t_sec <= 15.0:
            # Silence / study thinking pause
            pass

        elif 16.0 <= t_sec <= 23.5:
            # Student Discussion: dynamic pitches (220 Hz and 130 Hz turn-taking)
            speaker_f0 = 220.0 if math.sin(2 * math.pi * 0.4 * t_sec) > 0 else 135.0
            envelope = 0.6 * (1 + math.sin(2 * math.pi * 5.0 * t_sec))
            disc_wave = (
                0.30 * math.sin(2 * math.pi * speaker_f0 * t_sec) +
                0.18 * math.sin(2 * math.pi * 2 * speaker_f0 * t_sec)
            )
            audio[i] += disc_wave * envelope

        elif 24.0 <= t_sec <= 29.5:
            # Chatter / wrap up
            chatter = np.random.normal(0, 0.02)
            f0 = 175.0
            audio[i] += 0.15 * math.sin(2 * math.pi * f0 * t_sec) + chatter

    # Normalize to int16
    audio_norm = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio_norm * 32767).astype(np.int16)
    wavfile.write(str(output_audio_path), sample_rate, audio_int16)

    return str(output_video_path), str(output_audio_path)


if __name__ == "__main__":
    v_path = Path("data/sample_classroom.mp4")
    a_path = Path("data/sample_classroom.wav")
    print(f"Generating sample classroom dataset ({v_path}, {a_path})...")
    generate_classroom_demo(v_path, a_path, duration_sec=25)
    print("Classroom sample dataset generated successfully!")
