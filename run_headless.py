"""
Headless CLI Pipeline Runner for Multimodal Attendance & Engagement Agent.
Processes video and audio files from command line and generates attendance CSVs,
annotated output video, and executive intelligence markdown reports.
"""
import argparse
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np

from config import (
    YUNET_MODEL_PATH,
    DATA_DIR,
    EXPORTS_DIR
)
from vision.face_detector import FaceDetector
from vision.pose_estimator import HeadPoseEstimator
from vision.motion_analyzer import MotionAnalyzer
from vision.drowsiness_detector import DrowsinessDetector
from core.tracker import ParticipantTracker
from core.presence import PresenceManager
from audio.signal_processor import AudioSignalProcessor, AudioFeatures
from audio.vad import VoiceActivityDetector
from fusion.engagement_engine import MultimodalEngagementEngine
from agent.teaching_advisor import TeachingAdvisorAgent
from agent.session_analyzer import SessionAnalyticsReport
from demo_generator import generate_classroom_demo


def run_pipeline(
    video_path: Path,
    audio_path: Path,
    output_video_path: Path,
    output_csv_path: Path,
    output_report_path: Path,
    min_sustained_sec: float = 6.0,
    tardy_sec: float = 12.0,
    privacy_blur: bool = False,
    max_seconds: float = 60.0
):
    print(f"🎬 Initializing Multimodal Classroom Analysis:")
    print(f"   Video: {video_path}")
    print(f"   Audio: {audio_path}")
    print(f"   Min Sustained Presence: {min_sustained_sec}s | Tardy Cutoff: {tardy_sec}s")
    print(f"   Privacy Face Blur: {'Enabled' if privacy_blur else 'Disabled'}")

    detector = FaceDetector(model_path=YUNET_MODEL_PATH)
    pose_estimator = HeadPoseEstimator()
    motion_analyzer = MotionAnalyzer()
    drowsiness_detector = DrowsinessDetector(inactivity_threshold_sec=8.0)
    tracker = ParticipantTracker(iou_threshold=0.25)
    presence_manager = PresenceManager(min_sustained_seconds=min_sustained_sec, tardy_threshold_seconds=tardy_sec)
    audio_processor = AudioSignalProcessor(sample_rate=16000)
    vad = VoiceActivityDetector()
    fusion_engine = MultimodalEngagementEngine()
    advisor = TeachingAdvisorAgent(cooldown_sec=5.0)

    # Audio track loading
    audio_data = None
    audio_sr = 16000
    if audio_path and Path(audio_path).exists():
        try:
            audio_data, audio_sr = audio_processor.load_wav(str(audio_path))
            print(f"   Loaded Audio: {len(audio_data)/audio_sr:.1f}s at {audio_sr}Hz")
        except Exception as e:
            print(f"   Warning: Audio load failed: {e}")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 15.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    frame_count = 0
    dt = 1.0 / fps

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_count += 1
        current_time_sec = frame_count / fps
        if current_time_sec > max_seconds:
            break

        wallclock = datetime.now().strftime("%H:%M:%S")

        # 1. Vision: Detection & Tracking
        detected_faces = detector.detect(frame)
        tracked_students = tracker.update(detected_faces, frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. Extract Individual Features
        student_raw_data = {}
        for track in tracked_students:
            t_id = track.track_id
            pose = pose_estimator.estimate(track.face)
            motion = motion_analyzer.analyze(t_id, gray, track.bbox)
            inactivity = drowsiness_detector.update(t_id, pose, motion, dt=dt)
            student_raw_data[t_id] = (track.bbox, pose, motion, inactivity)

        # 3. Audio Features
        if audio_data is not None:
            sample_idx = int(current_time_sec * audio_sr)
            chunk_len = int(dt * audio_sr)
            chunk = audio_data[sample_idx:sample_idx + chunk_len] if sample_idx < len(audio_data) else np.zeros(64)
            audio_feats = audio_processor.extract_features(chunk, sr=audio_sr)
        else:
            audio_feats = AudioFeatures(0.02, 0.08, 1200.0, 2000.0, 160.0, True)

        audio_ctx = vad.classify(audio_feats)

        # 4. Multimodal Fusion
        snapshot = fusion_engine.step(
            current_time_sec=current_time_sec,
            dt=dt,
            student_data=student_raw_data,
            audio_ctx=audio_ctx,
            frame_shape=(height, width)
        )

        # 5. Sustained Presence (Scenario 1)
        active_ids = [t.track_id for t in tracked_students if t.is_active]
        eng_map = {t_id: m.composite_engagement for t_id, m in snapshot.student_metrics.items()}
        presence_manager.update(active_ids, dt=dt, current_time_sec=current_time_sec, engagement_map=eng_map)

        # 6. AI Teaching Advisor (Scenario 3)
        suggestion = advisor.evaluate(snapshot, fusion_engine.events, wallclock)

        # 7. Render Annotations
        annotated = frame.copy()
        if privacy_blur:
            bboxes = [t.bbox for t in tracked_students]
            annotated = ParticipantTracker.apply_privacy_blur(annotated, bboxes, kernel_size=35)

        for track in tracked_students:
            x, y, w, h = track.bbox
            m = snapshot.student_metrics.get(track.track_id)
            score = m.composite_engagement if m else 70.0
            color = (60, 180, 75) if score >= 75 else ((0, 165, 255) if score >= 50 else (50, 50, 220))

            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(annotated, (x, max(0, y - 22)), (x + w, y), color, -1)
            cv2.putText(annotated, f"{track.track_id}: {score:.0f}%", (x + 4, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

        # Overlay session status bar
        cv2.rectangle(annotated, (0, 0), (width, 30), (30, 30, 30), -1)
        cv2.putText(annotated, f"TIME: {current_time_sec:04.1f}s | ENGAGEMENT: {snapshot.mean_engagement:.0f}% | ACTIVE: {len(active_ids)} | AUDIO: {audio_ctx.mode.value[:18]}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

        out_writer.write(annotated)

    cap.release()
    out_writer.release()

    # Export Attendance CSV
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    presence_manager.export_csv(output_csv_path)

    # Generate Executive Intelligence Report
    reporter = SessionAnalyticsReport(presence_manager, fusion_engine, advisor)
    md_report = reporter.generate_markdown_report()
    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        f.write(md_report)

    # Print Summary Console Output
    stats = presence_manager.get_summary_stats()
    metrics = reporter.compute_metrics()
    print("\n" + "=" * 60)
    print("🎯 CLASSROOM SESSION PROCESSING COMPLETE")
    print("=" * 60)
    print(f"⏱️  Duration: {metrics.total_duration_sec:.1f}s")
    print(f"👥 Tracked Participants: {stats['total_tracked']}")
    print(f"✅ Verified Present Attendees: {stats['official_attendees']} (Rate: {stats['attendance_rate_pct']}%)")
    print(f"🚫 Transient Passersby Filtered: {stats['transient_filtered']}")
    print(f"📊 Mean Classroom Engagement: {metrics.mean_class_engagement}% (Peak: {metrics.peak_engagement}%)")
    print(f"⚡ Attention Drop Episodes: {metrics.attention_drop_count}")
    print(f"🌟 Interaction Peak Episodes: {metrics.interaction_peak_count}")
    print(f"💡 AI Teaching Suggestions Triggered: {len(advisor.suggestion_history)}")
    print(f"🏆 Overall Session Effectiveness Score: {metrics.overall_session_score} / 100")
    print("\n📁 Exported Files:")
    print(f"   - Attendance Register CSV: {output_csv_path}")
    print(f"   - Executive Markdown Report: {output_report_path}")
    print(f"   - Annotated Video: {output_video_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal Classroom Agent Headless Runner")
    parser.add_argument("--video", type=str, default="data/sample_classroom.mp4", help="Path to input video")
    parser.add_argument("--audio", type=str, default="data/sample_classroom.wav", help="Path to companion audio")
    parser.add_argument("--out-video", type=str, default="exports/annotated_classroom.mp4", help="Output annotated video")
    parser.add_argument("--out-csv", type=str, default="exports/attendance_register.csv", help="Output attendance CSV")
    parser.add_argument("--out-report", type=str, default="exports/classroom_report.md", help="Output markdown report")
    parser.add_argument("--min-presence", type=float, default=6.0, help="Min sustained presence in seconds")
    parser.add_argument("--tardy", type=float, default=12.0, help="Tardy threshold in seconds")
    parser.add_argument("--blur", action="store_true", help="Enable privacy face anonymization")
    args = parser.parse_args()

    v_in = Path(args.video)
    a_in = Path(args.audio)
    if not v_in.exists() or not a_in.exists():
        print("Sample data not found. Generating sample classroom video and audio...")
        generate_classroom_demo(v_in, a_in, duration_sec=25)

    run_pipeline(
        video_path=v_in,
        audio_path=a_in,
        output_video_path=Path(args.out_video),
        output_csv_path=Path(args.out_csv),
        output_report_path=Path(args.out_report),
        min_sustained_sec=args.min_presence,
        tardy_sec=args.tardy,
        privacy_blur=args.blur
    )
