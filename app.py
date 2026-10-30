"""
Multimodal Attendance & Engagement Agent - Interactive Streamlit Dashboard.
Integrates Computer Vision, Audio Signal Processing, and AI-Driven Decision Support
into a privacy-preserving, 100% offline classroom intelligence platform.
"""
from datetime import datetime
from pathlib import Path
import tempfile
import time
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
    YUNET_MODEL_PATH,
    DATA_DIR,
    EXPORTS_DIR,
    VisionConfig,
    AudioConfig,
    PresenceConfig,
    FusionConfig,
    AdvisorConfig
)
from vision.face_detector import FaceDetector
from vision.pose_estimator import HeadPoseEstimator, GazeState
from vision.motion_analyzer import MotionAnalyzer, MotionState
from vision.drowsiness_detector import DrowsinessDetector
from core.tracker import ParticipantTracker
from core.presence import PresenceManager, AttendanceStatus
from audio.signal_processor import AudioSignalProcessor, AudioFeatures
from audio.vad import VoiceActivityDetector, AcousticMode, AudioContext
from fusion.engagement_engine import MultimodalEngagementEngine, EventType
from agent.teaching_advisor import TeachingAdvisorAgent, SuggestionPriority
from agent.session_analyzer import SessionAnalyticsReport
from demo_generator import generate_classroom_demo


# Page layout
st.set_page_config(
    page_title="Multimodal Attendance & Engagement Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .kpi-card {
        background: #F8FAFC;
        border-radius: 8px;
        padding: 12px 18px;
        border-left: 4px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .alert-card-high {
        background: #FEF2F2;
        border-left: 5px solid #EF4444;
        padding: 14px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    .alert-card-action {
        background: #FFFBEB;
        border-left: 5px solid #F59E0B;
        padding: 14px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    .alert-card-flow {
        background: #F0FDF4;
        border-left: 5px solid #10B981;
        padding: 14px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


def get_color_for_engagement(score: float) -> tuple[int, int, int]:
    """Return BGR color for bounding boxes based on engagement score."""
    if score >= 75.0:
        return (60, 180, 75)   # Green (High Focus)
    elif score >= 50.0:
        return (0, 165, 255)   # Orange / Yellow (Moderate / Distracted)
    else:
        return (50, 50, 220)   # Red (Disengaged / Slumped)


def render_annotated_frame(
    frame: np.ndarray,
    tracked_students,
    student_metrics,
    privacy_blur: bool
) -> np.ndarray:
    """Render bounding boxes, gaze vectors, and engagement tags on frame."""
    annotated = frame.copy()

    # Apply privacy blur first if enabled
    if privacy_blur:
        bboxes = [t.bbox for t in tracked_students]
        annotated = ParticipantTracker.apply_privacy_blur(annotated, bboxes, kernel_size=35)

    for track in tracked_students:
        t_id = track.track_id
        x, y, w, h = track.bbox

        # Metrics for this student
        metrics = student_metrics.get(t_id)
        score = metrics.composite_engagement if metrics else 70.0
        color = get_color_for_engagement(score)

        # Draw rounded rectangle bounding box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

        # Draw banner header
        banner_h = 24
        cv2.rectangle(annotated, (x, max(0, y - banner_h)), (x + w, y), color, -1)
        
        # Student ID & Engagement
        label = f"{t_id}: {score:.0f}%"
        cv2.putText(annotated, label, (x + 5, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Draw gaze / status badge below bbox
        if metrics:
            status_text = f"{metrics.spatial_quadrant}"
            if metrics.is_drowsy:
                status_text += " [DROWSY]"
            cv2.putText(annotated, status_text, (x, y + h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)

    return annotated


def main():
    # Sidebar
    st.sidebar.title("🎛️ Classroom Agent Controls")

    # Source Selection
    source_mode = st.sidebar.radio(
        "Input Feed Source",
        ["Classroom Demo (4 Students + Sync Audio)", "Upload Video & Audio", "Live Webcam"],
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🛡️ Edge Privacy & Compliance")
    privacy_blur = st.sidebar.checkbox("Face Anonymization (Blurring)", value=False,
                                       help="Blurs faces on device before display to ensure FERPA/GDPR compliance.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Attendance Thresholds")
    min_sustained_sec = st.sidebar.slider("Min Sustained Presence (sec)", 2.0, 30.0, 6.0, 1.0,
                                          help="Minimum continuous presence required to mark attendance.")
    tardy_sec = st.sidebar.slider("Tardy Arrival Cutoff (sec)", 5.0, 60.0, 12.0, 1.0,
                                  help="Students arriving after this are tagged as Late Arrival.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚖️ Multimodal Fusion Weights")
    w_vis = st.sidebar.slider("Visual Attention (Gaze)", 0.0, 1.0, 0.50, 0.05)
    w_mot = st.sidebar.slider("Motion Vitality (Posture)", 0.0, 1.0, 0.25, 0.05)
    w_aud = st.sidebar.slider("Audio Synergy (Context)", 0.0, 1.0, 0.25, 0.05)
    # Normalize weights
    total_w = w_vis + w_mot + w_aud
    if total_w > 0:
        w_vis, w_mot, w_aud = w_vis / total_w, w_mot / total_w, w_aud / total_w

    # Header
    st.markdown('<div class="main-header">🎓 Multimodal Attendance & Engagement Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Privacy-Preserving Classroom Intelligence • Edge Computer Vision & Audio Signal Processing</div>', unsafe_allow_html=True)

    # Initialize components
    video_path = None
    audio_path = None

    if source_mode == "Classroom Demo (4 Students + Sync Audio)":
        v_demo = DATA_DIR / "sample_classroom.mp4"
        a_demo = DATA_DIR / "sample_classroom.wav"
        if not v_demo.exists() or not a_demo.exists():
            with st.spinner("Generating sample multimodal classroom dataset..."):
                generate_classroom_demo(v_demo, a_demo, duration_sec=25)
        video_path = str(v_demo)
        audio_path = str(a_demo)
        st.info("ℹ️ Running built-in **Sample Classroom Simulation**: 4 students (Alex: Attentive, Sam: Distracted, Jordan: Drowsy, Taylor: Late Arrival) + 1 Transient Passerby.")

    elif source_mode == "Upload Video & Audio":
        uploaded_vid = st.file_uploader("Upload Classroom Video (MP4 / AVI / MOV)", type=["mp4", "avi", "mov"])
        uploaded_aud = st.file_uploader("Upload Synchronized Audio (Optional WAV)", type=["wav"])

        if uploaded_vid is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_vid.read())
            video_path = tfile.name

        if uploaded_aud is not None:
            afile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            afile.write(uploaded_aud.read())
            audio_path = afile.name

    elif source_mode == "Live Webcam":
        video_path = 0  # Webcam index

    # UI Tabs
    tab_live, tab_attendance, tab_dynamics, tab_advisor, tab_report = st.tabs([
        "🎥 Live Classroom Feed",
        "📋 Attendance Register (Scenario 1)",
        "📈 Engagement Dynamics (Scenario 2)",
        "💡 AI Teaching Suggestions (Scenario 3)",
        "📑 Executive Session Report"
    ])

    # Run processing button
    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        start_processing = st.button("▶️ Start Classroom Analysis", type="primary")

    if start_processing:
        if video_path is None:
            st.error("Please provide a valid video input source.")
            return

        # Instantiate pipeline engines
        detector = FaceDetector(model_path=YUNET_MODEL_PATH)
        pose_estimator = HeadPoseEstimator()
        motion_analyzer = MotionAnalyzer()
        drowsiness_detector = DrowsinessDetector(inactivity_threshold_sec=8.0)
        tracker = ParticipantTracker(iou_threshold=0.25)
        presence_manager = PresenceManager(
            min_sustained_seconds=min_sustained_sec,
            tardy_threshold_seconds=tardy_sec
        )
        audio_processor = AudioSignalProcessor(sample_rate=16000)
        vad = VoiceActivityDetector()
        fusion_engine = MultimodalEngagementEngine(
            weight_visual=w_vis,
            weight_motion=w_mot,
            weight_audio=w_aud,
            attention_drop_thresh=45.0,
            interaction_peak_thresh=75.0
        )
        advisor = TeachingAdvisorAgent(cooldown_sec=6.0)

        # Load audio track if available
        audio_data = None
        audio_sr = 16000
        if audio_path and Path(audio_path).exists():
            try:
                audio_data, audio_sr = audio_processor.load_wav(audio_path)
            except Exception as e:
                st.warning(f"Audio track could not be loaded: {e}. Running with synthetic acoustic baseline.")

        # Open video capture
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 15.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if isinstance(video_path, str) else 150
        max_frames_to_process = min(total_frames, 400) if total_frames > 0 else 200

        with tab_live:
            # Layout: Video player and live KPI columns
            banner_placeholder = st.empty()
            col_vid, col_kpi = st.columns([2.5, 1.2])

            with col_vid:
                video_placeholder = st.empty()
                progress_bar = st.progress(0.0)

            with col_kpi:
                metric_active = st.empty()
                metric_eng = st.empty()
                metric_pres = st.empty()
                metric_mode = st.empty()

        frame_count = 0
        frame_skip = 2  # Process every 2nd frame for smooth speed
        dt = frame_skip / fps

        # Main Real-Time Inference Loop
        while cap.isOpened() and frame_count < max_frames_to_process:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            current_time_sec = frame_count / fps
            wallclock_str = datetime.now().strftime("%H:%M:%S")

            # 1. Vision Pipeline: Face Detection & Multi-Object Tracking
            detected_faces = detector.detect(frame)
            tracked_students = tracker.update(detected_faces, frame)

            # Gray frame for motion differencing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 2. Extract Individual Behavioral Indicators
            student_raw_data = {}
            for track in tracked_students:
                t_id = track.track_id
                # Head pose estimation
                pose = pose_estimator.estimate(track.face)
                # Motion dynamics
                motion = motion_analyzer.analyze(t_id, gray, track.bbox)
                # Inactivity / drowsiness tracking
                inactivity = drowsiness_detector.update(t_id, pose, motion, dt=dt)

                student_raw_data[t_id] = (track.bbox, pose, motion, inactivity)

            # 3. Audio Signal Processing
            if audio_data is not None:
                sample_idx = int(current_time_sec * audio_sr)
                chunk_len = int(dt * audio_sr)
                chunk = audio_data[sample_idx:sample_idx + chunk_len] if sample_idx < len(audio_data) else np.zeros(64)
                audio_feats = audio_processor.extract_features(chunk, sr=audio_sr)
            else:
                # Baseline simulated audio features
                audio_feats = AudioFeatures(0.02, 0.08, 1200.0, 2000.0, 160.0, True)

            audio_ctx = vad.classify(audio_feats)

            # 4. Multimodal Signal Fusion
            snapshot = fusion_engine.step(
                current_time_sec=current_time_sec,
                dt=dt,
                student_data=student_raw_data,
                audio_ctx=audio_ctx,
                frame_shape=frame.shape
            )

            # 5. Sustained Presence & Attendance State Machine (Scenario 1)
            active_ids = [t.track_id for t in tracked_students if t.is_active]
            eng_map = {t_id: m.composite_engagement for t_id, m in snapshot.student_metrics.items()}
            presence_manager.update(
                active_track_ids=active_ids,
                dt=dt,
                current_time_sec=current_time_sec,
                engagement_map=eng_map
            )

            # 6. AI Teaching Advisor Decision-Support (Scenario 3)
            active_suggestion = advisor.evaluate(
                snapshot=snapshot,
                recent_events=fusion_engine.events,
                wallclock_time=wallclock_str
            )

            # 7. Render UI Visualizations
            annotated_frame = render_annotated_frame(
                frame=frame,
                tracked_students=tracked_students,
                student_metrics=snapshot.student_metrics,
                privacy_blur=privacy_blur
            )

            # Convert to RGB for Streamlit display
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, use_container_width=True)
            progress_bar.progress(min(1.0, frame_count / max_frames_to_process))

            # Display active recommendation banner
            if active_suggestion:
                priority_class = "alert-card-flow"
                if active_suggestion.priority == SuggestionPriority.HIGH_PRIORITY:
                    priority_class = "alert-card-high"
                elif active_suggestion.priority == SuggestionPriority.ACTION_RECOMMENDED:
                    priority_class = "alert-card-action"

                banner_placeholder.markdown(f"""
                <div class="{priority_class}">
                    <strong>{active_suggestion.headline}</strong> ({active_suggestion.priority.value})<br>
                    💡 <em>{active_suggestion.actionable_tip}</em><br>
                    <small style="color: #64748B;">Rationale: {active_suggestion.pedagogical_rationale}</small>
                </div>
                """, unsafe_allow_html=True)

            # Update KPI counters
            presence_stats = presence_manager.get_summary_stats()
            metric_active.metric("Active Learners", f"{len(active_ids)} students")
            metric_eng.metric("Mean Engagement", f"{snapshot.mean_engagement}%",
                              delta=f"{round(snapshot.mean_engagement - 65.0, 1)}% vs target")
            metric_pres.metric("Verified Present", f"{presence_stats['present_verified']} / {presence_stats['total_tracked'] - presence_stats['transient_filtered']}")
            metric_mode.metric("Acoustic State", f"{audio_ctx.mode.value[:18]}...")

        cap.release()
        st.success("✅ Classroom session analysis completed!")

        # Populate Tab 1: Attendance Register
        with tab_attendance:
            st.subheader("📋 Automated Attendance Register (Scenario 1)")
            st.caption("Attendance is verified based on sustained presence rather than single-frame detections.")

            p_stats = presence_manager.get_summary_stats()
            col_a1, col_a2, col_a3, col_a4 = st.columns(4)
            col_a1.metric("Official Attendees", p_stats["official_attendees"])
            col_a2.metric("Late Arrivals", p_stats["late_arrivals"])
            col_a3.metric("Passersby Filtered", p_stats["transient_filtered"], help="Transient appearances < threshold filtered out")
            col_a4.metric("Attendance Rate", f"{p_stats['attendance_rate_pct']}%")

            att_df = presence_manager.to_dataframe()
            st.dataframe(att_df, use_container_width=True)

            # CSV Download
            csv_data = att_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Official Attendance Register (CSV)",
                data=csv_data,
                file_name="classroom_attendance_register.csv",
                mime="text/csv"
            )

            with st.expander("🔍 View Timestamped Presence Verification Audit Log"):
                audit_df = presence_manager.export_audit_log_dataframe()
                st.dataframe(audit_df, use_container_width=True)

        # Populate Tab 2: Engagement Dynamics
        with tab_dynamics:
            st.subheader("📈 Multimodal Engagement & Session Dynamics (Scenario 2)")
            st.caption("Temporal fusion of head pose, movement patterns, inactivity, and acoustic modes.")

            timeline = fusion_engine.timeline
            if timeline:
                times = [s.timestamp_sec for s in timeline]
                engs = [s.mean_engagement for s in timeline]

                # Main Engagement Plot with annotated events
                fig_eng = go.Figure()
                fig_eng.add_trace(go.Scatter(
                    x=times, y=engs,
                    mode='lines+markers',
                    name='Class Average Engagement',
                    line=dict(color='#2563EB', width=3),
                    marker=dict(size=4)
                ))

                # Highlight Attention Drops (< 45%)
                fig_eng.add_hline(y=45.0, line_dash="dash", line_color="#EF4444",
                                  annotation_text="Attention Drop Threshold (45%)")
                fig_eng.add_hline(y=75.0, line_dash="dash", line_color="#10B981",
                                  annotation_text="High Focus Target (75%)")

                # Annotate events
                for ev in fusion_engine.events:
                    color = "#EF4444" if ev.event_type == EventType.ATTENTION_DROP else "#10B981"
                    fig_eng.add_vline(x=ev.timestamp_sec, line_width=1.5, line_color=color, line_dash="dot")
                    fig_eng.add_annotation(
                        x=ev.timestamp_sec, y=ev.score,
                        text=ev.event_type.value,
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor=color,
                        font=dict(size=10, color=color)
                    )

                fig_eng.update_layout(
                    title="Classroom Engagement Trajectory Over Time",
                    xaxis_title="Session Elapsed Time (seconds)",
                    yaxis_title="Engagement Score (%)",
                    yaxis=dict(range=[0, 105]),
                    hovermode="x unified",
                    template="plotly_white"
                )
                st.plotly_chart(fig_eng, use_container_width=True)

                # Acoustic Mode & Quadrant Breakdown
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    # Acoustic distribution
                    modes = [s.audio_mode for s in timeline]
                    mode_counts = pd.Series(modes).value_counts().reset_index()
                    mode_counts.columns = ["Acoustic Mode", "Count"]
                    fig_pie = px.pie(
                        mode_counts,
                        names="Acoustic Mode",
                        values="Count",
                        title="Acoustic Classroom Context Breakdown",
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                with col_d2:
                    # Spatial Quadrant Heatmap / Bar
                    quad_data = {"Front-Left": [], "Front-Right": [], "Back-Left": [], "Back-Right": []}
                    for s in timeline:
                        for q, val in s.quadrant_averages.items():
                            quad_data[q].append(val)
                    quad_means = {q: np.mean(v) if v else 50.0 for q, v in quad_data.items()}
                    df_quad = pd.DataFrame(list(quad_means.items()), columns=["Quadrant", "Avg Engagement %"])
                    fig_bar = px.bar(
                        df_quad,
                        x="Quadrant",
                        y="Avg Engagement %",
                        color="Avg Engagement %",
                        color_continuous_scale="Viridis",
                        title="Spatial Engagement by Classroom Zone"
                    )
                    fig_bar.update_layout(yaxis=dict(range=[0, 100]))
                    st.plotly_chart(fig_bar, use_container_width=True)

        # Populate Tab 3: AI Teaching Suggestions
        with tab_advisor:
            st.subheader("💡 AI Pedagogical Decision Support (Scenario 3)")
            st.caption("Context-aware, non-intrusive teaching suggestions generated from multimodal classroom dynamics.")

            if advisor.suggestion_history:
                for idx, sug in enumerate(advisor.suggestion_history, 1):
                    with st.container():
                        badge_color = "🔴" if sug.priority == SuggestionPriority.HIGH_PRIORITY else ("🟡" if sug.priority == SuggestionPriority.ACTION_RECOMMENDED else "🟢")
                        st.markdown(f"#### {badge_color} [{sug.wallclock} (t={sug.timestamp_sec:.1f}s)] {sug.headline}")
                        st.markdown(f"**Recommended Action**: {sug.actionable_tip}")
                        st.markdown(f"*Pedagogical Rationale*: {sug.pedagogical_rationale}")
                        st.markdown("---")
            else:
                st.info("Classroom sustained balanced engagement throughout the session; no critical intervention was triggered.")

        # Populate Tab 4: Executive Session Report
        with tab_report:
            st.subheader("📑 Executive Classroom Intelligence Report")
            reporter = SessionAnalyticsReport(presence_manager, fusion_engine, advisor)
            md_report = reporter.generate_markdown_report()
            st.markdown(md_report)

            # Download report button
            st.download_button(
                label="📥 Download Executive Intelligence Report (Markdown)",
                data=md_report,
                file_name=f"classroom_intelligence_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )


if __name__ == "__main__":
    main()
