"""
Gradio interface with Simple RL feedback integration.
Uses gradient-based learning WITHOUT external libraries.
"""
import gradio as gr
from detector import DeepFakeDetector
import logging
from PIL import Image
import numpy as np
from config import Config

logger = logging.getLogger(__name__)

# Initialize the detector
print("Loading models...")
detector = DeepFakeDetector()
print("Ready!")

# Track last prediction for feedback
last_prediction = {"video": None, "image": None, "audio": None}


# ===== VIDEO ANALYSIS =====
def analyze_video_mode(video_path):
    """Analyze video with audio+visual cues and security checks."""
    if video_path is None:
        return "ERROR", 0.0, "Please upload a video", "", "", ""
    
    results = detector.analyze_video(video_path)
    last_prediction["video"] = results  # Store for feedback
    
    verdict = results['verdict']
    confidence = results['confidence']
    video_score = results.get('video_fake_score', 0.0)
    audio_score = results.get('audio_fake_score', 0.0)
    analysis_source = results.get('analysis_source', 'local')
    
    # Security alerts
    security_alert = results.get('security_alert')
    threat_vector = results.get('threat_vector')
    
    info = f"Video: {video_score:.1%} | Audio: {audio_score:.1%}" if audio_score else f"Video: {video_score:.1%}"
    
    warning = ""
    if security_alert == "MISMATCH_DETECTED":
        if threat_vector == "FACE_SWAP_ATTACK":
            warning = "⚠️ ALERT: Possible face swap with real audio detected!"
        elif threat_vector == "VOICE_CLONE_ATTACK":
            warning = "⚠️ ALERT: Possible voice cloning with real video detected!"
        else:
            warning = "⚠️ ALERT: Suspicious mismatch between audio and video!"
    elif security_alert == "LOW_CONFIDENCE":
        warning = "⚠️ Low confidence - manual review recommended"
    
    source_badge = f"🤖 Analyzed by: {'Gemini API' if analysis_source == 'gemini' else 'Local Models'}"
    
    # Get current RL weights if using RL mode
    weights_info = ""
    if hasattr(detector.fusion_strategy, 'video_weight'):
        v_w = detector.fusion_strategy.video_weight
        a_w = detector.fusion_strategy.audio_weight
        weights_info = f"📊 Current Weights: Video={v_w:.2f} ({v_w*100:.0f}%), Audio={a_w:.2f} ({a_w*100:.0f}%)"
    
    return verdict, confidence, info, warning, source_badge, weights_info


def submit_video_feedback(feedback_label, confidence_slider):
    """Submit user correction for video analysis."""
    if last_prediction["video"] is None:
        return "❌ No prediction to provide feedback for"
    
    if Config.FUSION_MODE == 'rl_adaptive':
        detector.fusion_strategy.update_from_feedback(
            last_prediction["video"],
            feedback_label,
            confidence_slider / 100.0
        )
        
        # Save learned weights
        detector.fusion_strategy.save_model()
        
        # Get stats
        stats = detector.fusion_strategy.get_performance_stats()
        v_w = detector.fusion_strategy.video_weight
        a_w = detector.fusion_strategy.audio_weight
        
        feedback_msg = f"""✅ Feedback Recorded!
        
📈 Statistics:
  • Total Feedback: {stats['total']}
  • Accuracy: {stats.get('accuracy', 0)*100:.1f}%
  • Correct: {stats.get('correct', 0)} | Incorrect: {stats.get('incorrect', 0)}

🎯 Updated Weights:
  • Video: {v_w:.3f} ({v_w*100:.1f}%)
  • Audio: {a_w:.3f} ({a_w*100:.1f}%)
  
The model is learning from your feedback!"""
        
        return feedback_msg
    else:
        return "⚠️ RL mode not enabled. Feedback not recorded."


# ===== IMAGE ANALYSIS =====
def analyze_image_mode(image):
    """Analyze single image."""
    if image is None:
        return "ERROR", 0.0, "Please upload an image", ""
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'), 'RGB')
    
    results = detector.analyze_image(image)
    last_prediction["image"] = results
    
    verdict = results['verdict']
    confidence = results['confidence']
    fake_prob = results['fake_probability']
    analysis_source = results.get('analysis_source', 'local')
    
    info = f"Fake probability: {fake_prob:.1%}"
    source_badge = f"🤖 Analyzed by: {'Gemini API' if analysis_source == 'gemini' else 'Local Model'}"
    
    return verdict, confidence, info, source_badge


# ===== AUDIO ANALYSIS =====
def analyze_audio_mode(audio_path):
    """Analyze audio only."""
    if audio_path is None:
        return "ERROR", 0.0, "Please upload an audio file"
    
    results = detector.analyze_audio(audio_path)
    last_prediction["audio"] = results
    
    verdict = results['verdict']
    confidence = results['confidence']
    fake_prob = results['fake_probability']
    
    info = f"Fake probability: {fake_prob:.1%}"
    
    return verdict, confidence, info


# ===== BUILD UI =====
with gr.Blocks(title="DeepFake Detector with RL") as app:
    gr.Markdown("# 🔍 DeepFake Detector with Reinforcement Learning")
    gr.Markdown(f"### Mode: **{Config.FUSION_MODE.upper()}** | Initial Weights: Video={Config.VIDEO_WEIGHT*100:.0f}%, Audio={Config.AUDIO_WEIGHT*100:.0f}%")
    
    with gr.Tabs():
        # ==================== VIDEO TAB ====================
        with gr.Tab("🎥 Video (Audio + Visual)"):
            with gr.Row():
                video_input = gr.Video(label="Upload Video")
                with gr.Column():
                    video_verdict = gr.Textbox(label="Verdict", interactive=False)
                    video_confidence = gr.Number(label="Confidence", interactive=False)
                    video_info = gr.Textbox(label="Details", interactive=False)
                    video_warning = gr.Textbox(label="Security Alert", interactive=False)
                    video_source = gr.Textbox(label="Analysis Source", interactive=False)
                    video_weights = gr.Textbox(label="RL Weights", interactive=False)
            
            video_btn = gr.Button("🔎 Analyze Video", variant="primary", size="lg")
            video_btn.click(
                fn=analyze_video_mode,
                inputs=video_input,
                outputs=[video_verdict, video_confidence, video_info, video_warning, video_source, video_weights]
            )
            
            # Feedback Section (ALWAYS VISIBLE for RL mode)
            if Config.FUSION_MODE == 'rl_adaptive':
                gr.Markdown("---")
                gr.Markdown("### 📝 Help the AI Learn! (Reinforcement Learning)")
                gr.Markdown("If the prediction was wrong, correct it below. The system will learn from your feedback!")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        feedback_correct = gr.Radio(
                            choices=["REAL", "FAKE", "SUSPICIOUS"],
                            label="✅ Correct Label",
                            info="What should the verdict have been?"
                        )
                    with gr.Column(scale=1):
                        feedback_confidence = gr.Slider(
                            minimum=0, maximum=100, value=100, step=5,
                            label="🎯 Your Confidence (%)",
                            info="How sure are you about this label?"
                        )
                
                feedback_btn = gr.Button("📤 Submit Feedback & Train Model", variant="secondary", size="lg")
                feedback_result = gr.Textbox(label="Feedback Status", interactive=False, lines=8)
                
                feedback_btn.click(
                    fn=submit_video_feedback,
                    inputs=[feedback_correct, feedback_confidence],
                    outputs=feedback_result
                )
                
             
        # ==================== IMAGE TAB ====================
        with gr.Tab("🖼️ Image Only"):
            with gr.Row():
                image_input = gr.Image(label="Upload Image", type="numpy")
                with gr.Column():
                    image_verdict = gr.Textbox(label="Verdict", interactive=False)
                    image_confidence = gr.Number(label="Confidence", interactive=False)
                    image_info = gr.Textbox(label="Details", interactive=False)
                    image_source = gr.Textbox(label="Analysis Source", interactive=False)
            image_btn = gr.Button("🔎 Analyze Image", variant="primary", size="lg")
            image_btn.click(
                fn=analyze_image_mode,
                inputs=image_input,
                outputs=[image_verdict, image_confidence, image_info, image_source]
            )
        
        # ==================== AUDIO TAB ====================
        with gr.Tab("🎵 Audio Only"):
            with gr.Row():
                audio_input = gr.Audio(label="Upload Audio", type="filepath")
                with gr.Column():
                    audio_verdict = gr.Textbox(label="Verdict", interactive=False)
                    audio_confidence = gr.Number(label="Confidence", interactive=False)
                    audio_info = gr.Textbox(label="Details", interactive=False)
            audio_btn = gr.Button("🔎 Analyze Audio", variant="primary", size="lg")
            audio_btn.click(
                fn=analyze_audio_mode,
                inputs=audio_input,
                outputs=[audio_verdict, audio_confidence, audio_info]
            )


if __name__ == "__main__":
    app.launch()
