"""
Demo script showing how the SIMPLE RL system learns from feedback.
No external libraries - just gradient-based learning!
"""
from rl_system import RLAdaptiveFusion

# Create RL fusion with your specified weights
rl_fusion = RLAdaptiveFusion(initial_video_weight=0.8, learning_rate=0.05)

print("=" * 70)
print("SIMPLE REINFORCEMENT LEARNING DEMO (No External Libraries)")
print("=" * 70)

print("\n🎯 Initial Weights:")
print(f"  Video: {rl_fusion.video_weight:.3f} (80%)")
print(f"  Audio: {rl_fusion.audio_weight:.3f} (20%)")
print("\nStarting with strong bias towards video!")

# Simulate some predictions and feedback
print("\n📊 Simulating 5 predictions with user feedback...\n")
print("=" * 70)

# Scenario 1: Model was CORRECT - reinforces current weights
print("\n1️⃣  Scenario: Model Correct (FAKE)")
print("-" * 70)
prediction_1 = {
    'verdict': 'FAKE',
    'confidence': 0.85,
    'fake_probability': 0.75,
    'video_fake_score': 0.8,  # Video: 80% fake
    'audio_fake_score': 0.6,  # Audio: 60% fake
}
print(f"Prediction: FAKE (75% confidence)")
print(f"  Video: 80% fake | Audio: 60% fake")
print(f"User feedback: FAKE ✓ (Correct!)")
rl_fusion.update_from_feedback(prediction_1, user_label='FAKE', user_confidence=1.0)
print(f"→ Weights: video={rl_fusion.video_weight:.3f}, audio={rl_fusion.audio_weight:.3f}")
print("  (Small adjustment - model was correct)")

# Scenario 2: Video was WRONG, Audio was RIGHT
print("\n2️⃣  Scenario: Video Wrong, Audio Right")
print("-" * 70)
prediction_2 = {
    'verdict': 'FAKE',
    'confidence': 0.7,
    'fake_probability': 0.68,
    'video_fake_score': 0.85,  # Video strongly said FAKE (wrong!)
    'audio_fake_score': 0.15,  # Audio said REAL (correct!)
}
print(f"Prediction: FAKE (68% confidence)")
print(f"  Video: 85% fake | Audio: 15% fake")
print(f"User feedback: REAL ✗ (Wrong! Audio was right)")
rl_fusion.update_from_feedback(prediction_2, user_label='REAL', user_confidence=1.0)
print(f"→ Weights: video={rl_fusion.video_weight:.3f}, audio={rl_fusion.audio_weight:.3f}")
print("  (Decreased video weight - it was wrong, trust audio more!)")

# Scenario 3: Audio was WRONG, Video was RIGHT
print("\n3️⃣  Scenario: Audio Wrong, Video Right")
print("-" * 70)
prediction_3 = {
    'verdict': 'REAL',
    'confidence': 0.6,
    'fake_probability': 0.3,
    'video_fake_score': 0.2,  # Video said REAL (correct!)
    'audio_fake_score': 0.7,  # Audio said FAKE (wrong!)
}
print(f"Prediction: REAL (30% fake)")
print(f"  Video: 20% fake | Audio: 70% fake")
print(f"User feedback: REAL ✓ (Correct! Video was right)")
rl_fusion.update_from_feedback(prediction_3, user_label='REAL', user_confidence=1.0)
print(f"→ Weights: video={rl_fusion.video_weight:.3f}, audio={rl_fusion.audio_weight:.3f}")
print("  (Increased video weight slightly - it was more accurate)")

# Scenario 4: Both wrong, but video less wrong
print("\n4️⃣  Scenario: Both Wrong, Video Less Wrong")
print("-" * 70)
prediction_4 = {
    'verdict': 'REAL',
    'confidence': 0.5,
    'fake_probability': 0.4,
    'video_fake_score': 0.5,  # Video uncertain (closer to truth)
    'audio_fake_score': 0.1,  # Audio very confident REAL (very wrong!)
}
print(f"Prediction: REAL (40% fake)")
print(f"  Video: 50% fake | Audio: 10% fake")
print(f"User feedback: FAKE ✗ (Wrong! But video was closer)")
rl_fusion.update_from_feedback(prediction_4, user_label='FAKE', user_confidence=0.8)
print(f"→ Weights: video={rl_fusion.video_weight:.3f}, audio={rl_fusion.audio_weight:.3f}")
print("  (Increased video weight - it was less wrong)")

# Scenario 5: Perfect prediction
print("\n5️⃣  Scenario: Perfect Prediction")
print("-" * 70)
prediction_5 = {
    'verdict': 'SUSPICIOUS',
    'confidence': 0.9,
    'fake_probability': 0.55,
    'video_fake_score': 0.6,
    'audio_fake_score': 0.45,
}
print(f"Prediction: SUSPICIOUS (55% fake)")
print(f"  Video: 60% fake | Audio: 45% fake")
print(f"User feedback: SUSPICIOUS ✓ (Perfect!)")
rl_fusion.update_from_feedback(prediction_5, user_label='SUSPICIOUS', user_confidence=1.0)
print(f"→ Weights: video={rl_fusion.video_weight:.3f}, audio={rl_fusion.audio_weight:.3f}")
print("  (Maintained weights - prediction was excellent)")

# Show statistics
stats = rl_fusion.get_performance_stats()
print("\n" + "=" * 70)
print("📈 FINAL STATISTICS:")
print("=" * 70)
print(f"Total Feedback: {stats['total']}")
print(f"Accuracy: {stats['accuracy']*100:.1f}%")
print(f"Correct Predictions: {stats['correct']}")
print(f"Incorrect Predictions: {stats['incorrect']}")
print(f"\n🎯 Learned Weights:")
print(f"  Video: {stats['current_weights']['video']:.3f} ({stats['current_weights']['video']*100:.1f}%)")
print(f"  Audio: {stats['current_weights']['audio']:.3f} ({stats['current_weights']['audio']*100:.1f}%)")

if 'avg_reward' in stats:
    print(f"\n⭐ Average Reward: {stats['avg_reward']:.2f}")

# Save the model
rl_fusion.save_model()
print("\n💾 Learned weights saved to 'rl_weights.json'")

print("\n" + "=" * 70)
print("🧠 KEY INSIGHTS:")
print("=" * 70)
print("✅ Simple gradient-based learning (no external libraries!)")
print("✅ Learns which modality (video/audio) is more reliable")
print("✅ Adjusts weights based on feedback")
print("✅ Saves and loads learned weights")
print("✅ The more feedback, the smarter it gets!")
print("=" * 70)
