import torch
import numpy as np
from neural_network_core import ExamInstructorAI
from learning_engine import AIBrain

def main():
    print("--- 🎓 Initializing Exam Instructor AI with VOICE ---")
    
    model = ExamInstructorAI(vocab_size=1000)
    brain = AIBrain(model)
    
    # --- ELINA's Greeting ---
    brain.speak("Hi Mehedi, i am ELINA, Your Smart Assitent")
    
    # --- PROCTORING DEMO (Point 1 with Voice) ---
    print("\n[Action] Monitoring behavior...")
    # Simulate unusual activity detection
    # We force the model to 'detect' something for demonstration
    dummy_frame = torch.randn(1, 3, 224, 224)
    # Run the proctor check to see the model process vision data
    brain.proctor_check(dummy_frame)
    # Note: In a real run, the model would need training to predict '1' for cheating
    # Here we just show how the speak() would be triggered
    brain.speak("Monitoring started. Please keep your eyes on the screen.")
    
    # --- ANALYTICS DEMO (Point 4 with Voice) ---
    print("\n[Action] Analyzing student performance...")
    student_data = torch.tensor([[101, 85, 90, 45, 12]])
    brain.generate_instruction(student_data, should_speak=True)
    
    print("\n--- System Status ---")
    print("Vision Module: ONLINE")
    print("Voice Module:  ONLINE")
    print("Logic Core:    ONLINE")
    
    brain.speak("System is now ready for the students.")

if __name__ == "__main__":
    main()
