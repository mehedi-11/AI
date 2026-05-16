# 🎓 ELINA: Exam Instructor AI (Multi-Modal with Voice)

This is a comprehensive AI solution for automated exam proctoring and student assistance. It features **Vision** for seeing, **Transformer Logic** for thinking, and **Voice Synthesis** for speaking.

## ✨ Core Features
1.  **AI Proctoring**: Monitors students via video/camera and issues **verbal warnings** if cheating is detected.
2.  **Voice Interaction**: Speaks to students and teachers using a human-like voice for instructions and alerts.
3.  **Dynamic Questioning**: Generates unique question patterns for every exam session.
4.  **Personalized Analytics**: Analyzes student performance and verbally summarizes improvement reports.

## 📁 Project Structure
- **`neural_network_core.py`**: Multi-modal architecture (CNN + Transformer).
- **`learning_engine.py`**: Training logic with integrated **Voice Engine (pyttsx3)**.
- **`main_application.py`**: The demo script with voice greetings and instructions.
- **`requirements.txt`**: Libraries (`torch`, `torchvision`, `opencv-python`, `pyttsx3`).

## 🚀 How to Execute
To hear the AI in action:
```powershell
python main_application.py
```

## 🛠️ Important Notes
- **Voices**: You can change the voice gender and speed in the `AIBrain.__init__` method in `learning_engine.py`.
- **Training**: The AI starts with a random voice logic. Train it with specific datasets to make its decision-making (proctoring/analytics) more accurate.

---
*Bridging the gap between automated integrity and interactive education.*
