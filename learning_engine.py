import torch
import torch.optim as optim
import torch.nn as nn
from gtts import gTTS 
import os
import threading
from neural_network_core import ExamInstructorAI

class AIBrain:
    """
    Intelligence Hub with High-Quality Voice using Windows VBScript for guaranteed playback.
    """
    def __init__(self, model: ExamInstructorAI, lr=0.0001):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.proctor_loss_fn = nn.CrossEntropyLoss()
        self.text_loss_fn = nn.CrossEntropyLoss()

    def speak(self, text):
        """Convert text to human female voice and play in background without blocking."""
        threading.Thread(target=self._speak, args=(text,), daemon=True).start()

    def _speak(self, text):
        """Internal method to handle the actual TTS and VBScript playback."""
        print(f"ELINA: {text}")
        try:
            tts = gTTS(text=text, lang='en', slow=False)
            filename = "temp_voice.mp3"
            tts.save(filename)
            full_path = os.path.abspath(filename)
            
            # Using VBScript for silent, robust background playback on Windows
            vbs_script = f'''
Dim oPlayer
Set oPlayer = CreateObject("WMPlayer.OCX")
oPlayer.URL = "{full_path}"
oPlayer.controls.play
While oPlayer.playState <> 1 ' 1 = Stopped
    WScript.Sleep 100
Wend
oPlayer.close
'''
            vbs_file = "play_audio.vbs"
            with open(vbs_file, "w") as f:
                f.write(vbs_script)
                
            # Run the VBScript
            os.system(f'cscript //nologo "{vbs_file}"')
            
            # Clean up
            if os.path.exists(filename):
                os.remove(filename)
            if os.path.exists(vbs_file):
                os.remove(vbs_file)
                
        except Exception as e:
            print(f"Voice Error: {e}")

    def proctor_check(self, frame_tensor):
        self.model.eval()
        with torch.no_grad():
            output = self.model(vision_input=frame_tensor)
            prediction = torch.argmax(output['proctor_logits'], dim=-1)
            if prediction.item() == 1:
                self.speak("Warning! Unusual activity detected.")
        return prediction.numpy()

    def generate_instruction(self, input_tokens, should_speak=False):
        self.model.eval()
        with torch.no_grad():
            output = self.model(text_input=input_tokens)
            prediction = torch.argmax(output['text_logits'], dim=-1)
            if should_speak:
                self.speak("I have analyzed the performance and generated the report.")
        return prediction.squeeze().numpy()

    def train_step(self, vision_data=None, vision_labels=None, text_in=None, text_target=None):
        self.model.train()
        self.optimizer.zero_grad()
        total_loss = 0
        if vision_data is not None:
            output = self.model(vision_input=vision_data)
            total_loss += self.proctor_loss_fn(output['proctor_logits'], vision_labels)
        if text_in is not None:
            output = self.model(text_input=text_in)
            total_loss += self.text_loss_fn(output['text_logits'].view(-1, output['text_logits'].size(-1)), text_target.view(-1))
        if total_loss != 0:
            total_loss.backward()
            self.optimizer.step()
            return total_loss.item()
        return 0.0
