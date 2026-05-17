import torch
import torch.nn as nn
import torch.nn.functional as F

class ExamInstructorAI(nn.Module):
    """
    Advanced Multi-Modal AI System designed for:
    1. Automated Proctoring (Cheating Detection via Vision)
    2. Exam Management & Scheduling (Logical Sequencing)
    3. Dynamic Question Generation (Transformer-based)
    4. Student Performance Analytics (Report Generation)
    """
    def __init__(self, vocab_size=5000, embed_size=256, max_seq_len=100):
        super(ExamInstructorAI, self).__init__()
        
        # --- 1. Vision Module (The "Eye") ---
        # Processes 224x224 RGB images/video frames
        self.vision_backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 14 * 14, embed_size) # Feature vector for activity detection
        )
        
        # --- 2. Language Module (The "Voice & Logic") ---
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_size)
        
        # Transformer for Questions and Analytics
        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=8,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        
        # --- 3. Output Heads ---
        # Proctoring Head: Binary classification (0: Normal, 1: Cheating/Unusual)
        self.proctor_head = nn.Linear(embed_size, 2)
        
        # Generation Head: Predicting the next token (Questions/Reports)
        self.generator_head = nn.Linear(embed_size, vocab_size)

    def forward(self, vision_input=None, text_input=None):
        """
        Dual-mode forward pass.
        If vision_input is provided, it acts as a Proctor.
        If text_input is provided, it acts as an Instructor/Assistant.
        """
        results = {}

        # Proctor Mode (Vision)
        if vision_input is not None:
            vision_features = self.vision_backbone(vision_input)
            results['proctor_logits'] = self.proctor_head(vision_features)

        # Instructor/Assistant Mode (Text/Logic)
        if text_input is not None:
            batch_size, seq_len = text_input.shape
            positions = torch.arange(0, seq_len).expand(batch_size, seq_len).to(text_input.device)
            
            x = self.token_embedding(text_input) + self.pos_embedding(positions)
            
            # Use Transformer to understand and generate
            transformer_out = self.transformer(x, x)
            results['text_logits'] = self.generator_head(transformer_out)

        return results
