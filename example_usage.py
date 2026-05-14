import numpy as np
from model import SkeletonModel
from brain import AIBrain

def main():
    # 1. Initialize the AI (Zero Knowledge)
    # Let's say it takes 2 inputs and produces 1 output
    input_size = 2
    output_size = 1
    
    model = SkeletonModel(input_size=input_size, output_size=output_size)
    brain = AIBrain(model, lr=0.01)
    
    print("--- AI Initialized (Knowledge: None) ---")
    
    # 2. Test initial prediction (Should be random)
    test_input = np.array([0.5, 0.8])
    initial_prediction = brain.think(test_input)
    print(f"Initial Prediction for [0.5, 0.8]: {initial_prediction}")
    
    # 3. Simulated Self-Learning Loop
    # Let's teach it that the output should be the sum of the inputs (a + b)
    print("\n--- Starting Self-Learning Process ---")
    for epoch in range(100):
        # Generate random training data
        a = np.random.rand()
        b = np.random.rand()
        x = np.array([a, b])
        y = np.array([a + b]) # Target: Sum
        
        loss = brain.learn(x, y)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.6f}")
            
    # 4. Test after learning
    final_prediction = brain.think(test_input)
    expected = test_input[0] + test_input[1]
    print(f"\n--- Learning Complete ---")
    print(f"Final Prediction for [0.5, 0.8]: {final_prediction} (Expected: {expected})")
    
    # 5. Save the 'Knowledge'
    brain.save_knowledge("learned_sum.pth")
    print("Knowledge saved to learned_sum.pth")

if __name__ == "__main__":
    main()
