
import sys
import importlib.util
from pathlib import Path
import pandas as pd
from interfaces import BaseStrategy

def test_load(file_to_test):
    # Mocking the uploaded file object
    class MockUploadedFile:
        def __init__(self, path):
            self.path = path
        def getbuffer(self):
            with open(self.path, "rb") as f:
                return f.read()

    uploaded_file = MockUploadedFile(file_to_test)
    
    try:
        # Save temp file
        file_path = Path("user_strategies") / "temp_strategy_test.py"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Dynamic Import logic
        spec = importlib.util.spec_from_file_location("temp_module_test", str(file_path))
        module = importlib.util.module_from_spec(spec)
        sys.modules["temp_module_test"] = module
        spec.loader.exec_module(module)
        
        # Find the class that inherits from BaseStrategy
        found = False
        for name, obj in module.__dict__.items():
            if isinstance(obj, type) and issubclass(obj, BaseStrategy) and obj is not BaseStrategy:
                print(f"Successfully loaded: {name}")
                instance = obj()
                print(f"Instance name: {instance.get_name()}")
                found = True
        if not found:
            print("No valid strategy class found.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    strategies = [
        "user_strategies/breakout.py",
        "user_strategies/mean_reversion.py",
        "user_strategies/momentum.py",
        "user_strategies/ma_cross.py",
        "user_strategies/volatility_breakout.py",
        "user_strategies/trend_following.py"
    ]
    for s in strategies:
        print(f"\nTesting {s}...")
        test_load(s)
