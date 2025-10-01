#!/usr/bin/env python3
"""
Test script to validate the project structure and imports.
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_imports():
    """Test that all modules can be imported successfully."""
    try:
        # Test core modules
        from emotion_recognition import utils, models, metrics
        from src.data import data_affectnet, data_raf_db
        
        print("✅ All core modules imported successfully")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_config_loading():
    """Test configuration loading."""
    try:
        from emotion_recognition.utils import load_config, validate_config
        
        config = load_config('emotion_recognition/config.yaml')
        is_valid = validate_config(config)
        
        if is_valid:
            print("✅ Configuration loaded and validated successfully")
            return True
        else:
            print("❌ Configuration validation failed")
            return False
            
    except Exception as e:
        print(f"❌ Config test error: {e}")
        return False

def test_device_setup():
    """Test device setup."""
    try:
        from emotion_recognition.utils import setup_device
        
        device = setup_device()
        print(f"✅ Device setup successful: {device}")
        return True
        
    except Exception as e:
        print(f"❌ Device setup error: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing project structure...")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Config Test", test_config_loading),
        ("Device Test", test_device_setup),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        if test_func():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Project structure is valid.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
