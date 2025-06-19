# verify_setup.py
"""
Verification script to ensure all components are working correctly
Run this after setup to verify your RAG system is ready
"""

import sys
import subprocess
import requests
import time

def check_python_packages():
    """Check if all required Python packages are installed"""
    print("🐍 Checking Python packages...")
    
    required_packages = [
        'chromadb', 'langchain', 'langchain_community', 
        'langchain_ollama', 'sentence_transformers', 
        'ollama', 'pytest', 'dotenv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All Python packages installed!")
    return True

def check_ollama_service():
    """Check if Ollama service is running"""
    print("\n🤖 Checking Ollama service...")
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("  ✅ Ollama service is running")
            return True
        else:
            print(f"  ❌ Ollama service returned status: {response.status_code}")
            return False
    except requests.exceptions.RequestException:
        print("  ❌ Ollama service is not running")
        print("  💡 Start with: ollama serve")
        return False

def check_ollama_models():
    """Check if required Ollama models are installed"""
    print("\n📦 Checking Ollama models...")
    
    required_models = ["llama3.2:3b", "nomic-embed-text"]
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            print("  ❌ Cannot check models - Ollama service issue")
            return False
        
        installed_models = [model["name"] for model in response.json()["models"]]
        
        missing_models = []
        for model in required_models:
            if any(model in installed for installed in installed_models):
                print(f"  ✅ {model}")
            else:
                print(f"  ❌ {model} - MISSING")
                missing_models.append(model)
        
        if missing_models:
            print(f"\n❌ Missing models: {', '.join(missing_models)}")
            print("Install with:")
            for model in missing_models:
                print(f"  ollama pull {model}")
            return False
        
        print("✅ All required models installed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Error checking models: {e}")
        return False

def test_ollama_embedding():
    """Test Ollama embedding functionality"""
    print("\n🔍 Testing Ollama embeddings...")
    
    try:
        from langchain_ollama import OllamaEmbeddings
        
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )
        
        # Test embedding
        test_text = "This is a test sentence for embedding."
        result = embeddings.embed_query(test_text)
        
        if isinstance(result, list) and len(result) > 0:
            print(f"  ✅ Embedding generated successfully (dimension: {len(result)})")
            return True
        else:
            print("  ❌ Invalid embedding result")
            return False
            
    except Exception as e:
        print(f"  ❌ Embedding test failed: {e}")
        return False

def test_ollama_llm():
    """Test Ollama LLM functionality"""
    print("\n💬 Testing Ollama LLM...")
    
    try:
        from langchain_ollama import OllamaLLM
        
        llm = OllamaLLM(
            model="llama3.2:3b",
            base_url="http://localhost:11434",
            temperature=0.1
        )
        
        # Test generation
        test_prompt = "What is 2+2? Answer with just the number."
        result = llm.invoke(test_prompt)
        
        if result and isinstance(result, str):
            print(f"  ✅ LLM responded: '{result.strip()}'")
            return True
        else:
            print("  ❌ Invalid LLM response")
            return False
            
    except Exception as e:
        print(f"  ❌ LLM test failed: {e}")
        return False

def test_rag_system():
    """Test basic RAG system functionality"""
    print("\n🎯 Testing RAG system...")
    
    try:
        from rag_system import RAGSystem
        
        # Initialize RAG system
        rag = RAGSystem()
        
        # Add test text
        test_text = """
        Python is a high-level programming language. 
        It was created by Guido van Rossum and first released in 1991.
        Python is known for its simple syntax and readability.
        """
        
        rag.add_text(test_text, "test_doc.txt")
        
        # Test query
        result = rag.query("Who created Python?")
        
        if result and result["answer"]:
            print(f"  ✅ RAG system working! Answer: {result['answer'][:100]}...")
            return True
        else:
            print("  ❌ RAG system test failed")
            return False
            
    except Exception as e:
        print(f"  ❌ RAG system test failed: {e}")
        return False

def print_system_info():
    """Print system information"""
    print("\n📊 System Information:")
    print(f"  Python version: {sys.version}")
    
    try:
        import torch
        print(f"  PyTorch available: {torch.__version__}")
    except ImportError:
        print("  PyTorch: Not installed")
    
    # Check available memory (approximate)
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"  Available RAM: {memory.available // (1024**3)} GB / {memory.total // (1024**3)} GB")
    except ImportError:
        print("  Memory info: psutil not available")

def main():
    """Run all verification checks"""
    print("🔧 RAG System Setup Verification")
    print("=" * 50)
    
    # Print system info
    print_system_info()
    
    # Run checks
    checks = [
        check_python_packages(),
        check_ollama_   service(),
        check_ollama_models(),
        test_ollama_embedding(),
        test_ollama_llm(),
        test_rag_system()
    ]
    
    print("\n" + "=" * 50)
    print("📋 VERIFICATION SUMMARY")
    print("=" * 50)
    
    passed = sum(checks)
    total = len(checks)
    
    if passed == total:
        print("🎉 ALL CHECKS PASSED!")
        print("✅ Your RAG system is ready to use!")
        print("\n🚀 Next steps:")
        print("  1. Run: python quick_start.py")
        print("  2. Or run: python text_loader.py")
        print("  3. Start asking questions about your documents!")
    else:
        print(f"⚠️  {passed}/{total} checks passed")
        print("❌ Please fix the issues above before proceeding")
        print("\n💡 Common solutions:")
        print("  - Install missing packages: pip install [package_name]")
        print("  - Start Ollama: ollama serve")
        print("  - Pull models: ollama pull llama3.2:3b")
        print("  - Pull models: ollama pull nomic-embed-text")

if __name__ == "__main__":
    main()