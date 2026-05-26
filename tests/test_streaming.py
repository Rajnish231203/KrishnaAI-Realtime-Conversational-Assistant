"""
Quick Test Script for Streaming Components
Tests each component individually for latency
"""

import asyncio
import time
import sys
import os

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    os.system('chcp 65001 >nul 2>&1')
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


async def test_stt():
    """Test STT latency via new STTManager (ElevenLabs primary, FasterWhisper fallback)"""
    print("\n" + "="*60)
    print("🎯 TESTING STT via STTManager")
    print("="*60)

    from backend.app.services.stt import STTManager
    import wave
    import io

    # Create 1 second of silence at 16kHz (PCM16 raw bytes — no WAV header)
    sample_rate = 16000
    samples = bytes([0] * (sample_rate * 1 * 2))  # 1 s × 16-bit

    stt = STTManager()
    print(f"  Primary  : {stt.active_provider}")
    print(f"  Fallback : {stt.fallback_provider or 'none'}")

    print("\nWarming up providers...")
    try:
        await stt.warmup()
    except Exception as exc:
        print(f"⚠️  Warmup issue (fallback may handle): {exc}")

    print("📝 Testing partial transcription...")
    start = time.time()
    result = await stt.transcribe_partial(samples)
    elapsed = (time.time() - start) * 1000

    # New contract: returns str — empty string on silence (not None)
    print(f"✅ Partial result: '{result}' (empty=silence, as expected)")
    print(f"⚡ Latency: {elapsed:.0f}ms")

    if elapsed < 400:
        print("✅ EXCELLENT - Under 400ms target!")
    elif elapsed < 600:
        print("⚠️ ACCEPTABLE - Could be faster")
    else:
        print("❌ SLOW - Needs optimization")

    await stt.shutdown()

async def test_llm():
    """Test LLM first token latency"""
    print("\n" + "="*60)
    print("🧠 TESTING STREAMING LLM")
    print("="*60)
    
    from backend.app.services.llm.streaming_llm import StreamingLLM
    
    llm = StreamingLLM()
    
    messages = [
        {"role": "system", "content": "You are Krishna. Respond briefly."},
        {"role": "user", "content": "What is dharma?"}
    ]
    
    print("🧠 Generating response...")
    start = time.time()
    first_token_time = None
    token_count = 0
    
    async for token in llm.stream_response(messages):
        if first_token_time is None:
            first_token_time = time.time()
            latency = (first_token_time - start) * 1000
            print(f"⚡ First token latency: {latency:.0f}ms")
            
            if latency < 300:
                print("✅ EXCELLENT - Under 300ms target!")
            elif latency < 500:
                print("⚠️ ACCEPTABLE - Could be faster")
            else:
                print("❌ SLOW - Needs optimization")
        
        token_count += 1
        print(token, end='', flush=True)
    
    total_time = (time.time() - start) * 1000
    print(f"\n\n✅ Generated {token_count} tokens in {total_time:.0f}ms")

async def test_tts():
    """Test TTS first audio latency"""
    print("\n" + "="*60)
    print("🔊 TESTING STREAMING TTS")
    print("="*60)
    
    from backend.app.services.tts.streaming_tts import StreamingTTS
    
    tts = StreamingTTS()
    
    text = "Greetings, dear one. I am here to guide you."
    
    print(f"🔊 Generating audio for: {text}")
    start = time.time()
    first_chunk_time = None
    chunk_count = 0
    
    async for chunk in tts.stream_audio(text):
        if first_chunk_time is None:
            first_chunk_time = time.time()
            latency = (first_chunk_time - start) * 1000
            print(f"⚡ First audio chunk latency: {latency:.0f}ms")
            
            if latency < 500:
                print("✅ EXCELLENT - Under 500ms target!")
            elif latency < 800:
                print("⚠️ ACCEPTABLE - Could be faster")
            else:
                print("❌ SLOW - Needs optimization")
        
        chunk_count += 1
    
    total_time = (time.time() - start) * 1000
    print(f"✅ Generated {chunk_count} audio chunks in {total_time:.0f}ms")

async def test_full_pipeline():
    """Test complete pipeline end-to-end"""
    print("\n" + "="*60)
    print("🚀 TESTING FULL PIPELINE")
    print("="*60)
    
    from backend.app.services.stt import STTManager
    from backend.app.services.llm.streaming_llm import StreamingLLM
    from backend.app.services.tts.streaming_tts import StreamingTTS
    
    # Simulate user saying "What is my purpose?"
    user_text = "What is my purpose in life?"
    
    print(f"👤 User: {user_text}")
    
    pipeline_start = time.time()
    
    # LLM
    llm = StreamingLLM()
    messages = [
        {"role": "system", "content": "You are Krishna. Respond in 1-2 sentences."},
        {"role": "user", "content": user_text}
    ]
    
    print("\n🧠 Krishna: ", end='', flush=True)
    
    llm_start = time.time()
    first_token_time = None
    response_text = ""
    
    async for token in llm.stream_response(messages):
        if first_token_time is None:
            first_token_time = time.time()
            llm_latency = (first_token_time - llm_start) * 1000
        
        response_text += token
        print(token, end='', flush=True)
    
    print("\n")
    
    # TTS
    tts = StreamingTTS()
    tts_start = time.time()
    first_audio_time = None
    tts_latency = 0  # Initialize to prevent UnboundLocalError

    
    # Split into sentences for streaming
    sentences = response_text.split('. ')
    
    for sentence in sentences:
        if not sentence.strip():
            continue
            
        async for chunk in tts.stream_audio(sentence):
            if first_audio_time is None:
                first_audio_time = time.time()
                tts_latency = (first_audio_time - tts_start) * 1000
    
    total_latency = (time.time() - pipeline_start) * 1000
    
    print("\n" + "="*60)
    print("📊 PIPELINE METRICS")
    print("="*60)
    print(f"🧠 LLM First Token: {llm_latency:.0f}ms")
    print(f"🔊 TTS First Audio: {tts_latency:.0f}ms")
    print(f"⚡ TOTAL LATENCY: {total_latency:.0f}ms")
    print("="*60)
    
    if total_latency < 1000:
        print("✅ EXCELLENT - Under 1 second! 🎉")
    elif total_latency < 1500:
        print("⚠️ GOOD - Close to target")
    else:
        print("❌ NEEDS WORK - Over 1.5 seconds")

async def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🧪 KRISHNA STREAMING COMPONENTS TEST SUITE")
    print("="*60)
    
    # Check what to test
    if len(sys.argv) > 1:
        test_name = sys.argv[1].lower()
        
        if test_name == 'stt':
            await test_stt()
        elif test_name == 'llm':
            await test_llm()
        elif test_name == 'tts':
            await test_tts()
        elif test_name == 'pipeline':
            await test_full_pipeline()
        else:
            print(f"❌ Unknown test: {test_name}")
            print("Available tests: stt, llm, tts, pipeline")
    else:
        # Run all tests
        await test_stt()
        await test_llm()
        await test_tts()
        await test_full_pipeline()
    
    print("\n✅ Testing complete!\n")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Tests interrupted")
