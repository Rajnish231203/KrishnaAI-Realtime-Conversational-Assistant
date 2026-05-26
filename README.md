# 🕉️ Krishna Real-Time Voice Assistant

**Sub-Second Latency Voice AI** ⚡ (500-800ms typical response time)

---

## 🚀 Quick Start (3 Steps)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup API Keys
Copy `.env.template` to `.env` and add your API keys:
```bash
copy .env.template .env
```

Edit `.env` and add:
```env
OPENAI_API_KEY=sk-your-key-here
GROQ_API_KEY=gsk-your-key-here  # Optional but recommended
```

### 3. Run
```bash
python launch.py
```

Then open: **http://localhost:8000/client.html**

---

## 🎯 What You Get

✅ **<1 second latency** (20-30x faster than traditional systems)  
✅ **Parallel streaming** - STT, LLM, TTS run simultaneously  
✅ **Barge-in support** - Interrupt AI while speaking  
✅ **Real-time metrics** - See latency breakdown  
✅ **Beautiful web UI** - Modern interface with visualizer  

---

## 📊 Performance

| Component | Target | Typical |
|-----------|--------|---------|
| STT (Partial) | <400ms | 200-300ms |
| LLM (First Token) | <300ms | 150-250ms |
| TTS (First Audio) | <500ms | 300-400ms |
| **TOTAL** | **<1000ms** | **500-800ms** ✅ |

---

## 🎤 How to Use

1. Click **"Hold to Talk"** or press **SPACEBAR**
2. Speak your question
3. Release when done
4. Krishna responds in <1 second!

---

## 🔧 API Keys

### Required
- **OpenAI**: https://platform.openai.com/api-keys

### Recommended (FREE & Faster)
- **Groq**: https://console.groq.com/keys (3-5x faster LLM)

### Optional (Better Voice)
- **ElevenLabs**: https://elevenlabs.io/api (Premium TTS)

---

## 🧪 Testing

Test individual components:
```bash
python test_streaming.py stt    # Test STT
python test_streaming.py llm    # Test LLM
python test_streaming.py tts    # Test TTS
python test_streaming.py        # Test all
```

---

## 📁 Files

- `streaming_server.py` - WebSocket server + orchestrator
- `streaming_stt.py` - Streaming Speech-to-Text
- `streaming_llm.py` - Streaming LLM (Krishna)
- `streaming_tts.py` - Streaming Text-to-Speech
- `client.html` - Web interface
- `streaming_client.js` - Client logic
- `launch.py` - Start everything
- `test_streaming.py` - Performance tests

---

## 🏗️ Architecture

```
Mic (20-40ms chunks) → WebSocket → STT → LLM → TTS → Speaker
                                    ↓     ↓     ↓
                              Partial  Tokens Audio
                              (200ms) (300ms) (500ms)
```

**Parallel Streaming = 20-30x Faster!**

---

## 🐛 Troubleshooting

### High Latency?
1. Use Groq: `USE_GROQ=True` in `.env`
2. Check internet speed
3. Run: `python test_streaming.py`

### Connection Failed?
1. Check server is running
2. Verify port 8765 is available
3. Check firewall settings

### No Audio?
1. Grant microphone permissions
2. Check browser console
3. Verify TTS API key

---

## 💡 Tips

**For Lowest Latency:**
- Use Groq for LLM
- Use ElevenLabs for TTS
- Good internet connection

**For Best Quality:**
- Use ElevenLabs TTS
- Adjust voice settings in `config.py`
- Use good microphone

---

## 🎉 Made with ❤️

Built with OpenAI, Groq, ElevenLabs, WebSockets, and Web Audio API

**"Through practice and detachment, the mind can be controlled."** - Bhagavad Gita
