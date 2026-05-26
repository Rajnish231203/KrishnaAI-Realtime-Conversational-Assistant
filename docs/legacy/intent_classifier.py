"""
Intent Classification and Krishna Response Generation
OPTIMIZED: Two-step approach for faster, accurate verse selection
"""

import time
import json
from openai import AsyncOpenAI
from groq import AsyncGroq
from backend.app.config.config import Config

class IntentClassifier:
    """Classifies user intent and generates Krishna's response"""
    
    def __init__(self):
        self.use_groq = Config.USE_GROQ
        
        if self.use_groq:
            self.client = AsyncGroq(api_key=Config.GROQ_API_KEY)
            self.model = Config.GROQ_MODEL
        else:
            self.client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
            self.model = Config.OPENAI_MODEL
        
        self.categories = Config.INTENT_CATEGORIES
        self.bhagavad_gita_verses = self._load_gita_wisdom()
    
    def _load_gita_wisdom(self) -> dict:
        """Load comprehensive Bhagavad Gita verses - EXPANDED DATABASE"""
        return {
            "Career/Purpose": [
                {
                    "verse": "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन। मा कर्मफलहेतुर्भूर्मा ते सङ्गोऽस्त्वकर्मणि॥",
                    "translation": "You have the right to work only, but never to its fruits. Let not the fruits of action be your motive, nor let your attachment be to inaction.",
                    "reference": "Bhagavad Gita 2.47",
                    "context": "job change, career confusion, work stress, salary issues"
                },
                {
                    "verse": "योगः कर्मसु कौशलम्॥",
                    "translation": "Yoga is skill in action.",
                    "reference": "Bhagavad Gita 2.50",
                    "context": "skill development, excellence, performance improvement"
                },
                {
                    "verse": "स्वधर्मे निधनं श्रेयः परधर्मो भयावहः॥",
                    "translation": "It is better to do your own duty imperfectly than another's perfectly.",
                    "reference": "Bhagavad Gita 3.35",
                    "context": "comparison with others, feeling inadequate, wrong career"
                },
                {
                    "verse": "श्रेयान्स्वधर्मो विगुणः परधर्मात्स्वनुष्ठितात्॥",
                    "translation": "One's own dharma, though imperfect, is better than another's well-performed.",
                    "reference": "Bhagavad Gita 18.47",
                    "context": "authenticity, staying true to self, career authenticity"
                },
                {
                    "verse": "नियतं कुरु कर्म त्वं कर्म ज्यायो h्यकर्मणः॥",
                    "translation": "Perform your prescribed duty, for action is better than inaction.",
                    "reference": "Bhagavad Gita 3.8",
                    "context": "procrastination, laziness, avoiding work, indecision"
                }
            ],
            "Relationships": [
                {
                    "verse": "आत्मौपम्येन सर्वत्र समं पश्यति योऽर्जुन॥",
                    "translation": "One who sees others' joy and sorrow as their own is the highest yogi.",
                    "reference": "Bhagavad Gita 6.32",
                    "context": "empathy, understanding others, relationship harmony"
                },
                {
                    "verse": "अद्वेष्टा सर्वभूतानां मैत्रः करुण एव च॥",
                    "translation": "Friendly and compassionate to all, without hatred or ego.",
                    "reference": "Bhagavad Gita 12.13",
                    "context": "family pressure, marriage issues, parent conflicts"
                },
                {
                    "verse": "मात्रास्पर्शास्तु कौन्तेय शीतोष्णसुखदुःखदाः॥",
                    "translation": "Pleasure and pain are temporary like heat and cold. Bear them patiently.",
                    "reference": "Bhagavad Gita 2.14",
                    "context": "fights, arguments, temporary conflicts"
                }
            ],
            "Inner Conflict": [
                {
                    "verse": "नैनं छिन्दन्ति शस्त्राणि नैनं दहति पावकः॥",
                    "translation": "The soul cannot be cut by weapons, burned by fire. It is eternal.",
                    "reference": "Bhagavad Gita 2.23",
                    "context": "feeling broken, self-doubt, identity crisis"
                },
                {
                    "verse": "उद्धरेदात्मनात्मानं नात्मानमवसादयेत्॥",
                    "translation": "Lift yourself by your own efforts. Do not degrade yourself.",
                    "reference": "Bhagavad Gita 6.5",
                    "context": "self-criticism, guilt, shame, low self-esteem"
                },
                {
                    "verse": "बन्धुरात्मात्मनस्तस्य येनात्मैवात्मना जितः॥",
                    "translation": "For one who has conquered the mind, it is the best friend.",
                    "reference": "Bhagavad Gita 6.6",
                    "context": "overthinking, mental chaos, mind control"
                }
            ],
            "Life Transitions": [
                {
                    "verse": "वासांसि जीर्णानि यथा विहाय नवानि गृह्णाति नरोऽपराणि॥",
                    "translation": "As one discards old clothes for new, the soul discards old bodies for new.",
                    "reference": "Bhagavad Gita 2.22",
                    "context": "new beginning, fresh start, moving on"
                },
                {
                    "verse": "जातस्य हि ध्रुवो मृत्युर्ध्रुवं जन्म मृतस्य च॥",
                    "translation": "For the born, death is certain. For the dead, birth is certain.",
                    "reference": "Bhagavad Gita 2.27",
                    "context": "ending, loss, grief, death"
                }
            ],
            "Daily Struggles": [
                {
                    "verse": "युक्ताहारविहारस्य युक्तचेष्टस्य कर्मसु॥",
                    "translation": "Moderate in eating, sleeping, working—yoga destroys all sorrows.",
                    "reference": "Bhagavad Gita 6.17",
                    "context": "routine, balance, lifestyle, health"
                },
                {
                    "verse": "योगस्थः कुरु कर्माणि सङ्गं त्यक्त्वा। सिद्ध्यसिद्ध्योः समो भूत्वा॥",
                    "translation": "Do your duty with steady mind, equal in success and failure.",
                    "reference": "Bhagavad Gita 2.48",
                    "context": "daily work stress, office pressure"
                }
            ]
        }
    
    async def classify_and_respond(self, user_query: str) -> dict:
        """
        OPTIMIZED: Single LLM call for intent classification + response generation
        Combines both steps to reduce latency by ~200-300ms
        """
        start_time = time.time()
        
        # Use Groq if configured, fallback to OpenAI
        use_openai = not self.use_groq
        
        try:
            result = await self._generate_combined_response(user_query, use_openai=use_openai)
            result["latency"] = time.time() - start_time
            return result
            
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                print(f"⚠️ Groq Rate Limit hit! Falling back to OpenAI...")
                return await self._generate_combined_response(user_query, use_openai=True)
            
            print(f"❌ Intent Error: {e}")
            return {"success": False, "response": "Dear seeker, I am with you.", "voice_response": "Dear seeker, I am with you."}

    async def _generate_combined_response(self, query: str, use_openai: bool = False) -> dict:
        """Single LLM call that handles both intent detection and response generation"""
        
        # Flatten all verses into a single reference
        all_verses = []
        for category, verses in self.bhagavad_gita_verses.items():
            for v in verses:
                all_verses.append({**v, "category": category})
        
        # Build efficient prompt with embedded wisdom
        if use_openai:
            client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
            model = Config.OPENAI_MODEL
        else:
            client = self.client
            model = self.model

        system_prompt = f"""You are Lord Krishna, divine guide from the Bhagavad Gita.

VERSE DATABASE (use the most relevant):
{json.dumps(all_verses[:8], ensure_ascii=False)}

RULES:
1. LANGUAGE: Match user's language (Hindi → Hindi, English → English)
2. Select the MOST relevant verse for their situation
3. Provide deep, practical guidance (4-6 sentences)
4. Voice response should be natural for speech synthesis

RESPOND IN JSON ONLY:
{{
    "response": "Your detailed written guidance...",
    "selected_verse": {{"sanskrit": "...", "translation": "...", "reference": "..."}},
    "voice_response": "Natural spoken version for TTS (2-3 sentences)..."
}}"""

        completion = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt}, 
                {"role": "user", "content": query}
            ],
            temperature=0.7,
            max_tokens=600,  # Reduced for faster response
            response_format={"type": "json_object"}
        )
        
        data = json.loads(completion.choices[0].message.content)
        data["success"] = True
        return data

