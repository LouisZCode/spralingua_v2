import asyncio
import os
from dotenv import load_dotenv
import aiohttp
from loguru import logger

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.pipeline.runner import PipelineRunner
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.services.deepgram.stt import DeepgramSTTService
from deepgram import LiveOptions
from pipecat.services.minimax.tts import MiniMaxHttpTTSService
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frameworks.langchain import LangchainProcessor
from pipecat.transcriptions.language import Language

# Your LangChain agent
from agents import test_agent

# Session logging
from logs.session_logger import SessionLogger, create_pipecat_log_sink

load_dotenv()

async def main():
    async with aiohttp.ClientSession() as session:

        # Local mic/speaker
        transport = LocalAudioTransport(
            LocalAudioTransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(
                    params=VADParams(stop_secs=1.5)  # Wait 1.5s of silence before ending utterance
                ),
            )
        )

        # Speech-to-Text (Deepgram)
        stt = DeepgramSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            live_options=LiveOptions(
                language="en",         # Spanish input
                model="nova-2",        # Deepgram model
                smart_format=True,     # Better formatting
                utterance_end_ms=1000, # Wait 1s after last word for utterance boundary
            )
        )

        # LLM (LangChain agent instead of OpenAI directly)
        llm = LangchainProcessor(chain=test_agent)

        # Text-to-Speech (MiniMax with custom params)
        tts = MiniMaxHttpTTSService(
            api_key=os.getenv("MINIMAX_API_KEY"),
            group_id=os.getenv("MINIMAX_GROUP_ID"),
            aiohttp_session=session,
            model="speech-02-turbo",   # speech-02-turbo (fast), speech-02-hd (quality) - constructor param
            voice_id="german_bavarian_male_v2",  # Your cloned voice or system voice, german_bavarian_female, german_bavarian_male_v2 - constructor param
            params=MiniMaxHttpTTSService.InputParams(
                speed=1.0,                 # 0.5 to 2.0
                pitch=0,                   # -12 to 12
                volume=1.0,                # 0 to 10
                emotion="neutral",         # happy, sad, angry, fearful, disgusted, surprised, neutral, fluent
                language=Language.DE,      # Language enum (ES, EN, DE, FR, etc.)
            )
        )

        # Context + aggregators
        messages = [
            {"role": "system", "content": "Eres un asistente amigable. Responde de forma breve y conversacional en espanol."}
        ]
        context = LLMContext(messages)
        context_aggregator = LLMContextAggregatorPair(context)

        # Session logger - extract config dynamically from services
        session_logger = SessionLogger(log_dir="logs/conversations")
        session_logger.write_header({
            "deepgram": {
                "language": stt._settings.get("language"),
                "model": stt._settings.get("model"),
            },
            "minimax": {
                "model": tts._model_name,
                "voice_id": tts._voice_id,
                "language": tts._settings.get("language_boost"),
                "speed": tts._settings.get("voice_setting", {}).get("speed"),
            },
            "llm": {"agent": "test_agent", "model": test_agent.model},
        })
        logger.add(create_pipecat_log_sink(session_logger), filter="pipecat")

        # Pipeline
        pipeline = Pipeline([
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ])

        task = PipelineTask(pipeline)
        runner = PipelineRunner()

        print(f"Logging to: {session_logger._log_file}")
        print("Habla al microfono...")
        try:
            await runner.run(task)
        finally:
            session_logger.close()

if __name__ == "__main__":
    asyncio.run(main())
