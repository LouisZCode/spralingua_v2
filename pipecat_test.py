import asyncio
import os
from dotenv import load_dotenv
import aiohttp

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.pipeline.runner import PipelineRunner
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.services.deepgram.stt import DeepgramSTTService
from deepgram import LiveOptions
from pipecat.services.minimax.tts import MiniMaxHttpTTSService
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frameworks.langchain import LangchainProcessor
from pipecat.transcriptions.language import Language

# Your LangChain agent
from agents import test_agent

load_dotenv()

async def main():
    async with aiohttp.ClientSession() as session:

        # Local mic/speaker
        transport = LocalAudioTransport(
            LocalAudioTransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
            )
        )

        # Speech-to-Text (Deepgram)
        stt = DeepgramSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            live_options=LiveOptions(
                language="es",         # Spanish input
                model="nova-2",        # Deepgram model
                smart_format=True,     # Better formatting
            )
        )

        # LLM (LangChain agent instead of OpenAI directly)
        llm = LangchainProcessor(chain=test_agent)

        # Text-to-Speech (MiniMax with custom params)
        tts = MiniMaxHttpTTSService(
            api_key=os.getenv("MINIMAX_API_KEY"),
            group_id=os.getenv("MINIMAX_GROUP_ID"),
            aiohttp_session=session,
            params=MiniMaxHttpTTSService.InputParams(
                model="speech-02-turbo",   # speech-02-turbo (fast), speech-02-hd (quality)
                voice_id="SadSally_German_v2",  # Your cloned voice or system voice
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

        print("Habla al microfono...")
        await runner.run(task)

if __name__ == "__main__":
    asyncio.run(main())
