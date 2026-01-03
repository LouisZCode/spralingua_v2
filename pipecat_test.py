import asyncio
import os
from dotenv import load_dotenv
import aiohttp

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.pipeline.runner import PipelineRunner
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.minimax.tts import MiniMaxHttpTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair

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

        # Speech-to-Text
        stt = DeepgramSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY")
        )

        # LLM
        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini"
        )

        # Text-to-Speech
        tts = MiniMaxHttpTTSService(
            api_key=os.getenv("MINIMAX_API_KEY"),
            group_id=os.getenv("MINIMAX_GROUP_ID"),
            aiohttp_session=session,
        )

        # Context + aggregators
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Keep responses brief and conversational."}
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

        print("Speak into your microphone...")
        await runner.run(task)

if __name__ == "__main__":
    asyncio.run(main())
