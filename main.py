import chainlit as cl
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv, find_dotenv
import os
from openai.types.responses import ResponseTextDeltaEvent
load_dotenv(find_dotenv())

gemini_api_key = os.getenv("GEMINI_API_KEY")

provider =  AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash",
    openai_client=provider,
)

run_config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True,
)

agent1 = Agent(
    instructions = """
            You are Nihal Khan Ghauri's personal AI assistant. Nihal is a Full-Stack Developer. Your role is to be helpful, friendly, 
            and knowledgeable. You assist with a wide range of development-related tasks, provide accurate and reliable information, 
            and help with technical and non-technical queries. If you encounter a situation where you lack the knowledge, you should 
            admit it rather than making up information.
            """,

    name="Nihal Khan Ghauri's Assistant",
)


result = Runner.run_sync(
    agent1,
    input="What is the capital of the moon?",
    run_config=run_config,
)

print(result.final_output)

@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(content = "Hello, I'm Nihal Khan Ghauri's Assistant. How can I help you today?").send()


@cl.on_message
async def main(message: cl.message):
    history = cl.user_session.get("history")

    msg = cl.Message(content = "")
    await msg.send()

    history.append({"role": "user", "content": message.content})
    result = Runner.run_streamed(
        agent1,
        input=history,
        run_config=run_config,
    )

    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            # print(event.data.delta, end="", flush=True)
            # msg.update(event.data.delta)
            await msg.stream_token(event.data.delta)


    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("history", history)
    

    



