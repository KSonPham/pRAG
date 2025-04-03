import chainlit as cl
import os
from passlib.context import CryptContext
from config import Config
from db import VectorDatabase
from agent import Agent
import datetime
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableConfig, RunnableLambda, RunnableMap
from langchain.schema import StrOutputParser
from langchain.callbacks.base import BaseCallbackHandler
from agent import ChatHistory
import ray
import asyncio
from typing import List
# if not ray.is_initialized():
    
# Add this helper function
async def track_processing(futures: List[ray.ObjectRef], files: List[cl.File]):
    """Track processing status of multiple files with streaming updates."""
    tasks = []
    
    # Create a step for each file
    for future, file in zip(futures, files):
        task = asyncio.create_task(
            track_single_file(future, file))
        tasks.append(task)
    
    # Wait for all tracking to complete
    await asyncio.gather(*tasks)

async def track_single_file(future: ray.ObjectRef, file: cl.File):
    """Track processing status of a single file with streaming updates."""
    async with cl.Step(name=f"Processing {file.name}", type="processing") as step:
        while True:
            try:
                # Check if processing is done with timeout
                done, _ = ray.wait([future], timeout=0.1)
                if done:
                    result = ray.get(future)
                    step.output = f"✅ Finished processing {file.name}"
                    await step.update()
                    break
                else:
                    step.output = f"⏳ Processing {file.name}..."
                    await step.update()
            except Exception as e:
                step.output = f"❌ Error processing {file.name}: {str(e)}"
                await step.update()
                break
            await asyncio.sleep(1)
            
# Create a password context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str):
    """Hash a password for storing."""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str):
    """Verify a password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)

# Pre-hash the admin password (do this once and store the hash)
# hashed_admin_password = hash_password("admin")

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # In real applications, fetch the user and hashed password from your database
    if username == "admin":
        if verify_password(password, Config.PWD):
            return cl.User(
                identifier="admin", 
                metadata={"role": "admin", "provider": "credentials"}
            )
    return None


@cl.on_chat_start
async def on_chat_start():
    app_user = cl.user_session.get("user")
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content=f"Hello {app_user.identifier}, please upload pdf files to begin!",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
            max_files=5,
        ).send()
    collection_name = f"{app_user.identifier}.collection"
    if not ray.is_initialized():
        ray.init()

    vector_db = VectorDatabase.remote(qdrant_url=os.getenv('QDRANT_URL'), embed_model_id=Config.EMBED_MODEL_ID)
    agent = Agent(llm="gemma3:12b")
    # Init chat history
    cl.user_session.set("chat_history", ChatHistory(limit=10))
  
    # Create personal data collection
    vector_db.create_collection.remote(collection_name=collection_name)
        # Show initial processing message
    initial_msg = await cl.Message(
        content=f"Starting processing for {len(files)} files..."
    ).send()

    futures = [vector_db.add_documents3.remote(file.path, collection_name) for file in files]
    # Wait for all tasks to complete
        # Track processing with streaming updates
    await track_processing(futures, files)
    
    # Update initial message when done
    initial_msg.content = f"✅ Finished processing {len(files)} files!"
    await initial_msg.update()
    # await initial_msg.update(content=f"✅ Completed processing {len(files)} files!")
    # await cl.Message(content=f"Finished processing {len(files)} files...").send()
    runnable = (
        RunnableMap({
            "original_input": RunnablePassthrough(),
            "vector_results": vector_db.get_context.remote
        })
        | RunnableLambda(lambda x: agent(
            x["original_input"], 
            vector_results=x["vector_results"],
            chat_history=cl.user_session.get("chat_history")
        ))
        | StrOutputParser()
    )
    cl.user_session.set("runnable", runnable)
    
    # await cl.Message(f"Hello {app_user.identifier}, how can i help you today?").send()
    # elements = [
    #   cl.Pdf(name="pdf1", display="side", path="./data/2311-SMERF.04079v1.pdf", page=1) # page, side, inline
    # ]
    # # Reminder: The name of the pdf must be in the content of the message
    # await cl.Message(content="Look at this local pdf1!", elements=elements).send()

# @cl.on_chat_end
# def end():
#     # print("goodbye", cl.user_session.get("id"))
#     client = cl.user_session.get("vector_db").client
#     collection_name = cl.user_session.get("collection_name")
#     client.delete_collection(collection_name=collection_name)

@cl.on_message
async def on_message(message: cl.Message):
    msg_timestamp = datetime.datetime.now().isoformat()
    history = cl.user_session.get("chat_history")
    # prompt = message.content
    runnable = cl.user_session.get("runnable")
    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        message.content,
        config=RunnableConfig(callbacks=[
            cl.LangchainCallbackHandler(),
        ]),
    ):
        await msg.stream_token(chunk)

    await msg.send()

    history.add_message({
        "timestamp": msg_timestamp,
        "user": message.content,
        "assistant": msg.content,
    })    
    cl.user_session.set("chat_history", history)


if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)