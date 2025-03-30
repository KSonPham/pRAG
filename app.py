import chainlit as cl
from typing import Optional
import chainlit as cl
from passlib.context import CryptContext
import os

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
        if verify_password(password, os.getenv("PWD")):
            return cl.User(
                identifier="admin", 
                metadata={"role": "admin", "provider": "credentials"}
            )
    return None


@cl.on_chat_start
async def on_chat_start():
    app_user = cl.user_session.get("user")
    await cl.Message(f"Hello {app_user.identifier}, how can i help you today?").send()


@cl.on_message
async def on_message(message: cl.Message):
    counter = cl.user_session.get("counter")
    counter += 1
    cl.user_session.set("counter", counter)
    await cl.Message(content=f"You sent {counter} message(s)!").send()

if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)