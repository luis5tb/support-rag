#!/usr/bin/env python3
"""
Simple test script to interact with the A2A agent.

This script demonstrates how to programmatically interact with the
A2A agent using the a2a-sdk client.

Usage:
    python test_a2a_agent.py
"""

import asyncio
import httpx
import uuid
from a2a.client.card_resolver import A2ACardResolver
from a2a.client.client_factory import ClientFactory, ClientConfig
from a2a.types import Message, TextPart


async def main():
    """Test the A2A agent by sending a message."""

    # A2A agent endpoint
    agent_url = "http://localhost:8001"

    print(f"Connecting to A2A agent at {agent_url}")
    print("-" * 60)

    # Create httpx client with longer timeout for RAG API calls
    timeout = httpx.Timeout(300.0, connect=10.0)  # 5 minute timeout, 10 second connect
    async with httpx.AsyncClient(timeout=timeout) as http_client:
        try:
            # Fetch the agent card to verify connection
            card_resolver = A2ACardResolver(http_client, agent_url)
            agent_card = await card_resolver.get_agent_card()

            print(f"Connected to agent: {agent_card.name}")
            print(f"Description: {agent_card.description}")

            # Display skills
            if agent_card.skills:
                skill_names = []
                for skill in agent_card.skills:
                    if hasattr(skill, 'name'):
                        skill_names.append(skill.name)
                    elif hasattr(skill, 'id'):
                        skill_names.append(skill.id)
                    else:
                        skill_names.append(str(skill))
                print(f"Skills: {', '.join(skill_names)}")
            else:
                print("Skills: None")

            print("-" * 60)

            # Create client from the agent card
            client_config = ClientConfig(httpx_client=http_client)
            factory = ClientFactory(config=client_config)
            client = factory.create(agent_card)

            # Interactive conversation loop
            print("\nInteractive chat started. Type 'quit', 'exit', or press Ctrl+C to end.\n")

            exit_chat = False
            while not exit_chat:
                try:
                    # Get user input
                    user_input = input("You: ").strip()

                    # Check for exit commands
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        print("\nExiting chat. Goodbye!")
                        exit_chat = True
                        continue

                    # Skip empty messages
                    if not user_input:
                        continue

                    print("Agent: ", end="", flush=True)

                    # Send message and get response
                    message = Message(
                        message_id=str(uuid.uuid4()),
                        role="user",
                        parts=[TextPart(text=user_input)]
                    )

                    response_received = False
                    async for event in client.send_message(message):
                        # Handle different event types - can be Task tuple or Message
                        if isinstance(event, tuple):
                            # (Task, Event) tuple
                            task, update = event

                            # Extract text from task artifacts
                            if task and hasattr(task, 'artifacts') and task.artifacts:
                                for artifact in task.artifacts:
                                    if hasattr(artifact, 'parts') and artifact.parts:
                                        for part in artifact.parts:
                                            # Part has a 'root' attribute that contains the actual content
                                            if hasattr(part, 'root'):
                                                root = part.root
                                                if hasattr(root, 'text') and root.text:
                                                    print(root.text, end="", flush=True)
                                                    response_received = True
                        elif isinstance(event, Message):
                            # Message response - extract from parts
                            if event.parts:
                                for part in event.parts:
                                    if hasattr(part, 'root'):
                                        root = part.root
                                        if hasattr(root, 'text') and root.text:
                                            print(root.text, end="", flush=True)
                                            response_received = True
                                    elif hasattr(part, 'text') and part.text:
                                        print(part.text, end="", flush=True)
                                        response_received = True

                    if not response_received:
                        print("[No response received]")

                    print()  # New line after response

                except KeyboardInterrupt:
                    print("\n\nExiting chat. Goodbye!")
                    exit_chat = True
                except EOFError:
                    print("\n\nExiting chat. Goodbye!")
                    exit_chat = True
                except Exception as inner_e:
                    # Handle errors during message sending
                    print(f"\nError: {inner_e}")
                    if "timeout" in str(inner_e).lower():
                        print("The RAG API is taking too long to respond. This might indicate:")
                        print("  - The RAG API is processing a complex query")
                        print("  - The RAG API server is overloaded or slow")
                        print("  - Network connectivity issues")
                    print()

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            print("\nMake sure the A2A agent is running on port 8001:")
            print("  uv run python src/cli.py a2a-agent -m gemini-2.5-flash --rag-api-endpoint http://127.0.0.1:8080/answer")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, asyncio.CancelledError):
        # Graceful exit - suppress cleanup errors
        pass
