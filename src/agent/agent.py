from utils.utils import *
import gradio as gr
from gradio import ChatMessage
import aiohttp
import ssl
from google.adk.agents import Agent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.runners import Runner
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.auth.credential_service.in_memory_credential_service import InMemoryCredentialService
from google.genai import types
import httpx
import os
import uuid
from typing import List
from dotenv import load_dotenv
import uvicorn

# Load environment variables from .env file if it exists
load_dotenv()


class SupportTicketAgent():
    """Base agent class that handles support ticket assistance logic."""

    def __init__(self, rag_api_endpoint, llm_api_endpoint, model_api_key, model, context_window_length=10000, skip_tls=False):
        self.logger = Logger("support-ticket-agent", "INFO").new_logger()

        if not url_is_valid(rag_api_endpoint):
            raise InvalidAPIEndpointError("Invalid RAG API endpoint URL")

        self.rag_api_endpoint = rag_api_endpoint
        self.llm_api_endpoint = llm_api_endpoint
        self.model_api_key = model_api_key
        self.model = model
        self.context_window_length = context_window_length
        self.skip_tls = skip_tls
        self.ticket_state = {}
        self.num_sources = 1
        self.only_high_similarity_nodes = False

        # Configure API keys based on the endpoint being used
        if llm_api_endpoint and "ollama" in llm_api_endpoint.lower() or llm_api_endpoint and llm_api_endpoint != "https://generativelanguage.googleapis.com":
            # Using Ollama or other OpenAI-compatible endpoint
            os.environ["OPENAI_API_KEY"] = self.model_api_key
            os.environ["OPENAI_API_BASE"] = self.llm_api_endpoint
            self.logger.info(f"Configured agent to use OpenAI-compatible endpoint: {self.llm_api_endpoint} with model: {self.model}")
        else:
            # Using Google Gemini
            os.environ["GEMINI_API_KEY"] = self.model_api_key
            self.logger.info(f"Configured agent to use Gemini model: {self.model}")

        self.system_prompt = """You are a helpful support ticket assistant. Your goal is to collect specific pieces of information from the user to create a complete support case query.

On the first turn, you must use the `update_ticket_state` tool to store the original user message in the `user_description` field. Store the original user message as is, without any modifications.

After the first turn, on each turn, follow these steps:
1. **Observe**: Understand what information the user is providing about their support issue.
2. **Reason**: Identify what information is still missing. The required fields are: `product_version`, `user_description`, `error_message`.
3. **Act (Update State)**: When the user provides information, use the `update_ticket_state` tool to save it. For example:
   - If user says "I'm using version 2.1", use update_ticket_state with field="product_version" and value="2.1"
   - If user says "I get error 404", use update_ticket_state with field="error_message" and value="error 404"
4. **Act (Search)**: Use the `search_support_tickets` tool to check if we have all required information. The tool will automatically ask for missing information or perform the search when complete.

When updating product_version, the version MUST be a valid version string, for example 1.0, v1.0, etc. You MUST NOT ASUME ANY VERSION, ASK THE USER FOR IT IF IT IS NOT PROVIDED.

**CRITICAL: Use tools to answer the user's question. ALWAYS USE TOOLS TO ANSWER THE USER'S QUESTION.**

**CRITICAL: When the `search_support_tickets` tool returns search results (not asking for more information), you MUST respond with ONLY those results. Do NOT use any other tools after receiving search results. The search results are the final answer to the user's question.**

Remember to be conversational and helpful throughout the interaction."""

        # Create ADK agent with tools using simple Agent class
        self.agent = Agent(
            name="support_ticket_assistant",
            model=self.model,
            instruction=self.system_prompt,
            description="An assistant that helps users create support ticket queries by gathering required information.",
            tools=[self.search_support_tickets, self.update_ticket_state]
        )

        # Create services
        self.session_service = InMemorySessionService()

        # Create runner for invoking the agent
        self.runner = Runner(
            app_name="support_ticket_assistant",  # Match the agent name
            agent=self.agent,
            artifact_service=InMemoryArtifactService(),
            session_service=self.session_service,
            memory_service=InMemoryMemoryService(),
            credential_service=InMemoryCredentialService(),
        )

        self.logger.info(f"Support ticket agent created successfully")

    def update_ticket_state(self, field: str, value: str) -> str:
        """
        Update the ticket query state with information provided by the user.

        Args:
            field (str): The field to update (product_version, error_message, or user_description)
            value (str): The value provided by the user

        Returns:
            str: Confirmation message
        """
        if field not in ["product_version", "error_message", "user_description"]:
            return f"Invalid field: {field}. Valid fields are: product_version, error_message, user_description"

        self.ticket_state[field] = value
        self.logger.info(f"Updated {field}: {value}")
        self.logger.info(f"Current state: {self.ticket_state}")
        return f"Updated {field} to: {value}"

    async def search_support_tickets(self, query: str) -> str:
        """
        This function searches for support tickets based on the provided query.
        If not all required information is available in the TICKET_QUERY_STATE,
        it will ask the user for the missing information one piece at a time.

        Args:
            query (str): The user's query. This is required by the agent but not
                         directly used to fill the state. The LLM uses the conversation
                         history to populate the state.

        Returns:
            str: A message to the user, either asking for more information or
                 confirming that the search is being performed.
        """
        self.logger.info(f"TICKET INFO: {self.ticket_state}")

        # Check for missing information in our state
        if not self.ticket_state.get("product_version"):
            return "I can help with that. What is the product version you are using?"
        if not self.ticket_state.get("error_message"):
            return "Thanks. What is the exact error message you are seeing?"

        # If we have all the information, we can now "call" our API
        self.logger.info(f"All information gathered: {self.ticket_state}")
        self.logger.info("Querying the support ticket API...")

        query = f"Original User query: {self.ticket_state.get('user_description')}\n Error Message: {self.ticket_state.get('error_message')}\n Product Version: {self.ticket_state.get('product_version')}"
        response = await self.answer_query(query, self.num_sources, self.only_high_similarity_nodes)

        # Clear the state for the next interaction
        self.ticket_state.clear()

        return response

    async def answer_query(self, user_query, num_sources, only_high_similarity_nodes):
        """Query the RAG API endpoint."""
        # Create the query parameters
        query_params = {
            "user_query": user_query,
            "num_sources": num_sources,
            "only_high_similarity_nodes": only_high_similarity_nodes
        }

        # Make API call to the RAG API endpoint
        try:
            async with aiohttp.ClientSession() as session:
                # Use the RAG API endpoint
                api_url = self.rag_api_endpoint

                # Configure SSL context if skip_tls is True
                if self.skip_tls:
                    ssl_context = ssl.create_default_context()
                    ssl_context.check_hostname = False
                    ssl_context.verify_mode = ssl.CERT_NONE
                    connector = aiohttp.TCPConnector(ssl=ssl_context)
                    session = aiohttp.ClientSession(connector=connector)

                async with session.post(api_url, json=query_params) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("response", "No response received from API")
                    else:
                        error_text = await response.text()
                        self.logger.error(f"API call failed with status {response.status}: {error_text}")
                        return f"Error: API call failed with status {response.status}"

        except Exception as e:
            self.logger.error(f"Failed to call RAG API: {e}")
            return f"Error: Failed to call RAG API - {str(e)}"


class GradioAgent():
    """Gradio web UI wrapper for the support ticket agent."""

    def __init__(self, agent_port, rag_api_endpoint, llm_api_endpoint, model_api_key, model, context_window_length, skip_tls):
        self.logger = Logger("gradio-agent-ui", "INFO").new_logger()
        self.agent_port = agent_port

        # Create the underlying support ticket agent
        self.support_agent = SupportTicketAgent(
            rag_api_endpoint=rag_api_endpoint,
            llm_api_endpoint=llm_api_endpoint,
            model_api_key=model_api_key,
            model=model,
            context_window_length=context_window_length,
            skip_tls=skip_tls
        )

        # Create session tracking for conversations
        self.user_id = "gradio_user"
        self.session_id = str(uuid.uuid4())

    async def respond(self, message: str, history: List[List[str]]) -> tuple[List[List[str]], str]:
        """
        Handle user messages and generate responses using the agent.

        Args:
            message (str): The user's message
            history (List[List[str]]): The chat history in Gradio format

        Returns:
            tuple: Updated history and empty string (for new message)
        """
        try:
            # Get response using get_agent_response
            response_text = await self.get_agent_response(message)

            # Return the updated history (Gradio will handle the display)
            return history + [[message, response_text]], ""

        except Exception as e:
            error_message = f"An error occurred: {e}"
            self.logger.error(error_message)
            print(error_message)
            import traceback
            traceback.print_exc()
            return history + [[message, error_message]], ""

    async def get_agent_response(self, message: str) -> str:
        """
        Get response from agent.

        Args:
            message (str): The user's message

        Returns:
            str: The agent's response
        """
        try:
            # Ensure session exists
            app_name = "support_ticket_assistant"
            existing_session = await self.support_agent.session_service.get_session(
                app_name=app_name,
                user_id=self.user_id,
                session_id=self.session_id
            )

            if existing_session is None:
                # Session doesn't exist, create it
                await self.support_agent.session_service.create_session(
                    app_name=app_name,
                    user_id=self.user_id,
                    session_id=self.session_id
                )

            # Create message content
            new_message = types.Content(
                role="user",
                parts=[types.Part(text=message)]
            )

            # Get response from ADK agent using Runner
            response_text = ""
            async for event in self.support_agent.runner.run_async(
                user_id=self.user_id,
                session_id=self.session_id,
                new_message=new_message
            ):
                # Extract text from agent response events
                if hasattr(event, 'content') and event.content:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            response_text += part.text

            return response_text

        except Exception as e:
            error_message = f"An error occurred: {e}"
            self.logger.error(error_message)
            print(error_message)
            import traceback
            traceback.print_exc()
            return error_message

    def clear_chat(self):
        """Clear the chat history and reset the state."""
        self.support_agent.ticket_state.clear()
        # Create a new session for the new conversation
        self.session_id = str(uuid.uuid4())
        return [], "", "No information gathered yet"

    def run(self):
        
        reranker_top_n_results = 1

        
        with gr.Blocks(theme="soft", title="Support Cases Chatbot 💬", fill_height=True) as webui:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## Settings")
                    num_sources = gr.Slider(1, 5, value=1, step=1,
                                  label="Number of sources for reranking",
                                  info="Choose how many sources should be returned by the reranker")
                    only_high_similarity_nodes = gr.Checkbox(False,label="Use only cases with high similarity",info="Disregard cases that are not highly similar with the submitted query")

                with gr.Column(scale=4):
                    gr.Markdown("# Support Cases Chatbot 💬")
                    gr.Markdown("Ask me questions about previous support cases")

                    chatbot = gr.Chatbot(
                        value=[],
                        height=None,
                        type="messages",
                        min_height=400,
                        max_height=1000,
                        elem_id="chatbot",
                        container=True,
                        render_markdown=True,
                    )
                    textbox = gr.Textbox(
                        placeholder="Type your question here and press Enter",
                        container=False,
                        scale=7
                    )

            # Step 1: Add user message to chat box
            def add_user_and_placeholder(message, chat_history):
                chat_history = chat_history or []
                chat_history.append(ChatMessage(role="user", content=message))
                return "", chat_history

            # Step 2: Generate real response and replace placeholder
            async def generate_response(chat_history, num_sources, only_high_similarity_nodes):
                if not chat_history:
                    return chat_history
                last_user_message = chat_history[-1]['content']
                # Update agent settings from UI
                self.support_agent.num_sources = num_sources
                self.support_agent.only_high_similarity_nodes = only_high_similarity_nodes
                response = await self.get_agent_response(last_user_message)
                chat_history.append(ChatMessage(role="assistant", content=response))
                return chat_history

            # Once user submits the query run the required functions
            textbox.submit(
                add_user_and_placeholder,
                inputs=[textbox, chatbot],
                outputs=[textbox, chatbot]
            ).then(
                fn=generate_response,
                inputs=[chatbot, num_sources, only_high_similarity_nodes],
                outputs=[chatbot]
            )
        try:
            webui.launch(server_port=self.agent_port, server_name="0.0.0.0")
        except:
             raise FailedToRunChatBotWebUI("Agent WebUI failed to start")


def create_a2a_app(rag_api_endpoint, llm_api_endpoint, model_api_key, model, port=8001, skip_tls=False):
    """
    Create an A2A-exposed FastAPI application for the support ticket agent.

    Args:
        rag_api_endpoint: The RAG API endpoint URL
        llm_api_endpoint: The LLM API endpoint URL
        model_api_key: API key for the LLM
        model: The model name to use
        port: Port number for the A2A server
        skip_tls: Whether to skip TLS verification

    Returns:
        FastAPI application configured for A2A protocol
    """
    logger = Logger("a2a-app-factory", "INFO").new_logger()

    # Create the agent
    agent_server = SupportTicketAgent(
        rag_api_endpoint=rag_api_endpoint,
        llm_api_endpoint=llm_api_endpoint,
        model_api_key=model_api_key,
        model=model,
        skip_tls=skip_tls
    )

    # Let ADK auto-generate the agent card from the agent's metadata
    # The agent card will be automatically built from:
    # - Agent name and description
    # - Tools (which become capabilities)
    # - Agent instruction (becomes part of description)
    logger.info(f"Creating A2A app with auto-generated agent card for agent: {agent_server.agent.name}")

    # Expose the agent via A2A protocol
    # The agent card will be automatically generated
    a2a_app = to_a2a(agent_server.agent, port=port)

    logger.info(f"A2A app created successfully. Agent card available at: http://localhost:{port}/.well-known/agent-card.json")

    return a2a_app
