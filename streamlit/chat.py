import streamlit as st
import asyncio
from ollama import ChatResponse
import ollama
import sys
import os


# Add parent directory to path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.tools.yahoo_finance_sync import scrape_yahoo_finance_news


# Set page configuration
st.set_page_config(page_title="Stock News Agent", layout="wide")
st.title("Stock News Assistant")

# Initialize session state for chat history and stock symbol
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "stock_symbol" not in st.session_state:
    st.session_state.stock_symbol = ""

def retrieve_stock_news(stock: str) -> list:
    """
    Summarize news articles for a given stock symbol.

    Args:
        stock (str): The stock ticker symbol (e.g., "NVDA").

    Returns:
        list: Summarized news articles with title,content,url and timestamp or an error message.
    """
    try:
        articles = scrape_yahoo_finance_news(stock)
        return articles  # Return the list directly

    except Exception as e:
        error_message = f"Error retrieving or summarizing news for {stock}: {str(e)}"
        print(error_message)
        return error_message

stock_symbol =""
# Define system prompt template
system_prompt_template ="""
You are a financial news analysis assistant. Your job is to analyze and summarize recent news articles for a specific stock symbol: **{stock_symbol}**. Only focus on this stock symbol-do not include information about any other stocks.

Instructions:
1. Retrieve and read the most recent news articles about {stock_symbol}.
2. For each article, provide the following details:
   - Article Title
   - Key content points (summarize the main facts and developments in detail)
   - Article URL
   - Publication Timestamp (date and time)
   - Sentiment Analysis (state clearly if the article is Bullish, Bearish, or Neutral for {stock_symbol}, and explain why)
3. Organize your response in this exact structure:
   - **Overview:** Write a brief summary (2–4 sentences) capturing the main themes and developments from all articles combined.
   - **Detailed Summaries:** For each article, use the following format:
     - Title:
     - Key Content Points:
     - URL:
     - Timestamp:
     - Sentiment Analysis:
4. State clearly the total number of articles retrieved and summarized at the top of your response.
5. Focus only on factual information and recent developments about {stock_symbol}.
6. Do NOT include news or information about any other stock symbol.
7. Present your answer in a clear, easy-to-read format with bullet points or lists where appropriate.

Important:
- Follow the output structure exactly as described above.
- Do not add extra commentary, opinions, or information about other stocks.
- Do not skip any of the required fields for each article.
- Do not summarize articles together-summarize each article separately in the Detailed Summaries section.

Articles:

"""
# Define available functions
available_functions = {
    'retrieve_stock_news': retrieve_stock_news,
}

# Function to extract stock symbol from user input
def extract_stock_symbol(text):
    import re
    # Common patterns for stock symbols (1-5 uppercase letters)
    patterns = [
        r'\b([A-Z]{1,5})\b',  # Standard stock symbol format
        r'\$([A-Z]{1,5})\b',  # Stock symbols with $ prefix
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[0]
    
    return None

async def call_function(tool_call):
    """Call the function specified in the tool call and return the output."""
    if function_to_call := available_functions.get(tool_call.function.name):
        with st.status(f"Retrieving news for {tool_call.function.arguments.get('stock', '')}...", expanded=True):
            st.write(f"Calling function: {tool_call.function.name}")
            output = function_to_call(**tool_call.function.arguments)
            st.write(f"Retrieved {len(output) if isinstance(output, list) else 0} articles")
        return output
    else:
        st.error(f"Function {tool_call.function.name} not found")
        return None

# Display chat messages from session state
for message in st.session_state.messages:
    # Only display user and final assistant messages, skip system, tool, tool_call, and intermediate messages
    if (message["role"] == "user" or 
        (message["role"] == "assistant" and not message.get("intermediate", False))):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# User input
with st.sidebar:
    st.subheader("About")
    st.write("Enter a stock symbol (e.g., AAPL, MSFT, NVDA) to get the latest news summary.")
    
    if st.button("Clear Chat History"):
        # Reset chat history and stock symbol
        st.session_state.messages = []
        st.session_state.stock_symbol = ""
        st.rerun()

# Process user input
async def process_user_input(user_input):
    if not user_input:
        return
    
    # Extract stock symbol from user input
    stock_symbol = extract_stock_symbol(user_input)
    
    # Update stock symbol in session state if found
    if stock_symbol:
        st.session_state.stock_symbol = stock_symbol
    
    # Use the current stock symbol or empty string if not set
    current_stock_symbol = st.session_state.stock_symbol or ""
    
    # Create system prompt with the stock symbol
    system_prompt = system_prompt_template.format(stock_symbol=current_stock_symbol)
    
    
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Prepare messages for API with system prompt
    api_messages = [{"role": "system", "content": system_prompt}] + st.session_state.messages
    
    # Show thinking indicator while waiting for response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        try:
            client = ollama.AsyncClient()
            model_name = 'qwen3:1.7b'
            
            response: ChatResponse = await client.chat(
                model_name,
                messages=api_messages,
                tools=[retrieve_stock_news],
                options={"num_ctx": 131072}
            )
            
            if response.message.tool_calls:
                for tool_call in response.message.tool_calls:
                    output = await call_function(tool_call)
                    if output is not None:
                        # Add tool call message and tool response to api_messages but mark them as intermediates
                        api_messages.append({
                            'role': 'tool_call', 
                            'content': response.message.content, 
                            'intermediate': True
                        })
                        api_messages.append({
                            'role': 'tool', 
                            'content': str(output), 
                            'name': tool_call.function.name
                        })
                        
                        final_response = await client.chat(model_name, messages=api_messages)
                        message_placeholder.markdown(final_response.message.content)
                        st.session_state.messages.append({
                            'role': 'assistant', 
                            'content': final_response.message.content
                        })
            else:
                message_placeholder.markdown(response.message.content)
                st.session_state.messages.append({
                    'role': 'assistant', 
                    'content': response.message.content
                })
        except Exception as e:
            message_placeholder.markdown(f"Error: {str(e)}")
            st.error(f"An error occurred: {str(e)}")

# Get user input
if prompt := st.chat_input("Ask about a stock (e.g., 'Tell me about AAPL news')"):
    # Run async function using event loop
    asyncio.run(process_user_input(prompt))
    st.rerun()  # Rerun to display the new messages