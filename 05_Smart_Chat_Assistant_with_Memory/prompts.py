from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Always respond in English.

Conversation Summary (what was discussed before):
{summary}
"""),
    MessagesPlaceholder(variable_name="history"),  # last N recent messages
    ("human", "{question}"),
])