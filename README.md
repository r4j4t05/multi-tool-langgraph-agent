# 🤖 Multi-Tool LangGraph Agent

A Python-based AI agent built with **LangGraph** and **Google Gemini** that dynamically selects tools to solve user queries through a structured, multi-step workflow.

## 📌 Overview

This project demonstrates how an LLM-powered agent can interpret user requests, select appropriate tools, process their results, and generate a final response.

Instead of relying on a single LLM response, the agent uses conditional routing and shared state to coordinate multiple tools.

## ✨ Features

- 🧮 **Calculator** — Performs mathematical calculations.
- 📏 **Unit Converter** — Converts supported measurement units.
- 🔤 **String Analyzer** — Analyzes text and counts characters.
- 🧠 **Intelligent Routing** — Selects tools based on the user's request.
- 🔄 **State Management** — Maintains information across workflow steps.
- 📝 **Response Generation** — Produces a final answer using a Writer component.

## 🏗️ Workflow Architecture

```text
       User Query
           |
           v
      Agent (LLM)
           |
     Conditional Router
           |
     +-----+-----+
     |     |     |
     v     v     v
 Calculator | String
     Unit   | Analyzer
  Converter |
     +-----+-----+
           |
           v
      Agent (LLM)
           |
           v
         Writer
           |
           v
      Final Answer
```

*The agent routes requests to the appropriate tool and uses the resulting state to determine the next step.*

## 🛠️ Tech Stack

- **Language:** Python
- **Workflow Orchestration:** LangGraph
- **LLM Integration:** Google Gemini
- **LLM Framework:** LangChain
- **Environment Configuration:** python-dotenv

## ⚙️ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/r4j4t05/multi-tool-langgraph-agent.git
cd multi-tool-langgraph-agent
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install langgraph langchain-google-genai python-dotenv
```

### 4. Configure your API key

Create a `.env` file in the project directory:

```env
GOOGLE_API_KEY=your_google_gemini_api_key
```

Replace the placeholder with your own API key. Never commit your `.env` file or expose your API key publicly.

### 5. Run the application

Run the Python application:

```bash
python app.py
```

If you prefer to use the Jupyter Notebook, open `main.ipynb` in VS Code and execute its cells in order.

## 📂 Project Structure

```text
multi-tool-langgraph-agent/
├── app.py
├── main.ipynb
├── .gitignore
└── README.md
```

## 🎯 Learning Objectives

- Building stateful workflows with LangGraph.
- Implementing conditional routing between tools.
- Integrating LLMs with Python functions.
- Managing multi-step tool execution.
- Handling intermediate results and final responses.

## 🔐 Security

- Keep API keys in environment variables.
- Exclude `.env` and virtual environments using `.gitignore`.
- Never publish credentials or other sensitive information.

## 👨‍💻 Author

**Rajat Kumar Singh**

GitHub: [@r4j4t05](https://github.com/r4j4t05)

---

⭐ If you find this project useful, consider starring the repository.
