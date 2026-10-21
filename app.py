import ast
import operator
import os
import re
from typing import TypedDict, Optional

import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

load_dotenv()

st.set_page_config(page_title="Multi-Tool Agent", page_icon="🧮", layout="centered")

# -----------------------------
# LLM setup
# -----------------------------
@st.cache_resource
def get_llm():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"),
        temperature=0,
    )


# -----------------------------
# Shared graph state
# -----------------------------
class AgentState(TypedDict, total=False):
    question: str
    message: str
    tool_result: str
    tool_call_count: int
    selected_tool: str
    final_answer: str
    steps: list[str]
    converted_number: str
    error: str


# -----------------------------
# Safe calculator
# -----------------------------
_ALLOWED_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def safe_calculate(expression: str):
    """Evaluate arithmetic expressions without Python eval()."""
    tree = ast.parse(expression.strip(), mode="eval")

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINARY_OPERATORS:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ValueError("Exponent is too large.")
            return _ALLOWED_BINARY_OPERATORS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY_OPERATORS:
            return _ALLOWED_UNARY_OPERATORS[type(node.op)](visit(node.operand))
        raise ValueError("Use a valid arithmetic expression.")

    return visit(tree)


def format_number(value) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


# -----------------------------
# Tool selection
# -----------------------------
TOOL_NAMES = {"Calculator", "Converter", "String_Analyzer", "Writer"}


def invoke_text(llm, prompt: str) -> str:
    response = llm.invoke(prompt)
    content = response.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()
    return str(content).strip()


def choose_tool(state: AgentState, llm=None) -> tuple[str, str]:
    """Deterministic routing prevents the LLM from repeatedly selecting a tool."""
    question = state["question"].lower()
    original_question = state["question"]
    count = state.get("tool_call_count", 0)

    expression_pattern = (
        r"(?<![\w.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
        r"(?:\s*[+*/%\-]\s*-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)+"
    )
    has_arithmetic = bool(re.search(expression_pattern, original_question))
    asks_conversion = any(
        unit in question
        for unit in ("convert", "centimeter", "centimetre", "meter", "metre", " cm ")
    )
    asks_count = any(word in question for word in ("character", "digit", "count", "length"))

    if count == 0 and has_arithmetic:
        return "Calculator", "Calculate the arithmetic expression in the user's request."
    if count == 1 and asks_conversion:
        return "Converter", "Convert the calculator result to the requested unit."
    if count == 2 and asks_count:
        return "String_Analyzer", "Count digits in the converted numeric value."
    return "Writer", "Present the completed result."


# -----------------------------
# Graph nodes
# -----------------------------
def make_nodes(llm):
    def agent_node(state: AgentState):
        tool, instruction = choose_tool(state, llm)
        steps = list(state.get("steps", []))
        steps.append(f"Agent chose {tool}: {instruction}")
        return {"selected_tool": tool, "message": instruction, "steps": steps}

    def calculator_node(state: AgentState):
        # Extract the complete arithmetic expression, e.g. 100*564+987.
        question = state["question"]
        expression_pattern = (
            r"(?<![\w.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
            r"(?:\s*[+*/%\-]\s*-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)+"
        )
        match = re.search(expression_pattern, question)
        if not match:
            raise ValueError("I couldn't find a supported arithmetic expression in your request.")

        expression = match.group(0).replace(",", "")
        result = safe_calculate(expression)
        result_text = format_number(result)
        steps = list(state.get("steps", []))
        steps.append(f"Calculator: `{expression}` → `{result_text}`")
        return {
            "tool_result": result_text,
            "tool_call_count": state.get("tool_call_count", 0) + 1,
            "steps": steps,
        }

    def converter_node(state: AgentState):
        question = state["question"].lower()
        value_text = state.get("tool_result", "")
        try:
            value = float(value_text.replace(",", ""))
        except ValueError as exc:
            raise ValueError("The previous calculator result isn't numeric.") from exc

        # This project currently supports the common length conversions in the request.
        if re.search(r"\bmeters?\b|\bmetres?\b", question) and re.search(r"\bcentimeters?\b|\bcentimetres?\b|\bcm\b", question):
            converted = value * 100
            target_unit = "centimeters"
        elif re.search(r"\bcentimeters?\b|\bcentimetres?\b|\bcm\b", question) and re.search(r"\bmeters?\b|\bmetres?\b", question):
            # Only select this branch if the user explicitly says convert centimeters to meters.
            if re.search(r"centimeters?\s+to\s+meters?|centimetres?\s+to\s+metres?", question):
                converted = value / 100
                target_unit = "meters"
            else:
                converted = value * 100
                target_unit = "centimeters"
        else:
            raise ValueError("This app currently supports meter-to-centimeter conversion for this workflow.")

        converted_text = format_number(converted)
        steps = list(state.get("steps", []))
        steps.append(f"Converter: `{value_text} meters` → `{converted_text} {target_unit}`")
        return {
            "tool_result": converted_text,
            "converted_number": converted_text,
            "tool_call_count": state.get("tool_call_count", 0) + 1,
            "steps": steps,
        }

    def string_analyzer_node(state: AgentState):
        # Count only the digits in the converted numeric result; don't count the
        # unit, spaces, commas, or decimal formatting.
        numeric_text = state.get("converted_number") or state.get("tool_result", "")
        digits = "".join(re.findall(r"\d", numeric_text))
        count = len(digits)
        steps = list(state.get("steps", []))
        steps.append(f"String Analyzer: `{numeric_text}` → **{count} digits** (units excluded)")
        return {
            "tool_result": str(count),
            "tool_call_count": state.get("tool_call_count", 0) + 1,
            "steps": steps,
        }

    def writer_node(state: AgentState):
        count_result = state.get("tool_result", "")
        question = state["question"].lower()
        steps = list(state.get("steps", []))
        if state.get("selected_tool") == "Writer":
            steps.append("Writer: final response prepared")
        if any(word in question for word in ("character", "digit")) and count_result.isdigit():
            answer = f"{count_result} characters"
        else:
            answer = count_result
        return {"final_answer": answer, "steps": steps}

    return agent_node, calculator_node, converter_node, string_analyzer_node, writer_node


def build_graph(llm):
    agent_node, calculator_node, converter_node, analyzer_node, writer_node = make_nodes(llm)
    graph = StateGraph(AgentState)
    graph.add_node("Agent", agent_node)
    graph.add_node("Calculator", calculator_node)
    graph.add_node("Converter", converter_node)
    graph.add_node("String_Analyzer", analyzer_node)
    graph.add_node("Writer", writer_node)
    graph.add_edge(START, "Agent")

    def route(state: AgentState):
        tool = state.get("selected_tool", "Writer")
        if tool not in TOOL_NAMES:
            return "Writer"
        return tool

    graph.add_conditional_edges("Agent", route)
    graph.add_edge("Calculator", "Agent")
    graph.add_edge("Converter", "Agent")
    graph.add_edge("String_Analyzer", "Agent")
    graph.add_edge("Writer", END)
    return graph.compile()


def run_agent(question: str, llm) -> AgentState:
    app = build_graph(llm)
    initial_state: AgentState = {
        "question": question,
        "message": "",
        "tool_result": "",
        "tool_call_count": 0,
        "selected_tool": "",
        "final_answer": "",
        "steps": [],
        "converted_number": "",
    }
    return app.invoke(initial_state, config={"recursion_limit": 12})


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🧮 Multi-Tool Agent")
st.caption("Calculator · Unit converter · String analyzer — the agent picks the tools for you.")

with st.form("agent_form"):
    question = st.text_area(
        "What would you like to do?",
        value="Calculate 1248755557 * 875, then convert the result from meters to centimeters, and finally count the characters in the converted result.",
        height=90,
    )
    submitted = st.form_submit_button("Run", type="primary")

if submitted:
    # Remove stale output so a failed run cannot display an old answer.
    st.session_state.pop("last_result", None)
    st.session_state.pop("last_question", None)
    llm = get_llm()
    if llm is None:
        st.error("GOOGLE_API_KEY was not found. Add it to your .env file and restart the app.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Running the agent..."):
            try:
                result = run_agent(question.strip(), llm)
                st.session_state["last_result"] = result
                st.session_state["last_question"] = question.strip()
            except Exception as exc:
                st.error(f"Agent failed: {exc}")

if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    st.subheader("Answer")
    st.success(result.get("final_answer", "No final answer was produced."))

    with st.expander("Steps", expanded=True):
        for index, step in enumerate(result.get("steps", []), start=1):
            st.markdown(f"{index}. {step}")

st.divider()
st.caption("Keep your Google Gemini API key in `.env`. Never commit credentials to GitHub.")
