"""Multi-tool LangGraph agent (Calculator, Converter, String Analyzer) as a Streamlit app.

Run:  streamlit run app.py
Needs GOOGLE_API_KEY in the environment or in a .env file next to this file.
"""
import ast
import operator
import os
import re
from pathlib import Path
from typing import Annotated, TypedDict
from operator import add

import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

load_dotenv(Path(__file__).parent / ".env")

MODEL = os.getenv("AGENT_MODEL", "gemini-3.1-flash-lite")
MAX_TOOL_CALLS = 5
TOOLS = ("Calculator", "Converter", "String_Analyzer", "Writer")


# --------------------------------------------------------------------------- #
# LLM helpers
# --------------------------------------------------------------------------- #
@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(model=MODEL)


def ask_llm(prompt: str) -> str:
    """Invoke the LLM and always return plain text (content may be str or blocks)."""
    content = get_llm().invoke(prompt).content
    if isinstance(content, str):
        return content.strip()
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts).strip()


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(node):
    """Evaluate arithmetic only (no names, calls or attributes) - replaces eval()."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("Exponent too large")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported expression")


def fmt(x: float) -> str:
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return f"{x:.10g}" if isinstance(x, float) else str(x)


# Linear units: factor = how many base units in one of this unit.
_LINEAR = {
    "length": {"meters": 1, "kilometers": 1000, "centimeters": 0.01, "millimeters": 0.001},
    "mass": {"grams": 1, "kilograms": 1000, "milligrams": 0.001, "tonnes": 1_000_000},
    "time": {"seconds": 1, "minutes": 60, "hours": 3600, "days": 86400, "weeks": 604800},
    "data": {"bytes": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4},
    "speed": {"m/s": 1, "km/h": 1 / 3.6},
}
_TEMPS = ("Celsius", "Fahrenheit", "Kelvin")

_ALIASES = {
    "meter": "meters", "m": "meters", "metre": "meters", "metres": "meters",
    "kilometer": "kilometers", "km": "kilometers", "kilometre": "kilometers", "kilometres": "kilometers",
    "centimeter": "centimeters", "cm": "centimeters", "centimetre": "centimeters", "centimetres": "centimeters",
    "millimeter": "millimeters", "mm": "millimeters", "millimetre": "millimeters", "millimetres": "millimeters",
    "gram": "grams", "g": "grams", "kilogram": "kilograms", "kg": "kilograms",
    "milligram": "milligrams", "mg": "milligrams", "tonne": "tonnes", "ton": "tonnes", "tons": "tonnes",
    "second": "seconds", "sec": "seconds", "s": "seconds", "minute": "minutes", "min": "minutes",
    "hour": "hours", "hr": "hours", "h": "hours", "day": "days", "week": "weeks",
    "byte": "bytes", "b": "bytes", "kb": "KB", "mb": "MB", "gb": "GB", "tb": "TB",
    "kmh": "km/h", "kph": "km/h", "kmph": "km/h", "mps": "m/s",
    "c": "Celsius", "celsius": "Celsius", "f": "Fahrenheit", "fahrenheit": "Fahrenheit",
    "k": "Kelvin", "kelvin": "Kelvin",
}


def _norm_unit(u: str) -> str:
    u = u.strip().strip(".").strip()
    key = u.lower().replace("°", "").replace(" ", "")
    if key in _ALIASES:
        return _ALIASES[key]
    for table in _LINEAR.values():
        for name in table:
            if name.lower() == key:
                return name
    return u


def convert_units(value: float, src: str, dst: str):
    """Return the converted value, or None if the pair is unsupported."""
    src, dst = _norm_unit(src), _norm_unit(dst)
    if src in _TEMPS and dst in _TEMPS:
        c = {"Celsius": value, "Fahrenheit": (value - 32) * 5 / 9, "Kelvin": value - 273.15}[src]
        return {"Celsius": c, "Fahrenheit": c * 9 / 5 + 32, "Kelvin": c + 273.15}[dst], dst
    for table in _LINEAR.values():
        if src in table and dst in table:
            return value * table[src] / table[dst], dst
    return None, dst


def calculator(state):
    expr = state["message"]
    try:
        result = fmt(_safe_eval(ast.parse(expr, mode="eval")))
    except Exception:
        result = "Please give a valid arithmetic expression."
    return {
        "tool_result": result,
        "tool_call_count": state["tool_call_count"] + 1,
        "trace": [f"Calculator: `{expr}` -> {result}"],
    }


def converter(state):
    prompt = f"""Instruction: {state['message']}
Previous tool result: {state['tool_result']}

Extract the unit conversion to perform. Valid units: meters, kilometers, centimeters,
millimeters, grams, kilograms, milligrams, tonnes, Celsius, Fahrenheit, Kelvin, seconds,
minutes, hours, days, weeks, bytes, KB, MB, GB, TB, km/h, m/s.
If the previous tool result is the value to convert, use its number.

Return exactly THREE lines:
Line 1: numeric value
Line 2: source unit
Line 3: target unit
Do not convert. Do not add explanations."""
    try:
        lines = [l.strip() for l in ask_llm(prompt).splitlines() if l.strip()]
        value = float(re.search(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", lines[0]).group())
        result, dst = convert_units(value, lines[1], lines[2])
        out = f"{fmt(result)} {dst}" if result is not None else f"Conversion from {lines[1]} to {lines[2]} is not supported."
    except Exception:
        out = "Could not understand the requested conversion."
    return {
        "tool_result": out,
        "tool_call_count": state["tool_call_count"] + 1,
        "trace": [f"Converter -> {out}"],
    }


def string_analyzer(state):
    text = state["message"]
    prompt = f"""Text to analyze:

{text}

Extract the text that should be analyzed. If it is already the text, return it exactly.
Return ONLY the extracted text. No quotes, no explanations, do not count characters."""
    try:
        extracted = ask_llm(prompt) or text
    except Exception:
        extracted = text
    out = str(len(extracted.strip()))
    return {
        "tool_result": out,
        "tool_call_count": state["tool_call_count"] + 1,
        "trace": [f"String_Analyzer: `{extracted.strip()}` -> {out} characters"],
    }


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #
class State(TypedDict):
    question: str
    message: str
    tool_result: str
    tool_call_count: int
    selected_tool: str
    final_answer: str
    trace: Annotated[list, add]


def agent(state):
    prompt = f"""You are an agent.

Question: {state['question']}
Previous tool result: {state['tool_result']}
Previous selected tool: {state['selected_tool']}

Choose exactly ONE tool: Calculator, Converter, String_Analyzer, Writer

Rules:
- If the previous tool was String_Analyzer, choose Writer.
- If the previous tool result already completes the user's request, choose Writer.
- Do not repeat a tool that already completed its operation.
- If a tool said an operation is not supported, choose Writer.
- When choosing String_Analyzer to analyze the previous tool result, pass the COMPLETE previous
  tool result (with units/words) as the instruction.

Your response MUST be exactly TWO lines:
Line 1: only the tool name.
Line 2: only the instruction for that tool.
Example:
Calculator
25*4"""
    lines = [l.strip() for l in ask_llm(prompt).splitlines() if l.strip()]
    tool = lines[0].strip("`*: ") if lines else "Writer"
    tool = next((t for t in TOOLS if t.lower() == tool.lower()), "Writer")
    message = lines[1] if len(lines) > 1 else ""
    note = f"Agent chose {tool}" + (f": `{message}`" if tool != "Writer" else "")
    return {"selected_tool": tool, "message": message, "trace": [note]}


def route(state):
    if state["tool_call_count"] >= MAX_TOOL_CALLS:
        return "Writer"
    return state["selected_tool"] if state["selected_tool"] in TOOLS else "Writer"


def writer(state):
    prompt = f"""You are the final answer writer.

Original user question:
{state['question']}

Tool result:
{state['tool_result']}

Give the final answer using ONLY the tool result.
- Do not calculate or convert anything yourself.
- Do not change the tool result or invent units/values.
- If the result is a character count, report it as a character count.
- If it says an operation is not supported or invalid, clearly say that.
Be brief."""
    return {"final_answer": ask_llm(prompt)}


@st.cache_resource
def build_app():
    g = StateGraph(State)
    g.add_node("Agent", agent)
    g.add_node("Calculator", calculator)
    g.add_node("Converter", converter)
    g.add_node("String_Analyzer", string_analyzer)
    g.add_node("Writer", writer)
    g.add_edge(START, "Agent")
    g.add_conditional_edges("Agent", route, list(TOOLS))
    for tool in ("Calculator", "Converter", "String_Analyzer"):
        g.add_edge(tool, "Agent")
    g.add_edge("Writer", END)
    return g.compile()


def run_agent(question: str) -> State:
    return build_app().invoke(
        {
            "question": question,
            "message": "",
            "tool_result": "",
            "tool_call_count": 0,
            "selected_tool": "",
            "final_answer": "",
            "trace": [],
        }
    )


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Multi-Tool Agent", page_icon="🧮")
st.title("🧮 Multi-Tool Agent")
st.caption("Calculator · Unit converter · String analyzer — the agent picks the tools for you.")

if not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY is not set. Add it to the .env file and restart.")
    st.stop()

with st.form("ask"):
    question = st.text_area(
        "What would you like to do?",
        placeholder="Calculate 25 * 4, convert the result from meters to kilometers, then count the characters in it.",
        height=100,
    )
    submitted = st.form_submit_button("Run", type="primary")

if submitted:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                result = run_agent(question.strip())
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.stop()
        st.subheader("Answer")
        st.success(result["final_answer"] or result["tool_result"] or "No answer produced.")
        with st.expander("Steps"):
            for i, step in enumerate(result["trace"], 1):
                st.markdown(f"{i}. {step}")
