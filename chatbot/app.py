"""Minimal host app for testing the chatbot on its own."""
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from vornis_chatbot import render_vornis_chatbot

st.set_page_config(page_title="Vornis AI Chatbot", page_icon="🪐")
render_vornis_chatbot()
