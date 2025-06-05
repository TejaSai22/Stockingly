import streamlit as st

st.title("Minimal Test App")
st.write("If you can see this, Streamlit is working correctly!")

# Add a simple interactive element
name = st.text_input("Enter your name", "")
if name:
    st.write(f"Hello, {name}!")
