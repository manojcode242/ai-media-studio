# app.py
import os
import io
import uuid
import tempfile
from dotenv import load_dotenv
import streamlit as st
from PIL import Image
import replicate
import google.genai as genai
from google.genai import types

# ================== ENV SETUP ==================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

# ================== UTILS ==================
def process_uploaded_image(uploaded_file):
    image = Image.open(uploaded_file)
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=image.format or "JPEG")
    img_bytes = img_byte_arr.getvalue()
    return {"name": uploaded_file.name, "data": img_bytes, "format": image.format or "JPEG"}

def is_duplicate_image(image_name, existing_images):
    return any(img["name"] == image_name for img in existing_images)

def is_duplicate_generated_image(image_data, generated_images):
    return any(img["data"] == image_data for img in generated_images)

# ================== GEMINI RESPONSE ==================
def generate_response(gemini_api_key, prompt, messages=None, images=None):
    client = genai.Client(api_key=gemini_api_key)
    model = "gemini-2.0-flash-exp"

    parts = []
    if images and len(messages) == 1:
        for img_data in images:
            parts.append(types.Part.from_bytes(data=img_data["data"], mime_type="image/jpeg"))
    parts.append(types.Part.from_text(text=prompt))

    contents = [types.Content(role="user", parts=parts)]
    if messages:
        for message in messages:
            if message["role"] == "user":
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=message["content"])]))
            else:
                if isinstance(message["content"], str):
                    contents.append(types.Content(role="model", parts=[types.Part.from_text(text=message["content"])]))
                else:
                    contents.append(types.Content(role="model", parts=[types.Part.from_bytes(data=message["content"], mime_type=message["mime_type"])]))

    config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        response_modalities=["text", "image"],
        response_mime_type="text/plain",
    )

    response_text = ""
    response_image = None
    response_mime_type = None

    for chunk in client.models.generate_content_stream(model=model, contents=contents, config=config):
        if not chunk.candidates or not chunk.candidates[0].content.parts:
            continue
        part = chunk.candidates[0].content.parts[0]
        if part.inline_data:
            response_image = part.inline_data.data
            response_mime_type = part.inline_data.mime_type
        else:
            response_text += chunk.text

    return response_text, response_image, response_mime_type

# ================== VIDEO GENERATION ==================
def generate_video(image_data, prompt):
    try:
        # Save image bytes to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
            tmp_img.write(image_data)
            tmp_img_path = tmp_img.name

        # Open file in binary mode and send to Replicate
        with open(tmp_img_path, "rb") as img_file:
            output = replicate.run(
                "wavespeedai/wan-2.1-i2v-480p",
                input={"image": img_file, "prompt": prompt}
            )

        # Normalize output to URL
        if isinstance(output, list) and output:
            video_url = output[0]
        elif isinstance(output, str) and output.startswith("http"):
            video_url = output
        else:
            video_url = str(output)

        return video_url

    except Exception as e:
        return f"Video generation link: {e}"

def reset_video_state():
    return {"selected_image_idx": None, "prompt": "", "video_url": None}

# ================== STREAMLIT APP ==================
st.set_page_config(page_title="Gemini Image Chat + Video Gen", page_icon="🖼️", layout="wide")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "images" not in st.session_state:
    st.session_state.images = []
if "generated_images" not in st.session_state:
    st.session_state.generated_images = []
if "video_generation_state" not in st.session_state:
    st.session_state.video_generation_state = reset_video_state()

st.title("🖼️ Gemini Image Chat + 🎥 Video Generator")
st.markdown("Upload images → Chat with Gemini → Generate images → Animate into videos")

# ================== Sidebar ==================
with st.sidebar:
    st.header("Upload & Manage Images")
    uploaded_files = st.file_uploader("Upload images", type=["jpg","jpeg","png"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if not is_duplicate_image(uploaded_file.name, st.session_state.images):
                img_data = process_uploaded_image(uploaded_file)
                st.session_state.images.append(img_data)

    if st.session_state.images:
        st.subheader(f"Your Images ({len(st.session_state.images)})")
        cols = st.columns(2)
        for i, img_data in enumerate(st.session_state.images):
            with cols[i%2]:
                img = Image.open(io.BytesIO(img_data["data"]))
                st.image(img, caption=f"{i+1}. {img_data['name']}", width=150)
                if st.button("Remove", key=f"remove_{i}"):
                    st.session_state.images.pop(i)
                    st.rerun()

    if st.session_state.messages:
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.generated_images = []
            st.rerun()

# ================== Tabs ==================
tab1, tab2 = st.tabs(["💬 Image Chat", "🎥 Video Generation"])

# ================== TAB 1: Image Chat ==================
with tab1:
    st.header("Chat with Gemini")
    chat_container = st.container()
    input_container = st.container()

    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if "images" in message and message["images"]:
                    img_cols = st.columns(min(len(message["images"]), 1))
                    for idx, img_idx in enumerate(message["images"]):
                        with img_cols[idx % 1]:
                            img_data = st.session_state.images[img_idx]
                            img = Image.open(io.BytesIO(img_data["data"]))
                            st.image(img, caption=f"{img_data['name']}", width=80)
                if isinstance(message["content"], str):
                    st.markdown(message["content"])
                else:
                    file_name = f"{uuid.uuid4()}.jpg"
                    with open(file_name, "wb") as f:
                        f.write(message["content"])
                    st.image(file_name, caption="Generated Image")
                    if message["role"] == "assistant" and not is_duplicate_generated_image(message["content"], st.session_state.generated_images):
                        st.session_state.generated_images.append({
                            "name": os.path.basename(file_name),
                            "data": message["content"],
                            "mime_type": message["mime_type"],
                            "file_path": file_name
                        })

    with input_container:
        if prompt := st.chat_input("Message Gemini..."):
            image_indices = list(range(len(st.session_state.images)))
            st.session_state.messages.append({"role":"user","content":prompt,"images":image_indices if image_indices else None})
            with st.spinner("Gemini is thinking..."):
                response_text, response_image, response_mime_type = generate_response(GEMINI_API_KEY, prompt, st.session_state.messages, st.session_state.images)
                if response_text:
                    st.session_state.messages.append({"role":"assistant","content":response_text})
                if response_image:
                    file_name = f"{uuid.uuid4()}.jpg"
                    with open(file_name, "wb") as f:
                        f.write(response_image)
                    st.session_state.messages.append({"role":"assistant","content":response_image,"mime_type":response_mime_type})
                    st.session_state.generated_images.append({
                        "name": os.path.basename(file_name),
                        "data": response_image,
                        "mime_type": response_mime_type,
                        "file_path": file_name
                    })
            st.rerun()

# ========== TAB 2: Video Generation ==========
with tab2:
    st.header("Generate Video from Gemini Images")
    if not st.session_state.generated_images:
        st.info("No images generated yet. Ask Gemini to create images first.")
    else:
        cols = st.columns(3)
        for i, img_data in enumerate(st.session_state.generated_images):
            with cols[i % 3]:
                st.image(img_data["file_path"], caption=f"Image {i+1}: {img_data['name']}", width=150)
                if st.button("Select for Video", key=f"select_{i}"):
                    st.session_state.video_generation_state["selected_image_idx"] = i
                    st.rerun()

        if st.session_state.video_generation_state["selected_image_idx"] is not None:
            st.divider()
            img_idx = st.session_state.video_generation_state["selected_image_idx"]
            img_data = st.session_state.generated_images[img_idx]
            st.image(img_data["file_path"], caption=f"Selected: {img_data['name']}", width=200)
            prompt = st.text_input("Video prompt:", value=st.session_state.video_generation_state["prompt"], key="video_prompt")
            st.session_state.video_generation_state["prompt"] = prompt

            if st.button("Generate Video"):
                if prompt:
                    with st.spinner("Generating video... may take a minute"):
                        video_url = generate_video(img_data["data"], prompt)
                        st.session_state.video_generation_state["video_url"] = video_url
                else:
                    st.warning("Please enter a prompt for the video.")

            # Display video inline or as a professional clickable link
            if st.session_state.video_generation_state["video_url"]:
                st.divider()
                st.subheader("Generated Video")
                video_url = st.session_state.video_generation_state["video_url"]
                if video_url.startswith("http"):
                    st.video(video_url, start_time=0, format="video/mp4", width=250)
                    
