import streamlit as st
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import time
import os
import re
import html

os.environ["PYTHONIOENCODING"] = "utf-8"

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

st.set_page_config(page_title="CareerVihari AI", page_icon="favcon1.jpg", layout="centered")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'chat_started' not in st.session_state:
    st.session_state.chat_started = False
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}
if 'conversation_state' not in st.session_state:
    st.session_state.conversation_state = 'initial'
if 'stage_prompt_displayed' not in st.session_state:
    st.session_state.stage_prompt_displayed = False
if 'show_undergrad_form' not in st.session_state:
    st.session_state.show_undergrad_form = False
if 'show_postgrad_form' not in st.session_state:
    st.session_state.show_postgrad_form = False
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

stream_career_paths = {
    "MPC": ["engineering", "architecture", "physics research"],
    "BIPC": ["medicine", "biotechnology", "nursing"],
    "COMMERCE": ["accounting", "finance", "business management"],
}

major_higher_study = {
    "computer science": ["M.Tech in Computer Science", "MS in Computer Science", "MBA in Technology Management"],
    "mechanical engineering": ["M.Tech in Mechanical Engineering", "MS in Mechanical Engineering", "MBA"],
    "biology": ["M.Sc in Biology", "PhD in Biological Sciences", "MBA in Biotechnology Management"],
    "mba": ["PhD in Management", "Executive MBA", "Specialized Certifications in Finance or Marketing"],
}

@st.cache_resource
def load_chatbot_data():
    try:
        with open('intent_new.json', 'r', encoding='utf-8') as file:
            data = file.read()
            if not data.strip():
                raise ValueError("intent_new.json is empty!")
            intents = json.loads(data)
        
        print("Type of intents:", type(intents))
        if isinstance(intents, (list, tuple)):
            print("Intents content:", intents[:2])
        else:
            print("Intents content (not a list):", intents)
        
        if isinstance(intents, str):
            intents = json.loads(intents)
        
        if isinstance(intents, dict) and 'intents' in intents:
            intents = intents['intents']
        
        if not isinstance(intents, list):
            raise ValueError(f"Expected intents to be a list, but got {type(intents)}")
        if not all(isinstance(i, dict) and 'tag' in i for i in intents):
            raise ValueError("Intents must be a list of dictionaries with 'tag' keys")
        
        words = pickle.load(open('words_new.pkl', 'rb'))
        classes = pickle.load(open('classes_new.pkl', 'rb'))
        vectorizer = pickle.load(open('vectorizer_new.pkl', 'rb'))
        model = load_model('model_new.h5')
        
        return intents, words, classes, vectorizer, model
    except Exception as e:
        st.error(f"Error loading chatbot data: {str(e)}")
        raise Exception(f"Error loading chatbot data: {str(e)}")

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, vectorizer):
    sentence_words = clean_up_sentence(sentence)
    sentence = ' '.join(sentence_words)
    vector = vectorizer.transform([sentence]).toarray()
    return vector

def predict_class(sentence, model, words, classes, vectorizer):
    bow = bag_of_words(sentence, vectorizer)
    res = model.predict(bow)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    if not return_list:
        return_list.append({'intent': 'unknown', 'probability': '1.0'})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json
    
    print("Type of list_of_intents:", type(list_of_intents))
    print("First item in list_of_intents:", list_of_intents[0] if list_of_intents else "Empty")
    
    if tag == 'unknown':
        result = "I’m not sure how to help with that. Could you tell me more or ask something else?"
    else:
        for i in list_of_intents:
            if not isinstance(i, dict):
                raise TypeError(f"Expected intent to be a dictionary, but got {type(i)}: {i}")
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
        else:
            result = "I’m not sure how to respond to that."
    
    name = st.session_state.user_data.get('name', 'friend')
    stage = st.session_state.user_data.get('stage', '')
    stream = st.session_state.user_data.get('stream', 'your stream')
    major = st.session_state.user_data.get('major', 'your major')
    year = st.session_state.user_data.get('year', 'your year')
    field = st.session_state.user_data.get('field', 'your field')
    
    if '{career_path}' in result:
        career_path = random.choice(stream_career_paths.get(stream.upper(), ["various fields"]))
        result = result.replace('{career_path}', career_path)
    if '{higher_study}' in result:
        major_lower = major.lower()
        higher_study_options = major_higher_study.get(major_lower, ["a Master's degree"])
        higher_study = random.choice(higher_study_options)
        result = result.replace('{higher_study}', higher_study)
    if '{field}' in result:
        result = result.replace('{field}', field)
    
    result = result.replace('{name}', name).replace('{stage}', stage).replace('{stream}', stream).replace('{major}', major).replace('{year}', year)
    
    return result, tag

def custom_escape(message):
    escape_chars = {
        '<': '&lt;',
        '>': '&gt;',
        '&': '&amp;',
        '"': '&quot;',
        "'": '&apos;'
    }
    for char, escaped in escape_chars.items():
        message = message.replace(char, escaped)
    return message

st.markdown("""
<style>
    .chat-container { 
        background-color: white; 
        border-radius: 15px; 
        padding: 20px; 
        max-width: 700px; 
        margin: 0 auto; 
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); 
        display: flex; 
        flex-direction: column; 
        height: 500px; 
    }
    .chat-header { 
        background: linear-gradient(90deg, #6a1b9a, #1e88e5); 
        color: white; 
        padding: 15px; 
        border-radius: 10px 10px 0 0; 
        text-align: center; 
        font-size: 24px; 
        font-weight: bold; 
        position: sticky; 
        top: 0; 
        z-index: 1; 
    }
    .chat-messages { 
        flex: 1; 
        overflow-y: auto; 
        padding: 10px 0; 
        margin-bottom: 10px; 
    }
    .chat-message { 
        margin: 10px 0; 
        padding: 10px; 
        border-radius: 10px; 
        max-width: 70%; 
        word-wrap: break-word; 
    }
    .bot-message { 
        background-color: #e3f2fd; 
        color: #1e88e5; 
        margin-right: auto; 
    }
    .user-message { 
        background-color: #ede7f6; 
        color: #6a1b9a; 
        margin-left: auto; 
    }
    .stButton>button { 
        background-color: #6a1b9a; 
        color: white; 
        border-radius: 10px; 
        padding: 10px 20px; 
        border: none; 
        margin: 5px auto; 
        display: block; 
    }
    .stButton>button:hover { 
        background-color: #1e88e5; 
    }
    .stChatInput { 
        border-radius: 10px; 
        border: 1px solid #6a1b9a; 
        padding: 10px; 
        width: 100%; 
        box-sizing: border-box; 
    }
    .welcome-container { 
        text-align: center; 
        margin: 10px 0; 
    }
    .welcome-heading { 
        background: linear-gradient(90deg, #6a1b9a, #1e88e5); 
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent; 
        font-size: 36px; 
        font-weight: bold; 
        margin-bottom: 10px; 
    }
    .tagline { 
        text-align: center; 
        font-size: 20px; 
        color: #333; 
        margin-bottom: 10px; 
    }
    .stTextInput>div>input { 
        border-radius: 10px; 
        border: 1px solid #6a1b9a; 
        padding: 10px; 
    }
    .stSelectbox { 
        border-radius: 10px; 
        border: 1px solid #6a1b9a; 
        padding: 10px; 
    }
</style>
<script>
    function scrollToBottom() { 
        const chatMessages = document.querySelector('.chat-messages'); 
        if (chatMessages) { 
            chatMessages.scrollTop = chatMessages.scrollHeight; 
        } 
    }
    function setupAutoScroll() {
        const chatMessages = document.querySelector('.chat-messages');
        if (chatMessages) {
            const observer = new MutationObserver((mutations) => {
                mutations.forEach(() => {
                    scrollToBottom();
                });
            });
            observer.observe(chatMessages, { childList: true, subtree: true });
            scrollToBottom();
        }
    }
    function moveChatElements() { 
        setTimeout(() => { 
            const chatControls = document.querySelector('.chat-controls'); 
            if (chatControls) { 
                const chatInput = document.querySelector('[data-testid="stChatInput"]'); 
                if (chatInput && !chatControls.contains(chatInput)) { 
                    chatControls.appendChild(chatInput.parentElement); 
                } 
                const buttons = document.querySelectorAll('[data-testid="stButton"]'); 
                buttons.forEach(button => { 
                    if (!chatControls.contains(button)) { 
                        chatControls.appendChild(button.parentElement); 
                    } 
                }); 
                const textInputs = document.querySelectorAll('[data-testid="stTextInput"]'); 
                textInputs.forEach(input => { 
                    if (!chatControls.contains(input)) { 
                        chatControls.appendChild(input.parentElement); 
                    } 
                }); 
                const selectboxes = document.querySelectorAll('[data-testid="stSelectbox"]'); 
                selectboxes.forEach(selectbox => { 
                    if (!chatControls.contains(selectbox)) { 
                        chatControls.appendChild(selectbox.parentElement); 
                    } 
                }); 
            } 
            setupAutoScroll();
        }, 100); 
    }
    document.addEventListener('DOMContentLoaded', moveChatElements); 
    window.addEventListener('load', moveChatElements); 
    const observer = new MutationObserver(moveChatElements); 
    observer.observe(document.body, { childList: true, subtree: true });
</script>
""", unsafe_allow_html=True)

def add_message(sender, message, css_class):
    escaped_message = custom_escape(message)
    st.session_state.chat_history.append({"sender": sender, "message": escaped_message, "type": css_class})

def main():
    if not st.session_state.model_loaded:
        with st.spinner("Loading chatbot model..."):
            try:
                st.session_state.intents, st.session_state.words, st.session_state.classes, st.session_state.vectorizer, st.session_state.model = load_chatbot_data()
                st.session_state.model_loaded = True
            except Exception as e:
                st.error(f"Error loading model: {e}")
                return
    
    if not st.session_state.chat_started:
        st.markdown("""
        <div class='welcome-container'>
            <div class='welcome-heading'>CareerVihari AI</div>
            <div class='tagline'>Ready to Discover Your Dream Career?</div>
        </div>
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Explore Careers Together"):
                st.session_state.chat_started = True
                st.session_state.conversation_state = 'initial'
                st.session_state.chat_history = []
                add_message("Bot", "Hey there, dreamer! I'm CareerVihari AI, your guide to unlocking an amazing career path! 🚀", "bot-message")
                add_message("Bot", "I can't wait to get to know you better—what's your name? 🌟", "bot-message")
                st.rerun()
    
    if st.session_state.chat_started:
        container_html = '<div class="chat-container"><div class="chat-header">CareerVihari AI - Your Career Guidance Chatbot</div><div class="chat-messages">'
        for chat in st.session_state.chat_history:
            if isinstance(chat, dict) and "sender" in chat and "message" in chat and "type" in chat:
                container_html += f'<div class="chat-message {chat["type"]}">{chat["message"]}</div>'
        container_html += '</div><div class="chat-controls"></div></div>'
        st.markdown(container_html, unsafe_allow_html=True)

        if st.session_state.conversation_state == 'stage_selection' and not st.session_state.stage_prompt_displayed:
            name = st.session_state.user_data.get("name", "friend")
            add_message("Bot", f"Thank you, {name}! Now, please select your educational stage:", "bot-message")
            st.session_state.stage_prompt_displayed = True
            st.rerun()

        if st.session_state.conversation_state == 'stage_selection' and st.session_state.stage_prompt_displayed:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button("Post-10th"):
                    add_message("User", "Post-10th", "user-message")
                    st.session_state.user_data["stage"] = "Post-10th"
                    st.session_state.conversation_state = 'post_10th'
                    add_message("Bot", f"Congrats on finishing 10th, {st.session_state.user_data['name']}! Do you know which group you want to take, or do you need help deciding?", "bot-message")
                    st.rerun()
            with col2:
                if st.button("Post-12th"):
                    add_message("User", "Post-12th", "user-message")
                    st.session_state.user_data["stage"] = "Post-12th"
                    st.session_state.conversation_state = 'post_12th'
                    add_message("Bot", "Congrats on finishing 12th! What’s your stream?", "bot-message")
                    st.rerun()
            with col3:
                if st.button("Undergraduate"):
                    add_message("User", "Undergraduate", "user-message")
                    st.session_state.user_data["stage"] = "Undergraduate"
                    st.session_state.conversation_state = 'undergraduate'
                    add_message("Bot", "Welcome, undergrad! What’s your major, and what year are you in?", "bot-message")
                    st.session_state.show_undergrad_form = True
                    st.rerun()
            with col4:
                if st.button("Postgraduate"):
                    add_message("User", "Postgraduate", "user-message")
                    st.session_state.user_data["stage"] = "Postgraduate"
                    st.session_state.conversation_state = 'postgraduate'
                    add_message("Bot", "You’re a postgraduate—impressive! What’s your field of study?", "bot-message")
                    st.session_state.show_postgrad_form = True
                    st.rerun()

        if st.session_state.conversation_state == 'undergraduate' and st.session_state.show_undergrad_form:
            col1, col2 = st.columns([1, 1])
            with col1:
                major = st.text_input("Enter your major (e.g., Computer Science)", key="undergrad_major")
            with col2:
                year = st.selectbox("Select your year of study", ["1st Year", "2nd Year", "3rd Year", "4th Year"], key="undergrad_year")
            col_submit = st.columns([1, 1, 1])[1]
            with col_submit:
                if st.button("Submit"):
                    if major:
                        add_message("User", f"{major}, {year}", "user-message")
                        st.session_state.user_data["major"] = major
                        st.session_state.user_data["year"] = year
                        name = st.session_state.user_data.get("name", "friend")
                        add_message("Bot", f"Great to know, {name}! You’re a {year} {major} student. Are you enjoying your course?", "bot-message")
                        st.session_state.conversation_state = 'awaiting_course_enjoyment_response'
                        st.session_state.show_undergrad_form = False
                        st.rerun()

        if st.session_state.conversation_state == 'postgraduate' and st.session_state.show_postgrad_form:
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                field = st.text_input("Enter your field of study (e.g., MBA)", key="postgrad_field")
            col_submit = st.columns([1, 1, 1])[1]
            with col_submit:
                if st.button("Submit"):
                    if field:
                        add_message("User", field, "user-message")
                        st.session_state.user_data["field"] = field
                        name = st.session_state.user_data.get("name", "friend")
                        add_message("Bot", f"Sweet, {field} it is! What’s on your mind—career or research? Check out [ResearchGate](https://www.researchgate.net) for research insights.", "bot-message")
                        st.session_state.conversation_state = 'postgraduate_options'
                        st.session_state.show_postgrad_form = False
                        st.rerun()

        if not st.session_state.show_undergrad_form and not st.session_state.show_postgrad_form:
            user_input = st.chat_input("Type your message here...")
            
            if user_input:
                add_message("User", user_input, "user-message")
                
                if user_input.lower() in ["bye", "goodbye", "exit", "quit"]:
                    add_message("Bot", "Goodbye! Have a great day!", "bot-message")
                    time.sleep(2)
                    st.info("Chat ended. Restarting...")
                    time.sleep(1)
                    st.session_state.chat_history = []
                    st.session_state.chat_started = False
                    st.session_state.conversation_state = 'initial'
                    st.session_state.user_data = {}
                    st.session_state.stage_prompt_displayed = False
                    st.session_state.show_undergrad_form = False
                    st.session_state.show_postgrad_form = False
                    st.rerun()
                    return
                
                intents = predict_class(user_input, st.session_state.model, st.session_state.words, st.session_state.classes, st.session_state.vectorizer)
                bot_response, intent = get_response(intents, st.session_state.intents)

                if st.session_state.conversation_state == 'initial':
                    if intents[0]['intent'] == 'unknown' and len(user_input.split()) <= 2:
                        name = user_input
                        st.session_state.user_data['name'] = name
                        st.session_state.conversation_state = 'asking_email'
                        bot_response = f"Arey {name}, what a cool name! Please give me your email next, okay?"
                    elif intents[0]['intent'] == 'initial_name' or "name" in user_input.lower():
                        name = user_input
                        if "my name is" in user_input.lower():
                            name = user_input.lower().replace("my name is", "").strip()
                        st.session_state.user_data['name'] = name
                        st.session_state.conversation_state = 'asking_email'
                        bot_response = f"Arey {name}, what a cool name! Please give me your email next, okay?"
                    else:
                        bot_response = "Sorry, I didn’t get that! Can you tell me your name to start?"

                elif st.session_state.conversation_state == 'asking_email':
                    if re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", user_input):
                        st.session_state.user_data['email'] = user_input
                        st.session_state.conversation_state = 'stage_selection'
                        bot_response = f"Thanks for sharing, {st.session_state.user_data['name']}! Let me check… is this correct: {user_input}?"
                    else:
                        bot_response = "Oops, that doesn’t look like a proper email. Can you try again?"

                elif st.session_state.conversation_state == 'stage_selection':
                    bot_response = "Please use the buttons to select your educational stage (Post-10th, Post-12th, Undergraduate, or Postgraduate)."
                    st.session_state.stage_prompt_displayed = True

                elif st.session_state.conversation_state == 'post_10th':
                    user_input_lower = user_input.lower()
                    if "yes" in user_input_lower:
                        bot_response = f"Great, {st.session_state.user_data['name']}! Which group are you thinking of taking—MPC, BiPC, or Commerce?"
                        st.session_state.conversation_state = 'post_10th_group_selection'
                    elif "no need" in user_input_lower or "don't need" in user_input_lower:
                        bot_response = f"Okay, {st.session_state.user_data['name']}! It sounds like you might not be ready to decide yet. Do you want to explore some career paths, or would you like to know more about the groups you can choose after 10th (like MPC, BiPC, or Commerce)?"
                        st.session_state.conversation_state = 'post_10th_clarification'
                    elif "help" in user_input_lower or "decide" in user_input_lower:
                        bot_response = f"Let’s explore your options, {st.session_state.user_data['name']}! After 10th, you can choose groups like MPC (Maths, Physics, Chemistry), BiPC (Biology, Physics, Chemistry), or Commerce. Which one are you interested in, or do you want to know more about them?"
                        st.session_state.conversation_state = 'post_10th_group_selection'
                    else:
                        intents = predict_class(user_input, st.session_state.model, st.session_state.words, st.session_state.classes, st.session_state.vectorizer)
                        bot_response, _ = get_response(intents, st.session_state.intents)

                elif st.session_state.conversation_state == 'post_10th_clarification':
                    user_input_lower = user_input.lower()
                    if "career" in user_input_lower:
                        bot_response = f"Great! Let’s explore some career paths. After 10th, you can choose groups like MPC (leads to engineering, architecture), BiPC (leads to medicine, biotechnology), or Commerce (leads to accounting, business). Which one sounds interesting to you?"
                        st.session_state.conversation_state = 'post_10th_group_selection'
                    elif "group" in user_input_lower or "mpc" in user_input_lower or "bipc" in user_input_lower or "commerce" in user_input_lower:
                        bot_response = f"Let’s dive deeper! After 10th, you can choose MPC (Maths, Physics, Chemistry), BiPC (Biology, Physics, Chemistry), or Commerce. Which one are you interested in, or do you want to know more about them?"
                        st.session_state.conversation_state = 'post_10th_group_selection'
                    else:
                        intents = predict_class(user_input, st.session_state.model, st.session_state.words, st.session_state.classes, st.session_state.vectorizer)
                        bot_response, _ = get_response(intents, st.session_state.intents)

                elif st.session_state.conversation_state == 'post_10th_group_selection':
                    user_input_lower = user_input.lower()
                    if "mpc" in user_input_lower:
                        st.session_state.user_data['stream'] = "MPC"
                        bot_response = f"Cool, you're a {st.session_state.user_data['stream']} student! What would you like to know—careers, exams, or something else? Explore [Official Website](https://example.com) for resources."
                        st.session_state.conversation_state = 'post_10th_mpc'
                    elif "bipc" in user_input_lower:
                        st.session_state.user_data['stream'] = "BIPC"
                        bot_response = f"Cool, you're a {st.session_state.user_data['stream']} student! What would you like to know—careers, exams, or something else? Explore [Official Website](https://example.com) for resources."
                        st.session_state.conversation_state = 'post_10th_bipc'
                    elif "commerce" in user_input_lower:
                        st.session_state.user_data['stream'] = "COMMERCE"
                        bot_response = f"Cool, you're a {st.session_state.user_data['stream']} student! What would you like to know—careers, exams, or something else? Explore [Official Website](https://example.com) for resources."
                        st.session_state.conversation_state = 'post_10th_commerce'
                    else:
                        bot_response = "Please choose a group: MPC, BiPC, or Commerce. Or let me know if you want more details about them!"

                elif st.session_state.conversation_state in ['post_10th_mpc', 'post_10th_bipc', 'post_10th_commerce']:
                    user_input_lower = user_input.lower()
                    if "career" in user_input_lower:
                        bot_response = f"Lots of career options for you, {st.session_state.user_data['name']}! With your background, you can look into {random.choice(stream_career_paths.get(st.session_state.user_data['stream'], ['various fields']))}. Want more details? Check out [National Career Service](https://www.ncs.gov.in) for job opportunities."
                    elif "exam" in user_input_lower or "eapcet" in user_input_lower:
                        bot_response = f"AP EAPCET is the new name for EAMCET in Andhra, {st.session_state.user_data['name']}! It’s for engineering, agri, and pharmacy courses after 12th. You need PCM for engineering, PCB for others. Exam’s in May—start with 12th books! Want prep hacks? Check out [AP EAPCET Official Website](https://cets.apsche.ap.gov.in/EAPCET)."
                    elif "polycet" in user_input_lower:
                        bot_response = f"AP POLYCET is a great option for you, {st.session_state.user_data['name']}! It’s an entrance exam for polytechnic diploma courses in Andhra Pradesh after 10th. You can pursue diplomas in engineering fields like Mechanical, Civil, or Electrical with your MPC background. The exam usually happens around April-May. Want to know more? Check out [AP POLYCET Official Website](https://polycetap.nic.in)."
                    else:
                        intents = predict_class(user_input, st.session_state.model, st.session_state.words, st.session_state.classes, st.session_state.vectorizer)
                        bot_response, _ = get_response(intents, st.session_state.intents)

                elif st.session_state.conversation_state == 'post_12th':
                    if 'stream' not in st.session_state.user_data:
                        stream = user_input.upper()
                        if stream in stream_career_paths:
                            st.session_state.user_data['stream'] = stream
                            career_path = random.choice(stream_career_paths[stream])
                            bot_response = f"Yay, {stream}! This can lead to {career_path}. What’s next—career options or entrance exams?"
                            st.session_state.conversation_state = 'post_12th_stream_provided'
                        else:
                            bot_response = "I’m not sure about that stream. Could you specify MPC, BiPC, Commerce, or another stream?"
                    else:
                        user_input_lower = user_input.lower()
                        if any(word in user_input_lower for word in ['mpc', 'bipc', 'commerce']):
                            bot_response = f"Yes, you mentioned your stream is {st.session_state.user_data['stream']}. What would you like to know about it?"
                        else:
                            intents = predict_class(user_input, st.session_state.model, st.session_state.words, st.session_state.classes, st.session_state.vectorizer)
                            bot_response, _ = get_response(intents, st.session_state.intents)

                elif st.session_state.conversation_state == 'post_12th_stream_provided':
                    user_input_lower = user_input.lower()
                    if any(word in user_input_lower for word in ['mpc', 'bipc', 'commerce']):
                        bot_response = f"Yes, you mentioned your stream is {st.session_state.user_data['stream']}. What would you like to know about it?"
                    else:
                        intents = predict_class(user_input, st.session_state.model, st.session_state.words, st.session_state.classes, st.session_state.vectorizer)
                        bot_response, _ = get_response(intents, st.session_state.intents)

                elif st.session_state.conversation_state == 'postgraduate_options':
                    user_input_lower = user_input.lower()
                    if "career" in user_input_lower:
                        bot_response = f"With an {st.session_state.user_data['field']} background, you can explore roles like management consultant, business analyst, or entrepreneur! Want to explore job opportunities? Check out [LinkedIn](https://www.linkedin.com) for career options."
                    elif "research" in user_input_lower:
                        bot_response = f"Research in {st.session_state.user_data['field']} is a great choice! You can dive into areas like organizational behavior, finance, or marketing strategies. Check out [ResearchGate](https://www.researchgate.net) for research insights."
                    else:
                        intents = predict_class(user_input, st.session_state.model, st.session_state.words, st.session_state.classes, st.session_state.vectorizer)
                        bot_response, _ = get_response(intents, st.session_state.intents)

                elif st.session_state.conversation_state == 'awaiting_course_enjoyment_response':
                    user_input_lower = user_input.lower()
                    if intent == 'yes' or "yes" in user_input_lower or "enjoying" in user_input_lower:
                        bot_response = f"Awesome, {st.session_state.user_data['name']}! I’m glad you’re enjoying your {st.session_state.user_data['major']} course. What’s next—jobs, internships, or further studies?"
                        st.session_state.conversation_state = 'undergraduate_options'
                    elif intent == 'no' or "no" in user_input_lower or "not" in user_input_lower:
                        bot_response = f"Oh no, you’re not liking it? Let’s explore options for a {st.session_state.user_data['year']} {st.session_state.user_data['major']} student—jobs, internships, or further studies? Check out [Internshala](https://internshala.com) for opportunities."
                        st.session_state.conversation_state = 'undergraduate_options'
                    else:
                        bot_response = "I’m not sure if you’re enjoying your course or not. Could you say 'yes' or 'no'?"

                add_message("Bot", bot_response, "bot-message")
                st.rerun()

if __name__ == "__main__":
    main()
