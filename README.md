
# CareerVihari AI - Your Career Guidance Chatbot 🚀

![CareerVihari AI Banner](favcon1.jpg)

**CareerVihari AI** is an interactive chatbot designed to guide students and professionals in exploring career paths, educational options, and entrance exams. Built with Streamlit, this AI-powered chatbot provides personalized career advice based on the user's educational stage (e.g., Post-10th, Post-12th, Undergraduate, Postgraduate) and interests (e.g., MPC, BiPC, Commerce, MBA). Whether you're a student deciding on a stream after 10th or a postgraduate exploring career opportunities, CareerVihari AI is here to help you unlock your dream career! 🌟

## Features

- **Personalized Career Guidance**: Get tailored advice based on your educational stage and stream/major.
- **Exam Information**: Learn about entrance exams like AP POLYCET, AP EAPCET, and more.
- **Career Path Exploration**: Discover career options for streams like MPC (engineering, architecture), BiPC (medicine, biotechnology), Commerce (accounting, business), and postgraduate fields like MBA.
- **Interactive Chat Interface**: Engage in a conversational flow with a user-friendly interface.
- **Custom Favicon**: The app uses a custom favicon (`favcon1.jpg`) for a branded experience.
- **Responsive Design**: Built with Streamlit for a seamless experience on desktop and mobile devices.

## Live Demo

Try out the live version of CareerVihari AI here: [CareerVihari AI Live](https://your-app-name.streamlit.app)

## Project Structure

career-vihari-ai/
│
├── chatbot.py              # Main Streamlit app script
├── requirements.txt        # Python dependencies
├── intent_new.json         # Intents for the chatbot's NLP model
├── words_new.pkl           # Pickle file for words vocabulary
├── classes_new.pkl         # Pickle file for intent classes
├── vectorizer_new.pkl      # Pickle file for the vectorizer
├── model_new.h5            # Trained TensorFlow model for intent classification
├── favcon1.jpg             # Custom favicon for the app
└── README.md               # Project documentation

## Prerequisites

To run this project locally, ensure you have the following installed:

- Python 3.7 or higher
- Git (for cloning the repository)
- A code editor (e.g., VS Code)

## Setup Instructions

Follow these steps to set up and run the project locally:

1. **Clone the Repository**:

   git clone https://github.com/your-username/career-vihari-ai.git
   cd career-vihari-ai


2. **Create a Virtual Environment** (optional but recommended):

   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate


3. **Install Dependencies**:

   pip install -r requirements.txt


4. **Download NLTK Data**:
   The app uses NLTK for natural language processing. The required data (`punkt` and `wordnet`) is downloaded automatically when you run the script. Ensure you have an internet connection the first time you run the app.

5. **Run the App**:

   streamlit run chatbot.py

   - This will start the app locally, and you can access it at `http://localhost:8501` in your browser.

## Deployment

The app is deployed on **Streamlit Community Cloud** for public access. To deploy your own instance:

1. **Push Your Code to GitHub**:
   - Create a GitHub repository and push your project files.
   - Ensure all required files (e.g., `chatbot.py`, `requirements.txt`, `favcon1.jpg`, model files) are included.

2. **Deploy on Streamlit Community Cloud**:
   - Sign up at [Streamlit Community Cloud](https://streamlit.io/cloud).
   - Create a new app, select your GitHub repository, and specify `chatbot.py` as the main script.
   - Deploy the app and share the public URL.

For detailed deployment instructions, refer to the [Streamlit Community Cloud Documentation](https://docs.streamlit.io/streamlit-community-cloud/get-started).

## Usage

1. **Start the Chat**:
   - Open the app in your browser.
   - Click the "Explore Careers Together" button to begin.

2. **Provide Your Details**:
   - Enter your name and email when prompted.
   - Select your educational stage (Post-10th, Post-12th, Undergraduate, or Postgraduate).

3. **Explore Career Options**:
   - Based on your stage, the chatbot will guide you through options like choosing a stream (e.g., MPC, BiPC, Commerce), learning about entrance exams (e.g., AP POLYCET, AP EAPCET), or exploring career paths (e.g., engineering, medicine, business).

4. **Ask Questions**:
   - Ask about careers, exams, or further studies. For example:
     - "What careers can I pursue with MPC?"
     - "Tell me about AP POLYCET."
     - "What can I do with an MBA?"

5. **End the Chat**:
   - Type "bye", "goodbye", "exit", or "quit" to end the conversation and restart the app.

## Abbreviations

Here are some common abbreviations used in the project and related fields:

- **ML**: Machine Learning
- **DL**: Deep Learning
- **ECG**: Electrocardiogram
- **RF**: Random Forest
- **GB**: Gradient Boosting
- **CSV**: Comma-Separated Values
- **IDE**: Integrated Development Environment
- **AP POLYCET**: Andhra Pradesh Polytechnic Common Entrance Test
- **AP EAPCET**: Andhra Pradesh Engineering, Agriculture, and Pharmacy Common Entrance Test
- **MPC**: Maths, Physics, Chemistry
- **BiPC**: Biology, Physics, Chemistry

## Technologies Used

- **Streamlit**: For building the interactive web app.
- **Python**: Core programming language.
- **NLTK**: For natural language processing (tokenization, lemmatization).
- **TensorFlow**: For the intent classification model.
- **NumPy**: For numerical operations.
- **GitHub**: For version control and deployment.
- **Streamlit Community Cloud**: For hosting the live app.

## Contributing

Contributions are welcome! If you'd like to contribute to CareerVihari AI, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes and commit them (`git commit -m "Add your feature"`).
4. Push to your branch (`git push origin feature/your-feature`).
5. Open a pull request with a description of your changes.

## Issues and Feedback

If you encounter any issues or have feedback, please open an issue on the [GitHub Issues page](https://github.com/sigmauday/career-vihari-ai/issues). We’d love to hear from you!


## Acknowledgments

- Inspired by the need for accessible career guidance for students.
- Thanks to the Streamlit community for their amazing tools and documentation.
- Abbreviations section inspired by a friend's project.

## Contact

For inquiries, reach out to [udaykiranparnapalli@gmail.com](mailto:your-udaykiranparnapalli@gmail.com) or connect with me on [GitHub](https://github.com/sigmauday).


Happy career exploring with CareerVihari AI! 🌟
