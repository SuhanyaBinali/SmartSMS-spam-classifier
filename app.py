## In here this has streamlit code part for the web interface .
# -----------------------------------------------------------------------
import streamlit as st

from src.predict import predict_spam_details




st.set_page_config(
    page_title="SmartSMS - Spam Detector",
    page_icon="🛡️",
    layout="centered"
)




st.title("SmartSMS")

st.subheader("AI-Powered SMS Spam Detection")

st.write(
    "Detect suspicious SMS messages using Natural Language Processing "
    "and Machine Learning."
)

st.caption("NLP  •  TF-IDF  •  Linear SVM")

st.divider()




st.subheader("Analyze Your Message")

message = st.text_area(
    "Enter an SMS message",
    height=150,
    placeholder="Type or paste your SMS message here..."
)




character_count = len(message)
word_count = len(message.split())

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Characters",
        character_count
    )

with col2:
    st.metric(
        "Words",
        word_count
    )




if st.button(
    "Analyze Message",
    use_container_width=True
):

    if not message.strip():

        st.warning(
            "⚠️ Please enter an SMS message before analyzing."
        )

    else:

        result = predict_spam_details(message)

        prediction = result["prediction"]
        decision_score = result["decision_score"]

        st.divider()

        st.subheader("🔎 Analysis Result")



        if prediction == "spam":

            st.error(
                "🚨 SPAM MESSAGE DETECTED"
            )

            st.write(
                "The machine learning model classified this "
                "message as spam."
            )

            st.metric(
                "Model Decision Score",
                f"{decision_score:.4f}"
            )

            st.caption(
                "A positive Linear SVM decision score indicates "
                "the message is on the spam side of the decision boundary."
            )

            st.warning(
                "⚠️ Be cautious with messages requesting money, "
                "personal information, prizes, or urgent action."
            )




        else:

            st.success(
                "✅ LEGITIMATE MESSAGE"
            )

            st.write(
                "The machine learning model classified this "
                "message as ham (not spam)."
            )

            st.metric(
                "Model Decision Score",
                f"{decision_score:.4f}"
            )

            st.caption(
                "A negative Linear SVM decision score indicates "
                "the message is on the legitimate side of the decision boundary."
            )




# ==========================================
# Model Performance
# ==========================================

st.divider()

st.subheader("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Accuracy",
        "97.97%"
    )

with col2:
    st.metric(
        "Precision",
        "97.41%"
    )

with col3:
    st.metric(
        "Recall",
        "86.26%"
    )

with col4:
    st.metric(
        "F1 Score",
        "91.50%"
    )



# ==========================================
# How the Model Works
# ==========================================

st.divider()

st.subheader("📚 How SmartSMS Works")

st.write(
    "The message passes through a simple machine learning pipeline "
    "before the final spam or legitimate prediction is made."
)

with st.expander("🔍 View the ML Pipeline", expanded=False):

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown("### 📩")
        st.markdown("**SMS**")
        st.caption("User message")

    with col2:
        st.markdown("### 🧹")
        st.markdown("**Text Cleaning**")
        st.caption("Preprocess text")

    with col3:
        st.markdown("### 📊")
        st.markdown("**TF-IDF**")
        st.caption("Extract features")

    with col4:
        st.markdown("### 🤖")
        st.markdown("**Linear SVM**")
        st.caption("Classify message")

    with col5:
        st.markdown("### 🔎")
        st.markdown("**Result**")
        st.caption("Spam / Ham")






# ==========================================
# About the Model
# ==========================================

st.divider()

with st.expander("ℹ️ About SmartSMS"):

    st.write(
        """
        SmartSMS is a machine learning-based SMS classification system.

        The application uses Natural Language Processing (NLP) techniques
        to preprocess messages, TF-IDF for text feature extraction, and a
        Linear Support Vector Machine (Linear SVM) to classify messages as
        either spam or legitimate (ham).

        The Linear SVM provides a decision score rather than a calibrated
        probability. Therefore, the decision score shown above should not
        be interpreted as a percentage confidence.
        """
    )


# ==========================================
# Footer
# ==========================================

st.divider()

st.caption(
    "🛡️ SmartSMS · Machine Learning SMS Spam Classifier"
)

st.caption(
    "Built with Python · Scikit-learn · Streamlit"
)