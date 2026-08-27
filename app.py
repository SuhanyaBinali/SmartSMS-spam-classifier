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

        st.subheader("Analysis Result")



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
# Footer
# ==========================================

st.divider()

st.caption(
    "🛡️ SmartSMS · Machine Learning SMS Spam Classifier"
)

st.caption(
    "Built with Python · Scikit-learn · Streamlit"
)