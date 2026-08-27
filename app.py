import streamlit as st
import pandas as pd
import re
import os
import math

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SmartSMS - Spam Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# ============================================================
# DARK THEME
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    html, body, [class*="css"] {
        font-family: Arial, Helvetica, sans-serif;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(30, 80, 140, 0.25),
                transparent 35%
            ),
            radial-gradient(
                circle at 90% 80%,
                rgba(0, 120, 100, 0.20),
                transparent 35%
            ),
            linear-gradient(
                135deg,
                #050b16 0%,
                #081525 50%,
                #061a1b 100%
            );

        color: #f8fafc;
    }


    /* ========================================================
       MAIN CONTENT WIDTH
       ======================================================== */

    .block-container {
        max-width: 1050px !important;
        padding-top: 3rem !important;
        padding-bottom: 4rem !important;
    }


    /* ========================================================
       REMOVE STREAMLIT DEFAULTS
       ======================================================== */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }


    /* ========================================================
       HERO
       ======================================================== */

    .hero-title {
        text-align: center;
        font-size: 52px;
        font-weight: 800;
        color: #ffffff;
        margin-top: 10px;
        margin-bottom: 8px;
        letter-spacing: -1px;
    }

    .hero-subtitle {
        text-align: center;
        font-size: 18px;
        color: #8fb6d5;
        margin-bottom: 22px;
    }

    .hero-badge {
        text-align: center;
        display: block;
        width: fit-content;
        margin: 0 auto;
        padding: 9px 20px;
        border-radius: 30px;
        background: rgba(30, 100, 150, 0.22);
        border: 1px solid rgba(80, 200, 190, 0.35);
        color: #8ff0d0;
        font-size: 14px;
        font-weight: 700;
    }


    /* ========================================================
       SECTION HEADINGS
       ======================================================== */

    .section-title {
        font-size: 32px;
        font-weight: 800;
        color: #ffffff;
        margin-top: 45px;
        margin-bottom: 6px;
    }

    .section-description {
        font-size: 16px;
        color: #8ba8bf;
        margin-bottom: 22px;
        line-height: 1.6;
    }


    /* ========================================================
       STREAMLIT TEXT AREA
       ======================================================== */

    label {
        color: #dcecff !important;
        font-weight: 700 !important;
    }

    textarea {
        background: #0b1828 !important;
        color: #f8fafc !important;
        border: 1px solid #294861 !important;
        border-radius: 14px !important;
        font-size: 16px !important;
        line-height: 1.6 !important;
    }

    textarea:focus {
        border-color: #2dd4bf !important;
        box-shadow: 0 0 0 2px rgba(45, 212, 191, 0.15) !important;
    }


    /* ========================================================
       BUTTON
       ======================================================== */

    .stButton > button {
        width: 100%;
        min-height: 52px;
        border: none !important;
        border-radius: 12px !important;

        background:
            linear-gradient(
                90deg,
                #2563eb,
                #0f766e
            ) !important;

        color: white !important;
        font-size: 16px !important;
        font-weight: 800 !important;

        box-shadow:
            0 8px 25px rgba(20, 100, 150, 0.25);

        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);

        box-shadow:
            0 12px 30px rgba(20, 150, 150, 0.35);
    }


    /* ========================================================
       METRIC CARDS
       ======================================================== */

    .metric-label {
        color: #86a7c2;
        font-size: 14px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .metric-value {
        color: #ffffff;
        font-size: 32px;
        font-weight: 800;
    }


    /* ========================================================
       PROCESS CARDS
       ======================================================== */

    .process-number {
        font-size: 14px;
        font-weight: 800;
        color: #67e8f9;
        margin-bottom: 8px;
    }

    .process-title {
        font-size: 21px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 10px;
    }

    .process-text {
        font-size: 15px;
        line-height: 1.7;
        color: #91a9bc;
    }


    /* ========================================================
       RESULT
       ======================================================== */

    .result-title {
        font-size: 28px;
        font-weight: 800;
        color: #ffffff;
        text-align: center;
    }

    .result-description {
        color: #a9bdcc;
        font-size: 16px;
        text-align: center;
        line-height: 1.6;
    }

    .confidence {
        text-align: center;
        font-size: 20px;
        font-weight: 800;
        color: #5eead4;
        margin-top: 15px;
    }


    /* ========================================================
       FOOTER
       ======================================================== */

    .footer-title {
        text-align: center;
        color: #dce8f2;
        font-size: 17px;
        font-weight: 600;
        margin-top: 35px;
    }

    .footer-line {
        text-align: center;
        color: #7995aa;
        font-size: 14px;
        margin-top: 10px;
    }

    .footer-tech {
        text-align: center;
        color: #5eead4;
        font-size: 13px;
        font-weight: 700;
        margin-top: 10px;
    }


    /* ========================================================
       EXPANDER
       ======================================================== */

    [data-testid="stExpander"] {
        background: rgba(10, 30, 45, 0.75) !important;
        border: 1px solid rgba(80, 170, 190, 0.30) !important;
        border-radius: 14px !important;
    }

    [data-testid="stExpander"] summary {
        color: #e5f4ff !important;
        font-weight: 700 !important;
    }


    /* ========================================================
       DIVIDER
       ======================================================== */

    hr {
        border-color: rgba(100, 160, 180, 0.18) !important;
        margin-top: 45px !important;
        margin-bottom: 35px !important;
    }


    /* ========================================================
       MOBILE
       ======================================================== */

    @media (max-width: 700px) {

        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }

        .hero-title {
            font-size: 38px;
        }

        .hero-subtitle {
            font-size: 15px;
        }

        .section-title {
            font-size: 26px;
        }

        .process-title {
            font-size: 19px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Clean SMS text before machine learning prediction.
    """

    text = str(text).lower()

    text = re.sub(
        r"http\S+|www\S+",
        " URL ",
        text
    )

    text = re.sub(
        r"\d+",
        " NUMBER ",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource
def train_model():

    possible_files = [
        "spam.csv",
        "Spam.csv",
        "SMSSpamCollection.csv",
        "sms_spam.csv",
        "data/spam.csv",
        "dataset/spam.csv"
    ]

    dataset_path = None

    for file in possible_files:

        if os.path.exists(file):
            dataset_path = file
            break

    if dataset_path is None:
        return None, None, None

    try:

        df = pd.read_csv(
            dataset_path,
            encoding="latin-1"
        )

    except Exception:

        try:

            df = pd.read_csv(
                dataset_path,
                encoding="utf-8"
            )

        except Exception:

            return None, None, None


    # --------------------------------------------------------
    # FIND COLUMNS
    # --------------------------------------------------------

    message_column = None
    label_column = None

    columns_lower = {
        str(col).lower(): col
        for col in df.columns
    }

    message_candidates = [
        "message",
        "text",
        "sms",
        "v2",
        "content"
    ]

    label_candidates = [
        "label",
        "category",
        "class",
        "type",
        "v1"
    ]


    for candidate in message_candidates:

        if candidate in columns_lower:

            message_column = columns_lower[candidate]
            break


    for candidate in label_candidates:

        if candidate in columns_lower:

            label_column = columns_lower[candidate]
            break


    # Standard SMS Spam Collection format
    if message_column is None and len(df.columns) >= 2:

        message_column = df.columns[1]


    if label_column is None and len(df.columns) >= 1:

        label_column = df.columns[0]


    if message_column is None or label_column is None:

        return None, None, None


    # --------------------------------------------------------
    # PREPARE DATA
    # --------------------------------------------------------

    df = df[
        [label_column, message_column]
    ].dropna()


    df[label_column] = (
        df[label_column]
        .astype(str)
        .str.lower()
        .str.strip()
    )


    df[message_column] = (
        df[message_column]
        .astype(str)
        .apply(clean_text)
    )


    # --------------------------------------------------------
    # LABELS
    # --------------------------------------------------------

    df["target"] = df[label_column].apply(
        lambda x:
            1
            if x in [
                "spam",
                "1",
                "true",
                "yes"
            ]
            else 0
    )


    X = df[message_column]
    y = df["target"]


    # --------------------------------------------------------
    # MACHINE LEARNING PIPELINE
    # --------------------------------------------------------

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True
                )
            ),

            (
                "classifier",
                LinearSVC(
                    C=1.0
                )
            )
        ]
    )


    pipeline.fit(X, y)


    return (
        pipeline,
        len(df),
        dataset_path
    )


# ============================================================
# LOAD MODEL
# ============================================================

model, dataset_size, dataset_used = train_model()


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    '<div style="text-align:center;font-size:45px;">🛡️</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-title">SmartSMS</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-subtitle">'
    'AI-Powered SMS Spam Detection System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-badge">'
    'NLP&nbsp;&nbsp;•&nbsp;&nbsp;TF-IDF&nbsp;&nbsp;•&nbsp;&nbsp;Linear SVM'
    '</div>',
    unsafe_allow_html=True
)


st.write("")


# ============================================================
# ANALYZE SECTION
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Analyze Your Message'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Enter an SMS message and let the machine learning '
    'model determine whether it is spam or legitimate.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# MESSAGE INPUT
# ============================================================

message = st.text_area(
    "SMS Message",
    placeholder="Enter your SMS message here...",
    height=160
)


# ============================================================
# METRICS
# ============================================================

characters = len(message)

words = (
    len(message.split())
    if message.strip()
    else 0
)


metric_col1, metric_col2 = st.columns(2)


with metric_col1:

    with st.container(border=True):

        st.markdown(
            '<div class="metric-label">Characters</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="metric-value">{characters}</div>',
            unsafe_allow_html=True
        )


with metric_col2:

    with st.container(border=True):

        st.markdown(
            '<div class="metric-label">Words</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="metric-value">{words}</div>',
            unsafe_allow_html=True
        )


st.write("")


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze = st.button(
    "🔍  Analyze Message",
    use_container_width=True
)


# ============================================================
# CLASSIFICATION
# ============================================================

if analyze:

    if not message.strip():

        st.warning(
            "Please enter an SMS message before analyzing."
        )


    elif model is None:

        st.error(
            "The machine learning model could not be loaded."
        )

        st.info(
            "Please make sure spam.csv is in the same "
            "folder as app.py."
        )


    else:

        cleaned_message = clean_text(message)


        prediction = model.predict(
            [cleaned_message]
        )[0]


        decision_score = model.decision_function(
            [cleaned_message]
        )[0]


        # Approximate confidence
        confidence = (
            1 /
            (
                1 +
                math.exp(
                    -abs(float(decision_score))
                )
            )
        ) * 100


        # ----------------------------------------------------
        # SPAM
        # ----------------------------------------------------

        if prediction == 1:

            with st.container(border=True):

                st.markdown(
                    '<div style="text-align:center;'
                    'font-size:50px;">🚨</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    '<div class="result-title">'
                    'SPAM MESSAGE'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    '<div class="result-description">'
                    'This message has been classified as '
                    'potentially fraudulent or unwanted.'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="confidence">'
                    f'Detection Confidence: '
                    f'{confidence:.1f}%'
                    f'</div>',
                    unsafe_allow_html=True
                )


                st.progress(
                    min(
                        max(confidence / 100, 0),
                        1
                    )
                )


                st.warning(
                    "Be cautious with messages requesting "
                    "money, personal information, prizes, "
                    "or urgent action."
                )


        # ----------------------------------------------------
        # LEGITIMATE
        # ----------------------------------------------------

        else:

            with st.container(border=True):

                st.markdown(
                    '<div style="text-align:center;'
                    'font-size:50px;">✅</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    '<div class="result-title">'
                    'LEGITIMATE MESSAGE'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    '<div class="result-description">'
                    'This message appears to be a normal '
                    'or legitimate SMS.'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="confidence">'
                    f'Detection Confidence: '
                    f'{confidence:.1f}%'
                    f'</div>',
                    unsafe_allow_html=True
                )


                st.progress(
                    min(
                        max(confidence / 100, 0),
                        1
                    )
                )


# ============================================================
# HOW SMARTSMS WORKS
# ============================================================

st.markdown(
    '<div class="section-title">'
    'How SmartSMS Works'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Your message passes through three machine learning '
    'stages before the final classification.'
    '</div>',
    unsafe_allow_html=True
)


process1, process2, process3 = st.columns(3)


# ============================================================
# PROCESS 1
# ============================================================

with process1:

    with st.container(border=True):

        st.markdown(
            '<div class="process-number">01</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="process-title">'
            'Text Processing'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="process-text">'
            'The SMS message is cleaned and prepared '
            'for machine learning analysis. URLs, '
            'numbers and unnecessary characters are '
            'normalized before feature extraction.'
            '</div>',
            unsafe_allow_html=True
        )


# ============================================================
# PROCESS 2
# ============================================================

with process2:

    with st.container(border=True):

        st.markdown(
            '<div class="process-number">02</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="process-title">'
            'TF-IDF'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="process-text">'
            'The processed text is converted into '
            'numerical features using TF-IDF. '
            'Important words and word combinations '
            'receive meaningful feature weights.'
            '</div>',
            unsafe_allow_html=True
        )


# ============================================================
# PROCESS 3
# ============================================================

with process3:

    with st.container(border=True):

        st.markdown(
            '<div class="process-number">03</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="process-title">'
            'Classification'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="process-text">'
            'A Linear SVM model analyzes the extracted '
            'features and classifies the message as '
            'spam or legitimate.'
            '</div>',
            unsafe_allow_html=True
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.write("")

with st.expander("🧠  Model Information"):

    st.subheader("Machine Learning Pipeline")

    st.write(
        "🧹 Natural Language Processing (NLP)"
    )

    st.write(
        "📊 TF-IDF Feature Extraction"
    )

    st.write(
        "🧠 Linear Support Vector Machine (SVM)"
    )

    st.write(
        "📱 SMS Spam / Legitimate Classification"
    )


    if model is not None:

        st.success(
            f"Model trained successfully using "
            f"{dataset_size:,} SMS messages."
        )

        if dataset_used:

            st.caption(
                f"Dataset: {dataset_used}"
            )

    else:

        st.warning(
            "Model is currently unavailable. "
            "Add spam.csv to enable classification."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="footer-title">'
    '🛡️ SmartSMS · Machine Learning SMS Spam Classifier'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="footer-line">'
    'Built with Python · Scikit-learn · Streamlit'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="footer-tech">'
    'NLP · TF-IDF · Linear SVM'
    '</div>',
    unsafe_allow_html=True
)