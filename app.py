import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sqlalchemy import create_engine, text
import warnings
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pickle
import io
import base64
from plotly.subplots import make_subplots
from wordcloud import WordCloud
from concurrent.futures import ThreadPoolExecutor
import logging
from dotenv import load_dotenv
from datetime import datetime, timedelta
import ast
from werkzeug.security import generate_password_hash, check_password_hash
from ultralytics import YOLO
from dotenv import load_dotenv
import io
from flask import send_file
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import spacy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from PyPDF2 import PdfReader
import re
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import ssl
from waitress import serve
from sklearn.exceptions import InconsistentVersionWarning



warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



MODEL_FILE = 'churn_model.pkl'


# Initialize ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=4)

# Database connection
engine = create_engine('postgresql://postgres:kaniniintern@localhost:5432/postgres')

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d')
        return super(NumpyEncoder, self).default(obj)



#------------------------------------------------------------------------------------------------------------------
# Login/signup form 
@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({"success": False, "message": "Email and password are required"}), 400

    try:
        # Check if user already exists
        query = text("SELECT * FROM users WHERE email = :email")
        with engine.connect() as connection:
            result = connection.execute(query, {"email": email}).fetchone()

        if result:
            return jsonify({"success": False, "message": "User already exists"}), 400

        # Hash the password
        hashed_password = generate_password_hash(password)

        # Insert new user
        insert_query = text("INSERT INTO users (email, password) VALUES (:email, :password)")
        with engine.connect() as connection:
            connection.execute(insert_query, {"email": email, "password": hashed_password})
            connection.commit()

        return jsonify({"success": True, "message": "User created successfully"}), 201

    except Exception as e:
        logger.error(f"Error in signup: {str(e)}")
        return jsonify({"success": False, "message": "An error occurred"}), 500

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({"success": False, "message": "Email and password are required"}), 400

    try:
        # Check if user exists
        query = text("SELECT * FROM users WHERE email = :email")
        with engine.connect() as connection:
            result = connection.execute(query, {"email": email}).fetchone()

        if not result:
            return jsonify({"success": False, "message": "User not found"}), 404

        # Access the password using integer index
        # Assuming the password is the third column (index 2) in the result tuple
        hashed_password = result[2]

        # Check password
        if check_password_hash(hashed_password, password):
            return jsonify({"success": True, "message": "Login successful"}), 200
        else:
            return jsonify({"success": False, "message": "Incorrect password"}), 401

    except Exception as e:
        logger.error(f"Error in login: {str(e)}")
        return jsonify({"success": False, "message": "An error occurred"}), 500
    

#------------------------------------------------------------------------------------------------------------------  
# Data retrieving
def load_and_clean_data():
    query = text('SELECT * FROM laster')
    with engine.connect() as connection:
        df = pd.read_sql_query(query, connection)
    logger.info("Data successfully loaded from PostgreSQL.")
    return df

    # df=pd.read_csv('lastestdataset.csv')        # data from csv
    # return df
    


# -------------------------------------------------------------------------------------------------------------------
# eda insight views
def create_visualization(fig, title, finding):
    fig.update_layout(title=title)
    return {"figure": json.loads(fig.to_json()), "finding": finding}


@app.route('/api/eda', methods=['GET'])
def get_eda_data():
    try:
        cleaned_df = load_and_clean_data()

        if cleaned_df['CHURN'].dtype == 'object':
            cleaned_df['CHURN'] = cleaned_df['CHURN'].map({'No': 0, 'Yes': 1})
        cleaned_df['CHURN'] = pd.to_numeric(cleaned_df['CHURN'], errors='coerce')
        cleaned_df = cleaned_df.dropna(subset=['CHURN'])

        eda_data = {}

        # Churn Rate
        churn_rate = cleaned_df['CHURN'].mean()
        churn_df = pd.DataFrame({'Status': ['Churned', 'Retained'], 'Value': [churn_rate, 1 - churn_rate]})
        fig = px.pie(churn_df, values='Value', names='Status', title='Churn Rate', 
                    color='Status', color_discrete_map={'Churned': 'red', 'Retained': 'yellow'})
        eda_data['churn_rate'] = create_visualization(fig, 'Churn Rate', f"The pie chart shows the percentage of customers who have churned. The churn rate is {churn_rate:.2%}.")

        # 2. Relationship between CHURN and GWP
        fig = px.box(cleaned_df, x='CHURN', y='GWP', 
                    color='CHURN', color_discrete_map={0: 'blue', 1: 'red'},
                    labels={'CHURN': 'Churned', '0': 'Not Churned', '1': 'Churned'},
                    title='Churn and GWP')
        fig.update_yaxes(range=[cleaned_df['GWP'].quantile(0.01), cleaned_df['GWP'].quantile(0.99)])
        eda_data['churn_gwp'] = create_visualization(fig, 'Churn and GWP', "This histogram shows the distribution of Gross Written Premium (GWP) among customers who have churned versus those who have not. It can help identify if there's a relationship between GWP and churn.")

        # 3. Relationship between CHURN and NUMBER_OF_CLAIM
        fig = px.box(cleaned_df, x='CHURN', y='NUMBER_OF_CLAIM', 
                    color='CHURN', color_discrete_map={0: 'blue', 1: 'red'},
                    labels={'CHURN': 'Churned', '0': 'Not Churned', '1': 'Churned'},
                    title='Churn and Number of claim')
        fig.update_yaxes(range=[0, cleaned_df['NUMBER_OF_CLAIM'].quantile(0.99)])
        eda_data['churn_claim'] = create_visualization(fig, 'Churn and Number of claim',  "This histogram shows the distribution of churn across different age groups. It can help identify if certain age groups are more prone to churning.")

        # 4. Distribution of CHURN across different Age groups
        age_bins = pd.cut(cleaned_df['Age'], bins=[0, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        churn_by_age = cleaned_df.groupby(age_bins, observed=True)['CHURN'].mean().reset_index()
        churn_by_age['Age'] = churn_by_age['Age'].astype(str)
        fig = px.bar(churn_by_age, x='Age', y='CHURN', 
                    title='Churn and Age groups',
                    labels={'CHURN': 'Churn Rate', 'Age': 'Age Group'})
        eda_data['churn_age'] = create_visualization(fig, 'Churn and Age groups', "This visualization shows how the number of claims correlates with the churn rate. It can help identify if customers with more claims are more likely to churn.")

        # 5. Churn Rate by PAYMENT_MODE
        churn_by_payment = cleaned_df.groupby('PAYMENT_MODE', observed=True)['CHURN'].mean().reset_index()
        fig = px.bar(churn_by_payment, x='PAYMENT_MODE', y='CHURN', 
                    title='Churn Rate by Payment mode', color='PAYMENT_MODE')
        eda_data['churn_payment_mode'] = create_visualization(fig, 'Churn Rate by Payment mode', "This histogram shows how different payment modes affect the churn rate. It can help identify if certain payment modes are associated with higher churn rates.")
        
        # 6. Distribution of Churn Rate by PAYMENT_MODE
        churn_by_payment = cleaned_df.groupby('PAYMENT_MODE', observed=True)['CHURN'].mean().reset_index()
        fig = px.bar(churn_by_payment, x='PAYMENT_MODE', y='CHURN', 
                    title='Distribution of Churn Rate by Payment mode', color='PAYMENT_MODE')
        eda_data['churn_payment_mode_dist'] = create_visualization(fig, 'Distribution of Churn Rate by Payment mode', "This histogram displays the distribution of churn across various payment modes. It helps identify if certain payment methods are more prone to churning, which can guide targeted retention strategies.")


        # 7. Churn Rate by LOB
        churn_by_lob = cleaned_df.groupby('LOB', observed=True)['CHURN'].mean().reset_index()
        fig = px.bar(churn_by_lob, x='LOB', y='CHURN', 
                    title='Churn Rate by LOB', color='LOB')
        eda_data['churn_lob'] = create_visualization(fig, 'Churn Rate by LOB', "This bar chart illustrates how different lines of business affect the churn rate. It can reveal if certain LOBs are associated with higher customer retention or loss, potentially informing decisions on product offerings.")

        # 8. Customer Loyalty Score Distribution
        fig = px.histogram(cleaned_df, x='Loyalty_Score', title='Customer Loyalty Score Distribution', 
                        nbins=50, color='CHURN', color_discrete_map={0: 'blue', 1: 'red'},
                        marginal='box')
        fig.update_xaxes(range=[0, 100])
        fig.update_yaxes(title_text='Count')
        eda_data['loyalty_score_dist'] = create_visualization(fig, 'Customer Loyalty Score Distribution', 
                                "Histogram showing the distribution of customer loyalty scores. Loyalty scores differ between churned and retained customers.")

        # 9. Churn Rate by Gender
        churn_by_gender = cleaned_df.groupby('GENDER', observed=True)['CHURN'].mean().reset_index()
        fig = px.bar(churn_by_gender, x='GENDER', y='CHURN', 
                    title='Churn Rate by Gender', color='GENDER')
        eda_data['churn_gender'] = create_visualization(fig, 'Churn Rate by Gender', 
                                    "This chart shows how churn rates differ between genders, which can inform gender-specific retention strategies.")

        # 10. Churn Rate by Marital Status
        churn_by_marital = cleaned_df.groupby('MARITAL_STATUS', observed=True)['CHURN'].mean().reset_index()
        fig = px.bar(churn_by_marital, x='MARITAL_STATUS', y='CHURN', 
                    title='Churn Rate by Marital Status', color='MARITAL_STATUS')
        eda_data['churn_marital'] = create_visualization(fig, 'Churn Rate by Marital Status', 
                                    "This visualization presents churn rates for different marital statuses, potentially revealing patterns in customer retention based on marital status.")

        # 11. Churn Rate by Education Level
        churn_by_education = cleaned_df.groupby('Education_Level', observed=True)['CHURN'].mean().reset_index()
        fig = px.bar(churn_by_education, x='Education_Level', y='CHURN', 
                    title='Churn Rate by Education Level', color='Education_Level')
        eda_data['churn_education'] = create_visualization(fig, 'Churn Rate by Education Level', 
                                    "This chart presents churn rates for different education levels, which can guide targeted retention strategies based on educational background.")

        # 12. Churn Rate by Customer Satisfaction Score
        churn_by_satisfaction = cleaned_df.groupby('Customer_Satisfaction_Score', observed=True)['CHURN'].mean().reset_index()
        fig = px.bar(churn_by_satisfaction, x='Customer_Satisfaction_Score', y='CHURN', 
                    title='Churn Rate by Customer_Satisfaction_Score', color='Customer_Satisfaction_Score')
        eda_data['churn_satisfaction'] = create_visualization(fig, 'Churn Rate by Customer Satisfaction Score', 
                                    "This chart presents how churn rates vary with customer satisfaction scores, highlighting the importance of customer satisfaction in retention.")

        # 13. Churn Rate by Region
        churn_by_region = cleaned_df.groupby('Region', observed=True)['CHURN'].mean().reset_index()
        fig = px.bar(churn_by_region, x='Region', y='CHURN', 
                    title='Churn Rate by Region', color='Region')
        eda_data['churn_region'] = create_visualization(fig, 'Churn Rate by Region', 
                                    "This visualization shows churn rates across different regions, potentially revealing geographical patterns in customer retention.")

        # 14. Churn Rate by Number_Of_Dependants
        churn_by_dependants = cleaned_df.groupby('Number_of_Dependants', observed=True)['CHURN'].mean().reset_index()
        fig = px.bar(churn_by_dependants, x='Number_of_Dependants', y='CHURN', 
                    title='Churn Rate by Number Of Dependants', color='Number_of_Dependants')
        eda_data['churn_dependants'] = create_visualization(fig, 'Churn Rate by Number Of Dependants', 
                                     "This visualization shows how the number of dependants affects churn rate, potentially revealing the impact of family size on customer retention.")
        

        return jsonify(eda_data)

    except Exception as e:
        logger.error(f"Error in get_eda_data: {str(e)}")
        return jsonify({"error": str(e)}), 500



#-------------------------------------------------------------------------------------------------------------------
# sentiment analysis
def load_and_preprocess_data():
    try:
        engine = create_engine('postgresql://postgres:kaniniintern@localhost:5432/postgres')
        df = pd.read_sql('SELECT * FROM laster', engine)
        logger.info("Data successfully loaded from PostgreSQL.")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()  # Return an empty DataFrame if there's an error

def plot_sentiment_distribution(df):
    if 'Feedback_Sentiment' not in df.columns:
        logger.warning("Feedback_Sentiment column not found in DataFrame")
        return json.dumps({})
    
    fig = px.histogram(df, x='Feedback_Sentiment', nbins=20,
                       range_x=[0, 10],
                       title='Overall Sentiment Distribution',
                       labels={'Feedback_Sentiment': 'Sentiment Score', 'count': 'Count'},
                       color_discrete_sequence=['#636EFA'])
    fig.update_layout(bargap=0.1)
    return fig.to_json()

def plot_sentiment_trend(df):
    if 'APPROVAL_DATE' not in df.columns or 'Feedback_Sentiment' not in df.columns:
        logger.warning("APPROVAL_DATE or Feedback_Sentiment column not found in DataFrame")
        return json.dumps({})
    
    try:
        df['APPROVAL_DATE'] = pd.to_datetime(df['APPROVAL_DATE'])
        sentiment_trend = df.groupby(df['APPROVAL_DATE'].dt.to_period('M'))['Feedback_Sentiment'].mean().reset_index()
        sentiment_trend['APPROVAL_DATE'] = sentiment_trend['APPROVAL_DATE'].dt.to_timestamp()
        
        fig = px.line(sentiment_trend, x='APPROVAL_DATE', y='Feedback_Sentiment',
                      title='Average Sentiment Score Trend',
                      labels={'APPROVAL_DATE': 'Date', 'Feedback_Sentiment': 'Average Sentiment Score'},
                      range_y=[0, 10])
        return fig.to_json()
    except Exception as e:
        logger.error(f"Error in plot_sentiment_trend: {str(e)}")
        fig = go.Figure()
        fig.add_annotation(text="Error occurred in sentiment trend", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig.to_json()

def plot_top_themes_comparison(df):
    if df.empty:
        logger.warning("DataFrame is empty")
        return json.dumps({})

    if 'Feedback_Theme' not in df.columns or 'Feedback_Sentiment' not in df.columns:
        logger.warning("Feedback_Theme or Feedback_Sentiment column not found in DataFrame")
        return json.dumps({})
    
    try:
        # Ensure Feedback_Sentiment is numeric
        df['Feedback_Sentiment'] = pd.to_numeric(df['Feedback_Sentiment'], errors='coerce')
        
        positive_themes = df[df['Feedback_Sentiment'] > 5]['Feedback_Theme'].value_counts()
        negative_themes = df[df['Feedback_Sentiment'] <= 5]['Feedback_Theme'].value_counts()

        fig = make_subplots(rows=1, cols=2, subplot_titles=('Top 10 Positive Feedback Themes', 
                                                            'Top 10 Negative Feedback Themes'))

        if not positive_themes.empty:
            positive_themes = positive_themes.head(10)
            fig.add_trace(
                go.Bar(y=positive_themes.index, x=positive_themes.values, orientation='h', name='Positive', marker_color='green'),
                row=1, col=1
            )
        else:
            fig.add_annotation(text="No positive themes found", xref="x1", yref="y1", x=0.5, y=0.5, showarrow=False)

        if not negative_themes.empty:
            negative_themes = negative_themes.head(10)
            fig.add_trace(
                go.Bar(y=negative_themes.index, x=negative_themes.values, orientation='h', name='Negative', marker_color='red'),
                row=1, col=2
            )
        else:
            fig.add_annotation(text="No negative themes found", xref="x2", yref="y2", x=0.5, y=0.5, showarrow=False)

        fig.update_layout(height=500, width=1000, title_text="Top Feedback Themes", showlegend=False)
        fig.update_xaxes(title_text="Count")
        return fig.to_json()
    except Exception as e:
        logger.error(f"Error in plot_top_themes_comparison: {str(e)}")
        # Return an empty plot in case of any error
        fig = go.Figure()
        fig.add_annotation(text="Error occurred", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig.to_json()

def generate_word_cloud(df):
    if 'Customer_Feedback' not in df.columns:
        logger.warning("Customer_Feedback column not found in DataFrame")
        return ""
    
    text = ' '.join(df['Customer_Feedback'].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    img = wordcloud.to_image()
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    return base64.b64encode(img_byte_arr).decode()

def plot_sentiment_by_age_group(df):
    if 'Age' not in df.columns or 'Feedback_Sentiment' not in df.columns:
        logger.warning("Age or Feedback_Sentiment column not found in DataFrame")
        return json.dumps({})
    
    age_bins = [0, 18, 30, 40, 50, 60, 70, 80, 100]
    age_labels = ['<18', '18-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80+']
    
    df['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)
    
    fig = px.box(df, x='Age_Group', y='Feedback_Sentiment',
                 title='Sentiment Score Distribution by Age Group',
                 labels={'Age_Group': 'Age Group', 'Feedback_Sentiment': 'Sentiment Score'})
    
    return fig.to_json()

def plot_churn_insights(df):
    if 'CHURN' not in df.columns or 'Feedback_Sentiment' not in df.columns:
        logger.warning("CHURN or Feedback_Sentiment column not found in DataFrame")
        return json.dumps({})
    
    try:
        churn_sentiments = df.groupby('CHURN')['Feedback_Sentiment'].mean().reset_index()
        
        fig = px.bar(churn_sentiments, x='CHURN', y='Feedback_Sentiment',
                     title="Average Sentiments Scores by Churn Status",
                     labels={"CHURN": "Churn Status", "Feedback_Sentiment": "Average Sentiments Scores"},
                     color="Feedback_Sentiment", color_continuous_scale=["blue", "red"])
        
        return fig.to_json()
    except Exception as e:
        logger.error(f"Error in plot_churn_insights: {str(e)}")
        fig = go.Figure()
        fig.add_annotation(text="Error occurred in churn insights", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig.to_json()

@app.route('/api/visualizations', methods=['GET'])
def get_visualizations():
    df = load_and_clean_data()
    
    if df.empty:
        return jsonify({"error": "Failed to load data"}), 500
    
    # Use ThreadPoolExecutor to run tasks concurrently
    with ThreadPoolExecutor() as executor:
        futures = {
            'sentiment_distribution': executor.submit(plot_sentiment_distribution, df),
            'sentiment_trend': executor.submit(plot_sentiment_trend, df),
            'top_themes_comparison': executor.submit(plot_top_themes_comparison, df),
            'word_cloud': executor.submit(generate_word_cloud, df),
            'sentiment_by_age_group': executor.submit(plot_sentiment_by_age_group, df),
            'churn_insights': executor.submit(plot_churn_insights, df)
        }
        
        results = {}
        for key, future in futures.items():
            try:
                results[key] = future.result()
            except Exception as e:
                logger.error(f"Error in {key}: {str(e)}")
                results[key] = json.dumps({})  # Return empty JSON for failed visualizations

    return jsonify(results)

    
#-------------------------------------------------------------------------------------------------------------------
# customer segmentation
def perform_rfm_analysis():
    data = load_and_clean_data()

    rfm_data = data[['CPR_NO', 'Date_of_Purchase', 'Premium_Amount']].copy()
    rfm_data['Date_of_Purchase'] = pd.to_datetime(rfm_data['Date_of_Purchase'], format='%Y-%m-%d')
    current_date = rfm_data['Date_of_Purchase'].max() + pd.Timedelta(days=1)

    rfm = rfm_data.groupby('CPR_NO').agg({
        'Date_of_Purchase': lambda x: (current_date - x.max()).days,
        'CPR_NO': 'count',
        'Premium_Amount': 'sum'
    }).rename(columns={
        'Date_of_Purchase': 'Recency',
        'CPR_NO': 'Frequency',
        'Premium_Amount': 'Monetary'
    }).reset_index()

    rfm['R_Score'] = pd.qcut(rfm['Recency'], q=5, labels=[5, 4, 3, 2, 1])
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=5, labels=[1, 2, 3, 4, 5])

    rfm['RFM_Segment'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

    def rfm_label(row):
        if row['R_Score'] >= 4 and row['F_Score'] >= 4 and row['M_Score'] >= 4:
            return 'Champions'
        elif row['R_Score'] >= 3 and row['F_Score'] >= 3 and row['M_Score'] >= 3:
            return 'Loyal Customers'
        elif row['R_Score'] >= 2 and row['F_Score'] >= 2 and row['M_Score'] >= 2:
            return 'Potential Loyalists'
        elif row['R_Score'] <= 2 and row['F_Score'] <= 2 and row['M_Score'] <= 2:
            return 'At Risk'
        else:
            return 'Others'

    rfm['RFM_Label'] = rfm.apply(rfm_label, axis=1)

    segment_counts = rfm['RFM_Label'].value_counts().reset_index()
    segment_counts.columns = ['RFM_Label', 'Count']
    
    pie_data = {
        'labels': segment_counts['RFM_Label'].tolist(),
        'values': segment_counts['Count'].tolist()
    }

    all_rfm_rows = rfm.to_dict(orient='records')

    return {
        'pie_data': pie_data,
        'all_rfm_rows': all_rfm_rows
    }

@app.route('/api/rfm-analysis', methods=['GET'])
def get_rfm_analysis():
    try:
        rfm_results = perform_rfm_analysis()
        return jsonify(rfm_results)
    except Exception as e:
        logger.error(f"Error in get_rfm_analysis: {str(e)}")
        return jsonify({"error": str(e)}), 500
    

#-------------------------------------------------------------------------------------------------------------------
# churn prediction
def train_and_save_model():
    df = load_and_clean_data()
    df['CHURN'] = df['CHURN'].map({'No': 0, 'Yes': 1})

    categorical = ['LOB', 'PRODUCT_PLAN', 'GENDER', 'MARITAL_STATUS', 'PAYMENT_TERM', 'Occupation', 'Education_Level']
    numerical = ['Age', 'Income', 'Credit_Score']

    X = df[categorical + numerical]
    y = df['CHURN']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=1)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical),
            ('cat', OneHotEncoder(sparse=False, handle_unknown='ignore'), categorical)
        ])

    best_params = {
        'n_estimators': 100,
        'min_samples_split': 5,
        'min_samples_leaf': 1,
        'max_features': 'sqrt',
        'max_depth': 20,
        'bootstrap': False,
        'class_weight': 'balanced'
    }

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', ExtraTreesClassifier(random_state=1, **best_params))
    ])

    pipeline.fit(X_train, y_train)

    with open(MODEL_FILE, 'wb') as file:
        pickle.dump(pipeline, file)

    print("Model trained and saved successfully.")

    return pipeline, X_test, y_test

def load_model():
    if not os.path.exists(MODEL_FILE):
        print("Model file not found. Training a new model...")
        return train_and_save_model()[0]
    
    with open(MODEL_FILE, 'rb') as file:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            return pickle.load(file)

def predict_churn(input_data):
    model = load_model()
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    return "Yes" if prediction == 1 else "No", probability



@app.route('/api/predict', methods=['POST'])
def predict():
    input_data = request.json
    prediction, probability = predict_churn(input_data)
    
    return jsonify({
        'prediction': prediction,
        'probability': probability,
    })

@app.route('/api/train', methods=['POST'])
def train():
    train_and_save_model()
    return jsonify({'message': 'Model trained successfully'})

#-------------------------------------------------------------------------------------------------------------------
# personalised dashboard

engine = create_engine('postgresql://postgres:kaniniintern@localhost:5432/postgres')

def get_lob_specific_data(lob, cpr_no):
    try:
        query = text("""
        SELECT * FROM laster 
        WHERE "LOB" = :lob AND "CPR_NO" = :cpr_no
        """)
        
        with engine.connect() as connection:
            df = pd.read_sql_query(query, connection, params={"lob": lob, "cpr_no": cpr_no})
        
        if df.empty:
            return {}
        
        lob_specific_data = {}
        
        if lob == "Auto":
            relevant_columns = ['Vehicle_Make', 'Vehicle_Model', 'Vehicle_Year', 'Mileage', 'Number_of_Accidents', 'Number_of_Traffic_Violations', 'Safety_Features', 'Anti_Theft_Device', 'Parking_Location', 'Annual_Mileage', 'Usage_Type']
        elif lob == "Property":
            relevant_columns = ['Property_Type', 'Property_Age', 'Property_Value', 'Square_Footage', 'Number_of_Floors', 'Construction_Type', 'Roof_Type', 'Security_System', 'Fire_Protection_System', 'Flood_Zone', 'Previous_Claims']
        elif lob == "Health":
            relevant_columns = ['BMI', 'Blood_Pressure', 'Cholesterol_Level', 'Smoker', 'Family_Medical_History', 'Pre_existing_Conditions', 'Prescription_Drugs', 'Last_Physical_Exam_Date', 'Preferred_Hospital', 'Dental_Coverage', 'Vision_Coverage']
        elif lob == "Travel":
            relevant_columns = ['Travel_Destination', 'Trip_Duration', 'Trip_Cost', 'Travel_Purpose', 'Number_of_Travelers', 'Adventure_Sports_Coverage', 'Pre_existing_Medical_Condition_Coverage', 'Trip_Cancellation_Coverage', 'Emergency_Medical_Evacuation_Coverage', 'Baggage_Loss_Coverage']
        
        for col in relevant_columns:
            if col in df.columns:
                lob_specific_data[col] = df[col].iloc[0]
        
        # Generate mock historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=150)
        date_range = pd.date_range(start=start_date, end=end_date, freq='ME')
        
        premium_trend = [
            {
                'date': date.strftime('%Y-%m-%d'),
                'value': np.random.randint(900, 1200)
            } for date in date_range
        ]
        
        claim_history = [
            {
                'date': date.strftime('%Y-%m-%d'),
                'value': np.random.randint(0, 3)
            } for date in date_range
        ]
        
        lob_specific_data['historical_data'] = {
            'premium_trend': premium_trend,
            'claim_history': claim_history
        }
        
        return lob_specific_data
    
    except Exception as e:
        logger.error(f"Error in get_lob_specific_data: {str(e)}")
        return {}

def get_policy_details(customer_data):
    return {
        "policy_number": customer_data['POLICY_NO'],
        "product_plan": customer_data['PRODUCT_PLAN'],
        "coverage_amount": float(customer_data['Coverage_Amount']),
        "premium_amount": float(customer_data['Premium_Amount']),
        "policy_expiration_date": customer_data['Policy_Expiration_Date'],
        "policy_renewal_status": customer_data['Policy_Renewal_Status']
    }

def get_claims_history(customer_data):
    # Simulated claims history data
    claims = [
        {"date": "2023-01-15", "amount": 500, "status": "Settled"},
        {"date": "2023-05-22", "amount": 1200, "status": "In Progress"},
        {"date": "2023-09-10", "amount": 800, "status": "Settled"}
    ]
    return {
        "total_claims": int(customer_data['NUMBER_OF_CLAIM']),
        "claim_frequency": float(customer_data['Claim_Frequency']),
        "recent_claims": claims
    }

def get_loyalty_program(customer_data):
    return {
        "loyalty_tier": customer_data['Loyalty_Tier'],
        "points_earned": int(customer_data['Points_Earned']),
        "rewards_redeemed": int(customer_data['Rewards_Redeemed']),
        "leaderboard_position": int(customer_data['Leaderboard_Position']),
        "benefits_available": customer_data['Benefits_Available']
    }

def get_risk_profile(customer_data):
    return {
        "risk_assessment_score": float(customer_data['Risk_Assessment_Score']),
        "credit_score": int(customer_data['Credit_Score']),
        "claim_history_impact": calculate_claim_history_impact(customer_data),
        "risk_factors": identify_risk_factors(customer_data)
    }

def calculate_claim_history_impact(customer_data):
    # Simulated calculation
    claim_frequency = float(customer_data['Claim_Frequency'])
    if claim_frequency < 0.5:
        return "Low"
    elif claim_frequency < 1.5:
        return "Medium"
    else:
        return "High"

def identify_risk_factors(customer_data):
    # Simulated risk factors
    factors = []
    if int(customer_data['Credit_Score']) < 650:
        factors.append("Low Credit Score")
    if float(customer_data['Claim_Frequency']) > 1.5:
        factors.append("High Claim Frequency")
    if float(customer_data['Outstanding_Premium']) > 0:
        factors.append("Outstanding Premium")
    return factors

def get_financial_overview(customer_data):
    return {
        "customer_lifetime_value": float(customer_data['Customer_Lifetime_Value']),
        "total_premiums_paid": calculate_total_premiums(customer_data),
        "outstanding_premium": float(customer_data['Outstanding_Premium']),
        "account_balance": float(customer_data['Account_Balance'])
    }

def calculate_total_premiums(customer_data):
    # Simulated calculation
    years_as_customer = (datetime.now() - pd.to_datetime(customer_data['Customer_Since'])).days / 365.25
    return years_as_customer * float(customer_data['Premium_Amount'])

def get_personalized_recommendations(customer_data):
    recommendations = []
    customer_name = customer_data['CUSTOMER_NAME']
    current_plan = customer_data['PRODUCT_PLAN']

    # Create a stylish table for plan recommendations
    table_data = {
        'headers': ['Current Plan', 'Plan to Recommend'],
        'rows': []
    }

    if current_plan == 'Basic':
        table_data['rows'].append(['Basic', 'Silver'])
        recommendations.append(f"Upgrade {customer_name}'s plan from Basic to Silver for enhanced coverage.")
    elif current_plan == 'Silver':
        table_data['rows'].append(['Silver', 'Gold'])
        recommendations.append(f"Suggest {customer_name} to upgrade from Silver to Gold plan for premium benefits.")
    elif current_plan == 'Gold':
        table_data['rows'].append(['Gold', 'Platinum'])
        recommendations.append(f"Offer {customer_name} our top-tier Platinum plan, upgrading from Gold for exclusive perks.")
    elif current_plan == 'Platinum':
        table_data['rows'].append(['Platinum', 'Platinum'])
        recommendations.append(f"Maintain {customer_name}'s Platinum status. Consider offering additional services or loyalty rewards.")
    elif current_plan == 'Third Party':
        table_data['rows'].append(['Third Party', 'Platinum'])
        recommendations.append(f"Recommend {customer_name} to switch from Third Party to our comprehensive Platinum plan.")

    # Additional recommendations based on other factors
    if float(customer_data['Engagement_Score']) <= 50:
        recommendations.append(f"Increase engagement with {customer_name} through personalized communications and offers.")
    elif float(customer_data['Engagement_Score']) > 50:
        recommendations.append(f"Leverage {customer_name}'s high engagement with targeted cross-selling opportunities.")
    if float(customer_data['Churn_Risk_Score']) > 5:
        recommendations.append(f"Implement retention strategies for {customer_name} due to high churn risk.")
    elif float(customer_data['Churn_Risk_Score']) <= 5:
        recommendations.append(f"Reinforce {customer_name}'s loyalty with exclusive benefits and personalized service.")
    if float(customer_data['Customer_Satisfaction_Score']) < 6:
        recommendations.append(f"Address {customer_name}'s concerns and improve satisfaction through dedicated support and tailored solutions.")
    
    return {
        'table': table_data,
        'recommendations': recommendations
    }

def generate_pdf_report(customer_data):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    heading2_style = styles['Heading2']
    body_style = styles['BodyText']

    elements = []

    elements.append(Paragraph("K-Insure Customer View", title_style))
    elements.append(Spacer(1, 0.5*inch))

    def format_value(value):
        if isinstance(value, (list, dict)):
            return format_complex_value(value)
        return str(value)

    def format_complex_value(value, indent=0):
        if isinstance(value, list):
            return '\n'.join([f"{'  ' * indent}- {format_complex_value(item, indent+1)}" for item in value])
        elif isinstance(value, dict):
            return '\n'.join([f"{'  ' * indent}{k}: {format_complex_value(v, indent+1)}" for k, v in value.items()])
        else:
            return str(value)

    for section, data in customer_data.items():
        elements.append(Paragraph(section.replace('_', ' ').title(), heading2_style))
        if isinstance(data, dict):
            for key, value in data.items():
                elements.append(Paragraph(f"{key.replace('_', ' ').title()}:", body_style))
                elements.append(Paragraph(format_value(value), body_style))
                elements.append(Spacer(1, 0.1*inch))
        else:
            elements.append(Paragraph(str(data), body_style))
        elements.append(Spacer(1, 0.25*inch))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# Routes
@app.route('/api/generate-pdf-data', methods=['POST'])
def generate_pdf_data():
    data = request.json
    lob = data['lob']
    cpr_no = data['cpr_no']

    try:
        customer_data = get_customer_dashboard(lob, cpr_no)
        if isinstance(customer_data, str):
            customer_data = json.loads(customer_data)
        pdf_buffer = generate_pdf_report(customer_data)
        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name=f'customer_dashboard_{cpr_no}.pdf',
            mimetype='application/pdf'
        )
    except Exception as e:
        logger.error(f"Error generating PDF: {str(e)}")
        return jsonify({"error": str(e)}), 500

    
@app.route('/api/customer-dashboard', methods=['POST'])
def get_customer_dashboard(lob=None, cpr_no=None):
    if request.method == 'POST':
        data = request.json
        lob = data.get('lob', lob)
        cpr_no = data.get('cpr_no', cpr_no)
    
    if not lob or not cpr_no:
        return jsonify({"error": "LOB and CPR_NO are required"}), 400
    
    try:
        query = text("""
        SELECT * FROM laster 
        WHERE "LOB" = :lob AND "CPR_NO" = :cpr_no
        """)
        
        with engine.connect() as connection:
            df = pd.read_sql_query(query, connection, params={"lob": lob, "cpr_no": cpr_no})
        
        if df.empty:
            logger.warning(f"No customer found for LOB: {lob} and CPR_NO: {cpr_no}")
            return jsonify({"error": "No customer found"}), 404
        
        customer_data = df.iloc[0].copy()
        
        # Parse dates with explicit format and dayfirst=True
        date_columns = ['APPROVAL_DATE', 'DATE_OF_BIRTH', 'Date_of_Purchase', 'Last_Interaction_Date', 'Last_Physical_Exam_Date', 'Customer_Since', 'Policy_Expiration_Date', 'Last_Policy_Change_Date']
        for col in date_columns:
            if col in customer_data:
                customer_data[col] = pd.to_datetime(customer_data[col], format='%d-%m-%Y', dayfirst=True, errors='coerce')
        
        # Parse sentiment score
        try:
            sentiment_dict = ast.literal_eval(customer_data['sentiment_score'])
            sentiment_score = sentiment_dict['compound']
        except:
            sentiment_score = 0  # Default value if parsing fails

        # Get interaction history
        interaction_count = customer_data['Interaction_Count']
        last_interaction_date = customer_data['Last_Interaction_Date']
        
        # Generate interaction history for the last 12 months
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        date_range = pd.date_range(start=start_date, end=end_date, freq='ME')
        
        interaction_history = []
        for date in date_range:
            if date <= last_interaction_date:
                interaction_history.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "interaction_count": int(interaction_count / len(date_range[date_range <= last_interaction_date]))
                })
            else:
                interaction_history.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "interaction_count": 0
                })
        
        # Generate engagement history for the last 12 months
        engagement_score = float(customer_data['Engagement_Score'])
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        date_range = pd.date_range(start=start_date, end=end_date, freq='ME')
        
        engagement_history = []
        for date in date_range:
            engagement_history.append({
                "date": date.strftime("%Y-%m-%d"),
                "engagement_score": float(engagement_score + np.random.uniform(-0.5, 0.5))  # Add some variation
            })

        dashboard_data = {
            "personal_info": {
                "CUSTOMER_NAME": customer_data['CUSTOMER_NAME'],
                "EMAIL": customer_data['EMAIL'],
                "GENDER": customer_data['GENDER'],
                "MARITAL_STATUS": customer_data['MARITAL_STATUS'],
                "Age": int(customer_data['Age']),
                "Occupation": customer_data['Occupation'],
                "Education_Level": customer_data['Education_Level']
            },
            "metrics": {
                "table_data": [
                    {"metric": "Churn Risk Score", "value": float(customer_data['Churn_Risk_Score'])},
                    {"metric": "Engagement Score", "value": float(customer_data['Engagement_Score'])},
                    {"metric": "Loyalty Score", "value": float(customer_data['Loyalty_Score'])}
                ],
                "chart_data": [
                    {"metric": "Customer Lifetime Value", "value": float(customer_data['Customer_Lifetime_Value'])},
                    {"metric": "GWP", "value": float(customer_data['GWP'])}
                ],
                "customer_name": customer_data['CUSTOMER_NAME'],
                "lob": lob
            },
            "lob_specific_data": get_lob_specific_data(lob, cpr_no),
            "interaction_history": interaction_history,
            "engagement_history": engagement_history,
            "risk_factors": {
                "Credit Score": int(customer_data['Credit_Score']),
                "Outstanding Premium": float(customer_data['Outstanding_Premium']),
                "Number of Claims": int(customer_data['NUMBER_OF_CLAIM']),
                "Risk Assessment Score": float(customer_data['Risk_Assessment_Score'])
            },
            "customer_feedback": {
                "Satisfaction Score": float(customer_data['Customer_Satisfaction_Score']),
                "Feedback Score": float(customer_data['Feedback_Score']),
                "Sentiment Score": float(sentiment_score)
            }
        }
        
        # Add new sections (as per the previous update)
        dashboard_data.update({
            "policy_details": get_policy_details(customer_data),
            "claims_history": get_claims_history(customer_data),
            "loyalty_program": get_loyalty_program(customer_data),
            "risk_profile": get_risk_profile(customer_data),
            "financial_overview": get_financial_overview(customer_data),
            "personalized_recommendations": get_personalized_recommendations(customer_data)
        })
        
        logger.info(f"Successfully retrieved dashboard data for LOB: {lob} and CPR_NO: {cpr_no}")
        return json.dumps(dashboard_data, cls=NumpyEncoder)
    
    except Exception as e:
        logger.error(f"Error in get_customer_dashboard: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/lob-specific', methods=['POST'])
def get_lob_specific():
    data = request.json
    lob = data['lob']
    cpr_no = data['cpr_no']
    
    try:
        lob_specific_data = get_lob_specific_data(lob, cpr_no)
        return jsonify(lob_specific_data)
    
    except Exception as e:
        logger.error(f"Error in get_lob_specific: {str(e)}")
        return jsonify({"error": str(e)}), 500

#-------------------------------------------------------------------------------------------------------------------
# geographical view

@app.route('/get_india_data')
def get_india_data():
    try:
        # Load the data from your database
        engine = create_engine('postgresql://postgres:kaniniintern@localhost:5432/postgres')
        query = text('SELECT * FROM laster')
        with engine.connect() as connection:
            df = pd.read_sql_query(query, connection)

        # Define accurate coordinates for Indian regions (using state coordinates)
        indian_regions_coordinates = {
            'Andhra Pradesh': [79.74, 15.91],
            'Arunachal Pradesh': [94.72, 28.21],
            'Assam': [92.93, 26.20],
            'Bihar': [85.31, 25.09],
            'Chhattisgarh': [81.86, 21.27],
            'Goa': [74.12, 15.29],
            'Gujarat': [71.19, 22.25],
            'Haryana': [76.08, 29.05],
            'Himachal Pradesh': [77.17, 31.10],
            'Jharkhand': [85.27, 23.61],
            'Karnataka': [75.71, 15.31],
            'Kerala': [76.27, 10.85],
            'Madhya Pradesh': [78.65, 22.97],
            'Maharashtra': [75.71, 19.75],
            'Manipur': [93.90, 24.66],
            'Meghalaya': [91.36, 25.46],
            'Mizoram': [92.93, 23.16],
            'Nagaland': [94.56, 26.15],
            'Odisha': [85.09, 20.95],
            'Punjab': [75.34, 31.14],
            'Rajasthan': [74.21, 27.02],
            'Sikkim': [88.51, 27.53],
            'Tamil Nadu': [78.65, 11.12],
            'Telangana': [79.01, 17.12],
            'Tripura': [91.98, 23.94],
            'Uttar Pradesh': [80.94, 26.84],
            'Uttarakhand': [79.01, 30.06],
            'West Bengal': [87.85, 22.98]
        }

        # Calculate region-specific data
        region_data = df.groupby('Region').agg({
            'CPR_NO': 'count',
            'Churn_Risk_Score': 'mean',
            'Customer_Lifetime_Value': 'mean',
            'PRODUCT_PLAN': lambda x: x.mode().iloc[0],
            'Claim_Frequency': 'mean',
            'Customer_Satisfaction_Score': 'mean',
            'Risk_Assessment_Score': 'mean',
            'Loyalty_Score': 'mean',
            'Age': 'mean',
            'Income': 'mean',
            'LOB': lambda x: x.mode().iloc[0],
            'Engagement_Score': 'mean',
            'Premium_Amount': 'mean',
            'Policy_Renewal_Status': lambda x: (x == 'Renewed').mean(),
        }).reset_index()

        # Prepare GeoJSON features
        features = []
        for _, row in region_data.iterrows():
            if row['Region'] in indian_regions_coordinates:
                feature = {
                    "type": "Feature",
                    "properties": row.to_dict(),
                    "geometry": {
                        "type": "Point",
                        "coordinates": indian_regions_coordinates[row['Region']]
                    }
                }
                features.append(feature)

        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        return jsonify(geojson)

    except Exception as e:
        logger.error(f"Error in get_india_data: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_region_churn_data', methods=['POST'])
def get_region_churn_data():
    region = request.form['region']
    try:
        engine = create_engine('postgresql://postgres:kaniniintern@localhost:5432/postgres')
        query = text('SELECT * FROM laster WHERE "Region" = :region')
        with engine.connect() as connection:
            df = pd.read_sql_query(query, connection, params={"region": region})

        region_info = df.agg({
            'CPR_NO': 'count',
            'Churn_Risk_Score': 'mean',
            'Customer_Lifetime_Value': 'mean',
            'PRODUCT_PLAN': lambda x: x.mode().iloc[0],
            'Claim_Frequency': 'mean',
            'Customer_Satisfaction_Score': 'mean',
            'Risk_Assessment_Score': 'mean',
            'Loyalty_Score': 'mean',
            'Age': 'mean',
            'Income': 'mean',
            'LOB': lambda x: x.mode().iloc[0],
            'Engagement_Score': 'mean',
            'Premium_Amount': 'mean',
            'Policy_Renewal_Status': lambda x: (x == 'Renewed').mean(),
        }).to_dict()

        region_info['Region'] = region

        return jsonify(region_info)

    except Exception as e:
        logger.error(f"Error in get_region_churn_data: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
#-------------------------------------------------------------------------------------------------------------------
# cost deduction for vehicle damage

# Add new car damage assessment functionality
class DetailedCostEstimator:
    def __init__(self):
        self.base_costs = {
            "bonnet damage": {
                "small": {"parts": 5000, "labor": 2000, "paint": 3000, "consumables": 500},
                "medium": {"parts": 10000, "labor": 4000, "paint": 6000, "consumables": 1000},
                "large": {"parts": 20000, "labor": 6000, "paint": 10000, "consumables": 1500}
            },
            "broken headlight": {
                "small": {"parts": 3000, "labor": 1000, "paint": 0, "consumables": 200},
                "medium": {"parts": 6000, "labor": 1500, "paint": 0, "consumables": 300},
                "large": {"parts": 12000, "labor": 2000, "paint": 0, "consumables": 500}
            },
            "broken taillight": {
                "small": {"parts": 2500, "labor": 1000, "paint": 0, "consumables": 200},
                "medium": {"parts": 5000, "labor": 1500, "paint": 0, "consumables": 300},
                "large": {"parts": 10000, "labor": 2000, "paint": 0, "consumables": 500}
            },
            "broken windshield": {
                "small": {"parts": 8000, "labor": 2000, "paint": 0, "consumables": 500},
                "medium": {"parts": 15000, "labor": 3000, "paint": 0, "consumables": 800},
                "large": {"parts": 25000, "labor": 4000, "paint": 0, "consumables": 1000}
            },
            "complete damage": {
                "large": {"parts": 300000, "labor": 100000, "paint": 50000, "consumables": 10000}
            },
            "door damage": {
                "small": {"parts": 6000, "labor": 2000, "paint": 3000, "consumables": 500},
                "medium": {"parts": 12000, "labor": 4000, "paint": 6000, "consumables": 1000},
                "large": {"parts": 25000, "labor": 7000, "paint": 10000, "consumables": 1500}
            },
            "fender dent": {
                "small": {"parts": 3000, "labor": 1500, "paint": 2000, "consumables": 300},
                "medium": {"parts": 6000, "labor": 3000, "paint": 4000, "consumables": 600},
                "large": {"parts": 12000, "labor": 5000, "paint": 7000, "consumables": 1000}
            },
            "front bumper damage": {
                "small": {"parts": 5000, "labor": 2000, "paint": 3000, "consumables": 500},
                "medium": {"parts": 10000, "labor": 4000, "paint": 6000, "consumables": 1000},
                "large": {"parts": 20000, "labor": 7000, "paint": 10000, "consumables": 1500}
            },
            "full frontal damage": {
                "large": {"parts": 150000, "labor": 50000, "paint": 40000, "consumables": 5000}
            },
            "full rear damage": {
                "large": {"parts": 120000, "labor": 40000, "paint": 35000, "consumables": 4000}
            },
            "minor dent": {
                "small": {"parts": 0, "labor": 1500, "paint": 1500, "consumables": 200},
                "medium": {"parts": 2000, "labor": 3000, "paint": 3000, "consumables": 400},
                "large": {"parts": 4000, "labor": 5000, "paint": 5000, "consumables": 600}
            },
            "mirror damage": {
                "small": {"parts": 2500, "labor": 1000, "paint": 1000, "consumables": 200},
                "medium": {"parts": 5000, "labor": 1500, "paint": 1500, "consumables": 300},
                "large": {"parts": 10000, "labor": 2000, "paint": 2000, "consumables": 500}
            },
            "rear bumper damage": {
                "small": {"parts": 5000, "labor": 2000, "paint": 3000, "consumables": 500},
                "medium": {"parts": 10000, "labor": 4000, "paint": 6000, "consumables": 1000},
                "large": {"parts": 20000, "labor": 7000, "paint": 10000, "consumables": 1500}
            },
            "rear windshield damage": {
                "small": {"parts": 7000, "labor": 2000, "paint": 0, "consumables": 500},
                "medium": {"parts": 12000, "labor": 3000, "paint": 0, "consumables": 800},
                "large": {"parts": 20000, "labor": 4000, "paint": 0, "consumables": 1000}
            },
            "scratches": {
                "small": {"parts": 0, "labor": 1000, "paint": 2000, "consumables": 300},
                "medium": {"parts": 0, "labor": 2000, "paint": 4000, "consumables": 600},
                "large": {"parts": 0, "labor": 3000, "paint": 7000, "consumables": 1000}
            },
            "window damage": {
                "small": {"parts": 4000, "labor": 1500, "paint": 0, "consumables": 300},
                "medium": {"parts": 7000, "labor": 2000, "paint": 0, "consumables": 500},
                "large": {"parts": 12000, "labor": 3000, "paint": 0, "consumables": 800}
            }
        }
        self.vehicle_factors = {
            "economy": 0.8,
            "mid_range": 1.0,
            "luxury": 1.5,
            "premium_luxury": 2.0
        }

    def estimate_detailed_cost(self, damage_type, confidence, image_area, damage_area, vehicle_category, vehicle_info):
        severity = self.assess_severity(damage_area / image_area, confidence)
        base_cost = self.base_costs.get(damage_type, {}).get(severity, {"parts": 0, "labor": 0, "paint": 0, "consumables": 0})
        vehicle_factor = self.vehicle_factors.get(vehicle_category, 1.0)

        detailed_cost = {
            "parts": base_cost["parts"] * vehicle_factor,
            "labor": base_cost["labor"] * vehicle_factor,
            "paint": base_cost["paint"] * vehicle_factor,
            "consumables": base_cost["consumables"] * vehicle_factor
        }
        detailed_cost["total"] = sum(detailed_cost.values())

        return {
            "damage_type": damage_type,
            "severity": severity,
            "confidence": confidence,
            "costs": detailed_cost,
            "vehicle_info": vehicle_info
        }

    def assess_severity(self, relative_area, confidence):
        if relative_area < 0.05 or confidence < 0.3:
            return "small"
        elif relative_area < 0.15 or confidence < 0.7:
            return "medium"
        else:
            return "large"

def generate_enhanced_report(detections, vehicle_info):
    total_cost = sum(d["costs"]["total"] for d in detections)
    parts_cost = sum(d["costs"]["parts"] for d in detections)
    labor_cost = sum(d["costs"]["labor"] for d in detections)
    paint_cost = sum(d["costs"]["paint"] for d in detections)
    consumables_cost = sum(d["costs"]["consumables"] for d in detections)

    report = {
        "report_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "vehicle_info": vehicle_info,
        "damage_summary": [
            {
                "damage_type": d["damage_type"],
                "severity": d["severity"],
                "confidence": d["confidence"],
                "costs": d["costs"]
            } for d in detections
        ],
        "cost_breakdown": {
            "parts": parts_cost,
            "labor": labor_cost,
            "paint": paint_cost,
            "consumables": consumables_cost,
            "total": total_cost
        },
        "repair_time_estimate": estimate_repair_time(detections),
        "additional_notes": [
            "This is an AI-generated estimate based on visual damage detection.",
            "Actual repair costs may vary depending on the specific repair shop, parts availability, and region.",
            "A professional inspection is strongly recommended for a more accurate assessment.",
            f"The total estimated cost of {total_cost:.2f} INR includes all detected damages.",
            "This estimate does not account for potential hidden damages not visible in the images.",
            f"Estimated repair time is {estimate_repair_time(detections)} days, but may vary based on shop workload and parts availability."
        ],
        "recommendations": generate_recommendations(detections, vehicle_info)
    }

    return report

def estimate_repair_time(detections):
    severity_time = {"small": 1, "medium": 2, "large": 4}
    total_days = sum(severity_time[d["severity"]] for d in detections)
    return min(total_days, 30)  # Cap at 30 days

def generate_recommendations(detections, vehicle_info):
    recommendations = []
    
    if any(d["damage_type"] in ["complete damage", "full frontal damage", "full rear damage"] for d in detections):
        recommendations.append("Consider consulting with your insurance provider about potential total loss scenarios.")
    
    if any(d["damage_type"] in ["broken windshield", "broken headlight", "broken taillight"] for d in detections):
        recommendations.append("Prioritize repairs of critical safety components like lights and windshield.")
    
    if sum(d["costs"]["total"] for d in detections) > 50000:
        recommendations.append("For extensive damages, obtain multiple repair quotes from authorized service centers.")
    
    if vehicle_info["year"] < datetime.now().year - 10:
        recommendations.append("For older vehicles, consider cost-effectiveness of repairs versus vehicle value.")
    
    return recommendations

def process_image(file, vehicle_category, vehicle_info):
    try:
        import numpy as np
        import cv2
        from ultralytics import YOLO
    except ImportError as e:
        logger.error(f"Error importing required libraries: {str(e)}")
        return None, []

    try:
        # Load the YOLO model
        model = YOLO('best.pt')  # Update this path to your model
        
        # Read and process the image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Get image dimensions
        image_height, image_width = image.shape[:2]
        image_area = image_height * image_width
        
        # Run inference
        results = model(image)
        
        # Process results
        detections = []
        cost_estimator = DetailedCostEstimator()
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                damage_type = model.names[cls]
                
                # Calculate damage area
                damage_area = (x2 - x1) * (y2 - y1)
                
                # Estimate cost
                cost_estimate = cost_estimator.estimate_detailed_cost(
                    damage_type, conf, image_area, damage_area, vehicle_category, vehicle_info
                )
                detections.append(cost_estimate)
                
                # Draw bounding box
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f"{damage_type}: {conf:.2f}"
                cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Encode the image to base64
        _, buffer = cv2.imencode('.jpg', image)
        image_data = base64.b64encode(buffer).decode('utf-8')
        
        return image_data, detections
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return None, []

@app.route('/api/assess-damage', methods=['POST'])
def assess_damage():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    vehicle_category = request.form.get('vehicle_category', 'mid_range')
    vehicle_make = request.form.get('vehicle_make', 'Unknown')
    vehicle_model = request.form.get('vehicle_model', 'Unknown')
    vehicle_year = int(request.form.get('vehicle_year', datetime.now().year))
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        vehicle_info = {
            "category": vehicle_category,
            "make": vehicle_make,
            "model": vehicle_model,
            "year": vehicle_year
        }
        
        image_data, detections = process_image(file, vehicle_category, vehicle_info)
        detailed_report = generate_enhanced_report(detections, vehicle_info)
        
        return jsonify({
            "image_data": image_data,
            "report": detailed_report
        })
    except Exception as e:
        logger.error(f"Error in assess_damage: {str(e)}")
        return jsonify({"error": "An error occurred while processing the image"}), 500
    
@app.route('/api/generate-pdf', methods=['POST'])
def generate_pdf():
    data = request.json
    report = data['report']

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    # Register a professional font (you may need to provide the actual font file)
    pdfmetrics.registerFont(TTFont('Roboto', 'Roboto-Regular.ttf'))
    pdfmetrics.registerFont(TTFont('Roboto-Bold', 'Roboto-Bold.ttf'))

    elements = []

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='CustomTitle', fontName='Roboto-Bold', fontSize=24, alignment=1, spaceAfter=0.5*inch))
    styles.add(ParagraphStyle(name='CustomHeading2', fontName='Roboto-Bold', fontSize=14, spaceBefore=0.25*inch, spaceAfter=0.25*inch))
    styles.add(ParagraphStyle(name='CustomBodyText', fontName='Roboto', fontSize=10, spaceBefore=0.1*inch))

    # Title
    elements.append(Paragraph("Car Damage Assessment Report", styles['CustomTitle']))
    elements.append(Spacer(1, 0.5*inch))

    # # Add company logo (you'll need to provide the actual logo file)
    # logo = Image('logo.png', width=1*inch, height=1*inch)
    # elements.append(logo)
    # elements.append(Spacer(1, 0.25*inch))

    # Vehicle Information
    elements.append(Paragraph("Vehicle Information", styles['CustomHeading2']))
    vehicle_info = [
        ["Make", report['vehicle_info']['make']],
        ["Model", report['vehicle_info']['model']],
        ["Year", str(report['vehicle_info']['year'])],
        ["Category", report['vehicle_info']['category']]
    ]
    t = Table(vehicle_info, colWidths=[2*inch, 4*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Roboto'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.25*inch))

    # Damage Summary
    elements.append(Paragraph("Damage Summary", styles['CustomHeading2']))
    damage_data = [["Damage Type", "Severity", "Confidence", "Total Cost"]]
    for damage in report['damage_summary']:
        damage_data.append([
            damage['damage_type'],
            damage['severity'],
            f"{damage['confidence']*100:.2f}%",
            f"₹{damage['costs']['total']:.2f}"
        ])
    t = Table(damage_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Roboto-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 1), (-1, -1), 'Roboto'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.25*inch))

    # Cost Breakdown
    elements.append(Paragraph("Cost Breakdown", styles['CustomHeading2']))
    cost_data = [["Category", "Amount"]]
    for key, value in report['cost_breakdown'].items():
        cost_data.append([key.capitalize(), f"₹{value:.2f}"])
    t = Table(cost_data, colWidths=[3*inch, 3*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Roboto-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 1), (-1, -1), 'Roboto'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.25*inch))

    # Additional Information
    elements.append(Paragraph("Additional Information", styles['CustomHeading2']))
    elements.append(Paragraph(f"Estimated Repair Time: {report['repair_time_estimate']} days", styles['CustomBodyText']))
    elements.append(Spacer(1, 0.25*inch))

    # Recommendations
    elements.append(Paragraph("Recommendations", styles['CustomHeading2']))
    for recommendation in report['recommendations']:
        elements.append(Paragraph(f"• {recommendation}", styles['CustomBodyText']))
    elements.append(Spacer(1, 0.25*inch))

    # Additional Notes
    elements.append(Paragraph("Additional Notes", styles['CustomHeading2']))
    for note in report['additional_notes']:
        elements.append(Paragraph(f"• {note}", styles['CustomBodyText']))

    doc.build(elements)
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name='damage_assessment_report.pdf', mimetype='application/pdf')

#------------------------------------------------------------------------------------------------------------------
# Document review

def review_pdf(pdf_file):
    # Read PDF content
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Define the sections we're interested in
    sections_of_interest = [
        "Personal Info", "Metrics", "Lob Specific Data", "Risk Factors",
        "Customer Feedback", "Policy Details", "Claims History", "Loyalty Program",
        "Risk Profile", "Financial Overview", "Personalized Recommendations"
    ]

    # Extract sections and their content
    sections = {}
    current_section = None
    for line in text.split('\n'):
        if line.strip() in sections_of_interest:
            current_section = line.strip()
            sections[current_section] = []
        elif current_section and line.strip():
            sections[current_section].append(line.strip())

    # Extract key information from each section
    important_info = []
    for section in sections_of_interest:
        content = sections.get(section, [])
        if content:
            section_text = " ".join(content)
            
            if section == "Personal Info":
                name_match = re.search(r"Customer Name:\s*(.*?)(?=\s+Email|\s*$)", section_text, re.IGNORECASE)
                gender_match = re.search(r"Gender:\s*(.*?)(?:\s+|$)", section_text)
                marital_match = re.search(r"Marital Status:\s*(.*?)(?:\s+|$)", section_text)
                age_match = re.search(r"Age:\s*(\d+)", section_text)
                occupation_match = re.search(r"Occupation:\s*(.*?)(?=\s+Education Level|\s*$)", section_text)
                education_match = re.search(r"Education Level:\s*(.*?)(?=$|\s+(?:Metrics|Lob Specific Data))", section_text, re.IGNORECASE)
                if name_match:
                    important_info.append(f"Customer Name: {name_match.group(1)}")
                if gender_match:
                    important_info.append(f"Gender: {gender_match.group(1)}")
                if marital_match:
                    important_info.append(f"Marital Status: {marital_match.group(1)}")
                if age_match:
                    important_info.append(f"Age: {age_match.group(1)}")
                if occupation_match:
                    important_info.append(f"Occupation: {occupation_match.group(1).strip()}")
                if education_match:
                    important_info.append(f"Education Level: {education_match.group(1).strip()}")
            elif section == "Metrics":
                match = re.search(r"Churn Risk Score value:\s*([\d.]+)", section_text)
                if match:
                    important_info.append(f"Churn Risk Score: {match.group(1)}")
            elif section == "Lob Specific Data":
                vehicle_make_match = re.search(r"Vehicle Make:\s*(.*?)(?:\s+|$)", section_text)
                vehicle_model_match = re.search(r"Vehicle Model:\s*(.*?)(?:\s+|$)", section_text)
                vehicle_year_match = re.search(r"Vehicle Year:\s*(\d+)", section_text)
                mileage_match = re.search(r"Mileage:\s*(\d+)", section_text)
                accidents_match = re.search(r"Number Of Accidents:\s*(\d+)", section_text)
                violations_match = re.search(r"Number Of Traffic Violations:\s*(\d+)", section_text)
                safety_match = re.search(r"Safety Features:\s*(.*?)(?:\s+|$)", section_text)
                anti_theft_match = re.search(r"Anti Theft Device:\s*(.*?)(?:\s+|$)", section_text)
                parking_match = re.search(r"Parking Location:\s*(.*?)(?:\s+|$)", section_text)
                annual_mileage_match = re.search(r"Annual Mileage:\s*(\d+)", section_text)
                usage_type_match = re.search(r"Usage Type:\s*(.*?)(?:\s+|$)", section_text)
                
                if vehicle_make_match:
                    important_info.append(f"Vehicle Make: {vehicle_make_match.group(1)}")
                if vehicle_model_match:
                    important_info.append(f"Vehicle Model: {vehicle_model_match.group(1)}")
                if vehicle_year_match:
                    important_info.append(f"Vehicle Year: {vehicle_year_match.group(1)}")
                if mileage_match:
                    important_info.append(f"Mileage: {mileage_match.group(1)}")
                if accidents_match:
                    important_info.append(f"Number of Accidents: {accidents_match.group(1)}")
                if violations_match:
                    important_info.append(f"Number of Traffic Violations: {violations_match.group(1)}")
                if safety_match:
                    important_info.append(f"Safety Features: {safety_match.group(1)}")
                if anti_theft_match:
                    important_info.append(f"Anti Theft Device: {anti_theft_match.group(1)}")
                if parking_match:
                    important_info.append(f"Parking Location: {parking_match.group(1)}")
                if annual_mileage_match:
                    important_info.append(f"Annual Mileage: {annual_mileage_match.group(1)}")
                if usage_type_match:
                    important_info.append(f"Usage Type: {usage_type_match.group(1)}")
            elif section == "Risk Factors":
                match = re.search(r"Credit Score:\s*([\d.]+)", section_text)
                if match:
                    important_info.append(f"Credit Score: {match.group(1)}")
            elif section == "Customer Feedback":
                match = re.search(r"Satisfaction Score:\s*([\d.]+)", section_text)
                if match:
                    important_info.append(f"Satisfaction Score: {match.group(1)}")
            elif section == "Policy Details":
                policy_number_match = re.search(r"Policy Number:\s*(.*?)(?:\s+|$)", section_text)
                product_plan_match = re.search(r"Product Plan:\s*(.*?)(?:\s+|$)", section_text)
                coverage_amount_match = re.search(r"Coverage Amount:\s*([\d.]+)", section_text)
                premium_amount_match = re.search(r"Premium Amount:\s*([\d.]+)", section_text)
                expiration_date_match = re.search(r"Policy Expiration Date:\s*(.*?)(?:\s+|$)", section_text)
                renewal_status_match = re.search(r"Policy Renewal Status:\s*(.*?)(?:\s+|$)", section_text)
                
                if policy_number_match:
                    important_info.append(f"Policy Number: {policy_number_match.group(1)}")
                if product_plan_match:
                    important_info.append(f"Product Plan: {product_plan_match.group(1)}")
                if coverage_amount_match:
                    important_info.append(f"Coverage Amount: ₹{coverage_amount_match.group(1)}")
                if premium_amount_match:
                    important_info.append(f"Premium Amount: ₹{premium_amount_match.group(1)}")
                if expiration_date_match:
                    important_info.append(f"Policy Expiration Date: {expiration_date_match.group(1)}")
                if renewal_status_match:
                    important_info.append(f"Policy Renewal Status: {renewal_status_match.group(1)}")
            elif section == "Claims History":
                match = re.search(r"Total Claims:\s*([\d.]+)", section_text)
                if match:
                    important_info.append(f"Total Claims: {match.group(1)}")
            elif section == "Loyalty Program":
                loyalty_tier_match = re.search(r"Loyalty Tier:\s*(.*?)(?:\s+|$)", section_text)
                points_earned_match = re.search(r"Points Earned:\s*(\d+)", section_text)
                rewards_redeemed_match = re.search(r"Rewards Redeemed:\s*(\d+)", section_text)
                leaderboard_position_match = re.search(r"Leaderboard Position:\s*(\d+)", section_text)
                benefits_available_match = re.search(r"Benefits Available:\s*(.*?)(?:\s+|$)", section_text)
                
                if loyalty_tier_match:
                    important_info.append(f"Loyalty Tier: {loyalty_tier_match.group(1)}")
                if points_earned_match:
                    important_info.append(f"Points Earned: {points_earned_match.group(1)}")
                if rewards_redeemed_match:
                    important_info.append(f"Rewards Redeemed: {rewards_redeemed_match.group(1)}")
                if leaderboard_position_match:
                    important_info.append(f"Leaderboard Position: {leaderboard_position_match.group(1)}")
                if benefits_available_match:
                    important_info.append(f"Benefits Available: {benefits_available_match.group(1)}")
            elif section == "Risk Profile":
                match = re.search(r"Risk Assessment Score:\s*([\d.]+)", section_text)
                if match:
                    important_info.append(f"Risk Assessment Score: {match.group(1)}")
            elif section == "Financial Overview":
                match = re.search(r"Customer Lifetime Value:\s*([\d.]+)", section_text)
                if match:
                    important_info.append(f"Customer Lifetime Value: ₹{match.group(1)}")
            elif section == "Personalized Recommendations":
                match = re.search(r"Recommend.*?\.", section_text)
                if match:
                    important_info.append(f"Recommendation: {match.group(0)}")

    return important_info

def send_email(pdf_buffer, filename):
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")
    receiver_email = "K-insure@outlook.com"

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = "K-INSURE CUSTOMER REVIEW"

    body = "Review the individual customer review pdf and report to the K-Insure control team."
    msg.attach(MIMEText(body, 'plain'))

    pdf_attachment = MIMEApplication(pdf_buffer.getvalue(), _subtype="pdf")
    pdf_attachment.add_header('Content-Disposition', f'attachment; filename={filename}')
    msg.attach(pdf_attachment)

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
        logger.info("Email sent successfully")
        return True
    except Exception as e:
        logger.error(f"Error sending email: {str(e)}")
        if isinstance(e, smtplib.SMTPAuthenticationError):
            logger.error("Authentication failed. Please check your email and app password.")
        elif isinstance(e, smtplib.SMTPException):
            logger.error("SMTP error occurred. Please check your email settings.")
        else:
            logger.error("An unexpected error occurred while sending the email.")
        return False

@app.route('/api/review-pdf', methods=['POST'])
def review_pdf_route():
    if 'pdf' not in request.files:
        return jsonify({"error": "No PDF file provided"}), 400

    pdf_file = request.files['pdf']
    if pdf_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        important_info = review_pdf(pdf_file)
        
        # Generate PDF
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
        
        # Register custom fonts (make sure these font files are in your project directory)
        pdfmetrics.registerFont(TTFont('Roboto', 'Roboto-Regular.ttf'))
        pdfmetrics.registerFont(TTFont('Roboto-Bold', 'Roboto-Bold.ttf'))

        styles = getSampleStyleSheet()
        styles['Title'].fontName = 'Roboto-Bold'
        styles['Title'].fontSize = 18
        styles['Title'].spaceAfter = 12
        styles['Title'].textColor = colors.darkblue

        styles['BodyText'].fontName = 'Roboto'
        styles['BodyText'].fontSize = 10
        styles['BodyText'].spaceAfter = 6

        elements = []

        # Title
        elements.append(Paragraph("K-Insure Customer Review", styles['Title']))
        elements.append(Spacer(1, 0.25*inch))

        # Table of extracted information
        data = [['Key Information', 'Value']]
        for info in important_info:
            key, value = info.split(':', 1)
            data.append([Paragraph(key.strip() + ":", styles['BodyText']), Paragraph(value.strip(), styles['BodyText'])])

        table = Table(data, colWidths=[2.5*inch, 3.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Roboto-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 1), (-1, -1), 'Roboto'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BOX', (0, 0), (-1, -1), 2, colors.black),
        ]))
        elements.append(table)

        elements.append(Spacer(1, 0.25*inch))
        elements.append(Paragraph("This review is intended for the K-insure review company and control team.", styles['BodyText']))

        doc.build(elements)
        buffer.seek(0)

        # Send email with the generated PDF
        email_sent = send_email(buffer, 'document_review.pdf')

        # Return the PDF file to the client
        buffer.seek(0)
        if email_sent:
            return send_file(
                buffer,
                as_attachment=True,
                download_name='document_review.pdf',
                mimetype='application/pdf'
            ), 200
        else:
            return send_file(
                buffer,
                as_attachment=True,
                download_name='document_review.pdf',
                mimetype='application/pdf'
            ), 206, {"X-Email-Status": "Failed to send email, but PDF generated successfully"}

    except Exception as e:
        logger.error(f"Error in review_pdf_route: {str(e)}")
        return jsonify({"error": str(e)}), 500


#------------------------------------------------------------------------------------------------------------------
# Document Summarization



#-------------------------------------------------------------------------------------------------------------------
# no warnings
def safe_log(x):
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            return np.log(x)
        except:
            return 0

#-------------------------------------------------------------------------------------------------------------------
# health check   
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200


#-------------------------------------------------------------------------------------------------------------------
# main function 
if __name__ == '__main__':
    # This block will execute when the script is run directly
    port = int(os.environ.get('PORT', 5000))
    threads = int(os.environ.get('THREADS', 4))  # Number of threads, adjust as needed
    logger.info(f"Starting server on port {port} with {threads} threads")
    serve(app, host='0.0.0.0', port=port, threads=threads, 
          channel_timeout=30, # Similar to Gunicorn's timeout
          cleanup_interval=5, # How often to check for timed out connections
          connection_limit=1000 # Similar to Gunicorn's worker_connections
    )