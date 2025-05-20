







# K-Insure (Backend):

## About the Project
K-Insure is a comprehensive Flask-based web application developed as an internship project for an IT company. It is designed for insurance companies to analyze customer data, perform churn prediction, conduct sentiment analysis, segment customers, and assess vehicle damage. The application integrates with a PostgreSQL database and leverages advanced machine learning and AI techniques to provide actionable insights. Key features include:

- **User Authentication**: Secure signup and login functionality with password hashing.
- **Exploratory Data Analysis (EDA)**: Visualizations for churn rate, customer demographics, payment modes, and more using Plotly.
- **Sentiment Analysis**: Analysis of customer feedback with visualizations like sentiment distribution, trends, and word clouds.
- **Customer Segmentation**: RFM (Recency, Frequency, Monetary) analysis to categorize customers into segments like Champions, Loyal Customers, etc.
- **Churn Prediction**: Machine learning model (ExtraTreesClassifier) to predict customer churn risk.
- **Personalized Customer Dashboard**: Detailed customer insights including policy details, claims history, loyalty program, and personalized recommendations.
- **Geographical Insights**: Region-based analysis of customer metrics with GeoJSON data for Indian states.
- **Vehicle Damage Assessment**: AI-driven damage detection using YOLO and cost estimation for vehicle repairs with PDF report generation.
- **Document Review**: PDF review and extraction of key customer information with automated email reporting.

The application uses a PostgreSQL database for data storage and employs libraries like Pandas, Scikit-learn, Ultralytics YOLO, and ReportLab for various functionalities.

## Prerequisites
To set up and run the K-Insure project, ensure you have the following installed:
- Python 3.8 or higher
- pip (Python package manager)
- PostgreSQL (version 12 or higher)
- A valid Gmail account with an App Password for email functionality
- Roboto font files (`Roboto-Regular.ttf` and `Roboto-Bold.ttf`) for PDF generation
- YOLO model file (`best.pt`) for vehicle damage assessment
- Access to a PostgreSQL database with the `laster` table containing customer data

## Setup Instructions
1. **Clone or Download the Project**
   - Obtain the project files (`app.py`, `requirements.txt`, `.env`) from the internship project repository, your IT company’s internal system, or the public release repository.
   - Ensure the `templates/` folder contains any required HTML files (e.g., `index.html` if used for a front-end interface).
   - Place `Roboto-Regular.ttf` and `Roboto-Bold.ttf` in the project directory for PDF generation (download from Google Fonts if not provided).
   - Place the YOLO model file `best.pt` in the project directory for damage assessment.

2. **Set Up a Virtual Environment (Recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   Install the required Python packages listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
   The `requirements.txt` includes:
   - Flask and Flask-CORS for the web framework
   - Pandas, NumPy, Plotly, Matplotlib, Seaborn for data analysis and visualization
   - Scikit-learn for machine learning
   - SQLAlchemy and psycopg2-binary for PostgreSQL integration
   - Ultralytics, OpenCV, and Pillow for image processing
   - ReportLab for PDF generation
   - Sumy and Spacy for text processing
   - WordCloud, NLTK, Requests, BeautifulSoup, and others for additional functionalities

4. **Configure Environment Variables**
   - Create or update the `.env` file in the project root with the following:
     ```
     SENDER_EMAIL=your_email@gmail.com
     SENDER_PASSWORD=your_gmail_app_password
     ```
     - Replace `your_email@gmail.com` with your Gmail address.
     - Replace `your_gmail_app_password` with a Gmail App Password (generate one from your Google Account settings with 2-Step Verification enabled).
   - If integrating OpenAI (e.g., for text embeddings), add:
     ```
     OPENAI_API_KEY=your_openai_api_key
     ```

5. **Set Up PostgreSQL Database**
   - Install and configure PostgreSQL on your system.
   - Create a database named `postgres` (or modify the connection string in `app.py`).
   - Ensure the `laster` table exists with the required columns (e.g., `CPR_NO`, `CHURN`, `GWP`, `Age`, `PAYMENT_MODE`, etc.).
   - Update the database connection string in `app.py` if necessary:
     ```python
     engine = create_engine('postgresql://postgres:your_password@localhost:5432/postgres')
     ```
     Replace `your_password` with your PostgreSQL password and adjust the host/port as needed.

6. **Directory Structure**
   Ensure the following structure in your project directory:
   ```
   project_directory/
   ├── templates/
   │   └── index.html  # If used for front-end interface
   ├── uploads/        # Created automatically for file uploads
   ├── best.pt         # YOLO model file
   ├── Roboto-Regular.ttf
   ├── Roboto-Bold.ttf
   ├── app.py
   ├── requirements.txt
   └── .env
   ```

## Running the Project
1. **Start the Flask Application**
   Navigate to the project directory and run:
   ```bash
   python app.py
   ```
   This starts the Waitress server on port 5000 with 4 threads. You should see logs indicating the server is running at `http://0.0.0.0:5000`.

2. **Access the Application**
   Open a web browser and navigate to `http://localhost:5000` for any front-end interface (if `index.html` is used). Most functionalities are API-based and can be accessed via tools like Postman or cURL.

3. **Using the Application**
   - **User Authentication**: Use `/api/signup` and `/api/login` endpoints to register or log in users.
   - **EDA Visualizations**: Access `/api/eda` to retrieve churn rate, GWP, and other visualizations.
   - **Sentiment Analysis**: Use `/api/visualizations` for sentiment distribution, trends, and word clouds.
   - **Customer Segmentation**: Access `/api/rfm-analysis` for RFM-based customer segmentation.
   - **Churn Prediction**: Use `/api/predict` to predict churn risk and `/api/train` to retrain the model.
   - **Customer Dashboard**: Access `/api/customer-dashboard` with `lob` and `cpr_no` to view personalized customer data.
   - **Geographical Insights**: Use `/get_india_data` for region-based analytics and `/get_region_churn_data` for specific region details.
   - **Vehicle Damage Assessment**: Use `/api/assess-damage` to upload images and get damage estimates, and `/api/generate-pdf` for PDF reports.
   - **Document Review**: Use `/api/review-pdf` to upload customer PDFs, extract key information, and receive an emailed report.

## Notes
- **Database**: Ensure the `laster` table in PostgreSQL contains all required columns for the features used (e.g., `CHURN`, `GWP`, `Feedback_Sentiment`, etc.).
- **Email Configuration**: The application sends emails using a Gmail account. Ensure the `.env` file has valid credentials and an App Password.
- **YOLO Model**: The `best.pt` file must be a trained YOLO model compatible with Ultralytics for damage detection.
- **Fonts**: The `Roboto-Regular.ttf` and `Roboto-Bold.ttf` files are required for PDF generation. Download from Google Fonts if not provided.
- **Performance**: The application uses ThreadPoolExecutor for concurrent visualization generation. Adjust the `max_workers` in `app.py` based on your system’s capabilities.
- **Security**: The `.env` file contains sensitive information (email credentials). Keep it secure and avoid sharing it publicly.
- **Dependencies**: Some packages (e.g., `ultralytics`, `spacy`) may require additional setup (e.g., download Spacy models: `python -m spacy download en_core_web_sm`).

## Troubleshooting
- **Module Not Found Error**: Ensure all dependencies are installed using `pip install -r requirements.txt`. Check for version compatibility issues.
- **Database Connection Issues**: Verify PostgreSQL is running, the connection string is correct, and the `laster` table exists.
- **Email Sending Errors**: Check the `.env` file for correct Gmail credentials and App Password. Ensure your Gmail account allows less secure apps or has 2-Step Verification enabled.
- **YOLO Model Errors**: Confirm the `best.pt` file exists and is compatible with the Ultralytics library.
- **PDF Generation Issues**: Ensure `Roboto-Regular.ttf` and `Roboto-Bold.ttf` are in the project directory.
- **Port Conflicts**: If port 5000 is in use, modify the port in `app.py` or set the `PORT` environment variable.
- **Memory Issues**: For large datasets or image processing, ensure sufficient system memory and adjust `max_workers` in ThreadPoolExecutor.

## License
This project was developed as part of an internship at KANINI Software Solutions India Private Limited. All rights are reserved by KANINI Software Solutions India Private Limited. The source code, documentation, and related assets are proprietary and primarily intended for internal use within KANINI Software Solutions India Private Limited or for educational purposes during the internship. The project has also been released for public use under the following conditions:

- **Public Use**: Public users may use, copy, and modify the source code for non-commercial, educational, or personal purposes only. Any commercial use, redistribution, or derivative works require prior written consent from KANINI Software Solutions India Private Limited.
- **Restrictions**: Unauthorized commercial use, distribution, or modification outside the scope of the internship or public non-commercial use is strictly prohibited without prior written consent from KANINI Software Solutions India Private Limited.
- **Attribution**: When using or sharing this project publicly for non-commercial purposes, please credit KANINI Software Solutions India Private Limited and the internship team.
- **Contact**: For inquiries regarding usage, permissions, or commercial licensing, contact KANINI Software Solutions India Private Limited’s project supervisor or legal department.

This license ensures that the project remains proprietary while allowing limited public access for educational and non-commercial use, as per the release terms.
