
Customer Churn Prediction Project - Setup Instructions
======================================================

1. Clone the GitHub Repository
-------------------------------
Open your terminal or command prompt and run:

    git clone https://github.com/demo44146/customer_churn_prediction.git
    cd customer_churn_prediction

2. (Optional) Create and Activate a Virtual Environment
--------------------------------------------------------

On Windows:
    python -m venv venv
    venv\Scripts\activate

On macOS/Linux:
    python3 -m venv venv
    source venv/bin/activate

3. Install Required Python Packages
------------------------------------

First, make sure pip is up to date:

    python -m pip install --upgrade pip

Then install all required packages:

    pip install -r requirements.txt

If requirements.txt is not available, manually install:

    pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn flask

4. Verify Model and Encoder Files
----------------------------------
Ensure the following files exist in the project directory:

    - customer_churn_model.pkl
    - encoders.pkl

If not, run the model training script to generate them.

5. Run the Flask Web Application
---------------------------------

Start the Flask app using:

    python app.py

The server will start and you'll see something like:

    * Running on http://127.0.0.1:5000/

6. Open the App in Browser
---------------------------

Go to http://127.0.0.1:5000/ in your browser.

- Fill in the form using dropdowns.
- Click "Predict".
- You'll see whether the customer is likely to churn, along with the probability.

7. Stop the Flask App
----------------------

To stop the server, press:

    Ctrl + C

8. (Optional) Retrain the Model
-------------------------------

If needed, run the training notebook/script to recreate:

    - customer_churn_model.pkl
    - encoders.pkl

These files are used by the Flask app to make predictions.
