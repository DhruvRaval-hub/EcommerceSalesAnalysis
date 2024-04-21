from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load your DataFrame 'df_selected' and target variable 'Profit' here
df_selected = pd.read_csv("/Users/dhruv/Desktop/capstone UI Done/df_Selected_new.csv")

# Assuming 'int64_df' is your DataFrame containing only int64 columns
X = df_selected.drop(columns=['Profit'])
y = df_selected['Profit']
# Assuming your DataFrame is loaded and processed here

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Initialize the Random Forest Regressor
regressor = ExtraTreesRegressor()

# Fit the model on the training data
regressor.fit(X_train, y_train)


@app.route('/')
def index():
    return render_template('index.html')


# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         input_data = [float(x) for x in request.form.values()]
#         order_amount = input_data[0]
#         item_cost = input_data[1]
#         item_quantity = input_data[2]
        
#         profit = (order_amount - (item_cost * item_quantity))

#         if profit <= 0:
#             result = "No Profit Made"
#         else:
#             result = "Profit Made: ${}".format(profit)

#         return render_template('result.html', prediction=result)

        # input_data_numpyarray = np.asarray(input_data)
        # input_reshape = input_data_numpyarray.reshape(1, -1)
        # prediction = regressor.predict(input_reshape)

@app.route('/predict', methods=['POST'])

def predict():
    if request.method == 'POST':
        company = request.form['company']
        input_data = [float(request.form['input1']), float(request.form['input2']), float(request.form['input3'])]
        input_data_numpyarray = np.asarray(input_data)
        input_reshape = input_data_numpyarray.reshape(1, -1)
        prediction = regressor.predict(input_reshape)

        # Assuming prediction is a list with one element
        result = "Expected Profit for next Month : ${:.2f}".format(prediction[0])
   

        prediction1 = input_data[0] - (input_data[1] * input_data[2])

        if prediction1 <= 0:
            result = "No Profit Made"
        else:
            result1 = "Actual Profit Made Based On Data Provided: ${}".format(prediction1)

        return render_template('result.html', prediction=result, prediction1=result1, company=company)


if __name__ == '__main__':
    app.run(debug=True)
