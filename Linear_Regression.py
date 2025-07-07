import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#Load data from CSV 
df=pd.read_csv('regression_data.csv')
#split data into features (X) and target (y)
X=df[['HoursStudied']]
y=df['ExamScore']
#split data into training and testing sets (80% training, 20% testing)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#Train the model
model=LinearRegression()
model.fit(X_train,y_train)
#Streamlit User Interface
st.title("Exam Score Predictior")
#User Input
hours_studied=st.number_input("Enter the number of hours studied",min_value=0.0, step=0.1)
#Predict Button
if st.button("Predict"):
    #Make prediction
    predicted_score=model.predict([[hours_studied]])[0]
    #Display prediction
    st.success("Predicted Exam Score: {:.2f}".format(predicted_score))
#Showo Sample Data
st.write("### Sample Training Data")
st.dataframe(df)
