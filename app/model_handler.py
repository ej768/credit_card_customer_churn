import xgboost as xgb
import pandas as pd
import numpy as np

class ModelHandler:
    def __init__(self, model_path: str):
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        
        # Define the expected columns (from X.columns)
        self.expected_columns = [
            'Total_Relationship_Count', 'Months_Inactive_12_mon',
            'Contacts_Count_12_mon', 'Total_Revolving_Bal', 'Total_Trans_Ct',
            'Gender_F', 'Gender_M', 'Education_Level_College',
            'Education_Level_Doctorate', 'Education_Level_Graduate',
            'Education_Level_High School', 'Education_Level_Post-Graduate',
            'Education_Level_Uneducated', 'Education_Level_Unknown',
            'Income_Category_$120K +', 'Income_Category_$40K - $60K',
            'Income_Category_$60K - $80K', 'Income_Category_$80K - $120K',
            'Income_Category_Less than $40K', 'Income_Category_Unknown'
        ]

    def encode_features(self, features: dict) -> pd.DataFrame:
        """
        Encode the categorical features and return a DataFrame with the correct columns.
        """
        # Create a DataFrame from the input dictionary
        df = pd.DataFrame([features])
        
        # One-hot encode categorical columns
        df_encoded = pd.get_dummies(df, columns=['Gender', 'Education_Level', 'Income_Category'], drop_first=False)

        # Ensure the order and presence of all expected columns
        df_encoded = df_encoded.reindex(columns=self.expected_columns, fill_value=0)

        return df_encoded

    def predict(self, features: dict):
        """
        Takes a dictionary of features, encodes them, returns prediction.
        """
        # Encode the features
        encoded_features = self.encode_features(features)
        
        # Convert the DataFrame to a numpy array
        feature_array = encoded_features.to_numpy().astype(float)

        # Perform prediction
        pred = self.model.predict(feature_array)
        prob = self.model.predict_proba(feature_array)

        return {
            "prediction": int(pred[0]),
            "churn_probability": float(prob[0][1])
        }
