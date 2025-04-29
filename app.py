import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import joblib

# Load your dataset
df = pd.read_csv('cleaned_restaurants_updated.csv')

# Handle missing or categorical values (Example: using LabelEncoder for 'Category' and 'Locality')
label_encoder_category = LabelEncoder()
df['Category'] = label_encoder_category.fit_transform(df['Category'])

label_encoder_locality = LabelEncoder()
df['Locality'] = label_encoder_locality.fit_transform(df['Locality'])

# Feature Engineering: Calculating additional columns
df['Avg_Rating'] = (df['Dining_Rating'] + df['Delivery_Rating']) / 2
df['Dining_Impact'] = df['Dining_Rating'] * df['Dining_Review_Count']
df['Delivery_Impact'] = df['Delivery_Rating'] * df['Delivery_Rating_Count']
df['Price_per_Rating'] = df['Dining_Rating'] + df['Delivery_Rating']
df['Review_Impact'] = df['Dining_Review_Count'] + df['Delivery_Rating_Count']

# Define features and target
features = df[['Category', 'Locality', 'Dining_Rating', 'Dining_Review_Count', 
                'Delivery_Rating', 'Delivery_Rating_Count', 'Latitude', 'Longitude', 
                'Avg_Rating', 'Dining_Impact', 'Delivery_Impact', 'Price_per_Rating', 'Review_Impact']]

target = df['Pricing_for_2']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")

# Optionally, save the trained model using joblib
joblib.dump(model, 'restaurant_price_model.pkl')
joblib.dump(label_encoder_category, 'label_encoder_category.pkl')
joblib.dump(label_encoder_locality, 'label_encoder_locality.pkl')

# Get user input for restaurant details
category = input("Enter the restaurant category (e.g., Indian, Fast Food): ")
locality = input("Enter the restaurant locality (e.g., Delhi, Bangalore): ")
dining_rating = float(input("Enter the Dining Rating (e.g., 4.5): "))
dining_review_count = int(input("Enter the Dining Review Count (e.g., 3222): "))
delivery_rating = float(input("Enter the Delivery Rating (e.g., 4.9): "))
delivery_rating_count = int(input("Enter the Delivery Rating Count (e.g., 275): "))
latitude = float(input("Enter the Latitude of the restaurant (e.g., 12.9352): "))
longitude = float(input("Enter the Longitude of the restaurant (e.g., 77.6245): "))

# Calculate additional columns based on the input
avg_rating = (dining_rating + delivery_rating) / 2
dining_impact = dining_rating * dining_review_count
delivery_impact = delivery_rating * delivery_rating_count
price_per_rating = dining_rating + delivery_rating
review_impact = dining_review_count + delivery_rating_count

# Create a dictionary for the new data
new_data = {
    'Category': category,
    'Locality': locality,
    'Dining_Rating': dining_rating,
    'Dining_Review_Count': dining_review_count,
    'Delivery_Rating': delivery_rating,
    'Delivery_Rating_Count': delivery_rating_count,
    'Latitude': latitude,
    'Longitude': longitude,
    'Avg_Rating': avg_rating,
    'Dining_Impact': dining_impact,
    'Delivery_Impact': delivery_impact,
    'Price_per_Rating': price_per_rating,
    'Review_Impact': review_impact
}

# Convert the new data into a DataFrame
new_data_df = pd.DataFrame([new_data])

# Encode categorical variables for new data using the saved encoders
# Handle unseen 'Category' value
try:
    new_data_df['Category'] = label_encoder_category.transform(new_data_df['Category'])
except ValueError:
    # Fallback to the most frequent category in training data
    most_frequent_category = label_encoder_category.transform([label_encoder_category.classes_[0]])[0]
    new_data_df['Category'] = most_frequent_category

# Handle unseen 'Locality' value
try:
    new_data_df['Locality'] = label_encoder_locality.transform(new_data_df['Locality'])
except ValueError:
    # Fallback to the most frequent locality in training data
    most_frequent_locality = label_encoder_locality.transform([label_encoder_locality.classes_[0]])[0]
    new_data_df['Locality'] = most_frequent_locality

# Predict the price for the new restaurant data
predicted_price = model.predict(new_data_df)
print(f"Predicted Price for the entered restaurant: ₹{predicted_price[0]:.2f}")