# 🍽️ Restaurant Price Prediction System

This project predicts the **average cost for two** at restaurants based on features such as location, category, and rating. It uses a **trained machine learning model** Users can input restaurant attributes and instantly get a price prediction.

---

## 📁 Project Structure

```
Restaurant-Price-Prediction-System/
│
├── app.py                                  # Flask backend application
├── cleaned_restaurants_updated.csv         # Updated dataset
├── label_encoder_category.pkl              # Category label encoder
├── label_encoder_locality.pkl              # Locality label encoder
├── restaurant_price_model.pkl              # Trained ML model
│
├── Dataset/
│   └── cleaned_restaurants.csv             # Original dataset
│             
│
└── README.md                               # Project documentation
```

---

## 🧠 Features

- Predict restaurant price based on location, category, and rating.
- User-friendly interface for input and prediction.
- Trained regression model using real restaurant data.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.x
- scikit-learn
- pandas
- (Optional for conversion: PHP 7.x if using a PHP interface)

### Installation

1. Clone or download the project:
   ```bash
   git clone https://github.com/yourusername/restaurant-price-prediction-system.git
   ```

2. Navigate to the project directory:
   ```bash
   cd restaurant-price-prediction-system
   ```

3. Install required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the Flask server:
   ```bash
   python app.py
   ```

---

## 🧪 Model Overview

- **Algorithm**: Regression Model (e.g., Linear Regression or Random Forest)
- **Inputs**: Location, Category, Rating
- **Output**: Predicted cost for two
- **Training Data**: Based on real-world restaurant data (Zomato/Kaggle sources)
- **Preprocessing**: Label encoding, feature cleaning

---

## ✨ Technologies Used

- **Machine Learning**: scikit-learn, pandas
- **Model File Format**: Pickle (`.pkl`)

---

## 📷 Screenshot
### RESULT
> ![RESULT](https://github.com/ShrihariKasar/Restaurant-Price-Prediction-System/blob/main/Screenshot%20(213).png)

---

## 👨‍💻 Developers

- **Shrihari Kasar**
- **Utkarsha Kakulte**

---

## 📄 License

This project is licensed under the **MIT License**. You are free to use, modify, and distribute it.

---

## 🤝 Contributing

Feel free to fork this repository and submit a pull request with improvements or fixes. All contributions are welcome!

```
