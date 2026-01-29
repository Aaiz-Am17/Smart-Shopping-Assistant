# Smart Shopping Assistant 🛍️✨

## Your Intelligent Guide to Smarter Purchases

[![GitHub stars](https://img.shields.io/github/stars/Aaiz-Am17/Smart-Shopping-Assistant?style=social)](https://github.com/Aaiz-Am17/Smart-Shopping-Assistant/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/Aaiz-Am17/Smart-Shopping-Assistant?color=brightgreen)](https://github.com/Aaiz-Am17/Smart-Shopping-Assistant/commits/main)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Built with CustomTkinter](https://img.shields.io/badge/GUI-CustomTkinter-blueviolet.svg)](https://customtkinter.tomschimansky.com/)
[![Powered by Gemini API](https://img.shields.io/badge/AI-Gemini%20API-orange.svg)](https://ai.google.dev/gemini-api)

---

## 💡 Project Overview & Motivation

The **Smart Shopping Assistant** is an intuitive desktop application designed to empower your purchasing decisions for home appliances, specifically Air Conditioners and Smart TVs.

**The core purpose of building this application is to address a common frustration in online and traditional shopping: the lack of truly personalized and intelligent product recommendations.**

Currently, most online shopping platforms offer generic filters (e.g., filter by brand, price range). While useful if you already know *exactly* what you want, they fall short when a customer is unsure or wants a product tailored to specific, nuanced needs. Imagine you need an AC that's low on power consumption but you're indifferent to its noise level, or a TV with specific smart features but aren't tied to a particular brand. Standard filters can't grasp these preferences.

This is where the Smart Shopping Assistant comes in. Think of it as an **AI-powered salesperson, but without the pressure of a sale.** It's built to:

1.  **Understand Your Needs:** Through carefully designed questions and filters, the application gathers information about your priorities and preferences for a product's features.
2.  **Predict Price Range:** Leveraging advanced machine learning models (Random Forest and XGBoost in an Ensemble Voting Regressor), the assistant provides accurate price predictions based on your tailored desires.
3.  **Offer Tailored Recommendations:** Integrated with the cutting-edge Google Generative AI (Gemini) API, the application goes a step further. Instead of just showing products, it processes your unique requirements and provides a concise, intelligent recommendation that caters to your specific wants. For example, if low power consumption is paramount for your AC, the AI will suggest a model that prioritizes this, even if noise level isn't a concern.

While the current version utilizes the Gemini API for recommendations, the main idea is that in a real-world scenario, this AI prompt would interface with a comprehensive product database via an SQL query in the backend. This would allow the system to suggest actual products from specific companies that perfectly match the user's filtered and AI-interpreted needs.

Say goodbye to endless Browse and generic filters, and hello to smart, confident buying with truly personalized assistance!

---

## ✨ Key Features

* **Intelligent Price Prediction:**
    * **Air Conditioners:** Predicts prices based on crucial factors like `Condenser Coil material`, `Refrigerant type`, `Power Consumption (Low/Medium/High)`, and `Noise Level (Low/Medium/High)`.
    * **Smart TVs:** Estimates prices considering `Operating System`, `Picture Quality (e.g., 4K, Full HD)`, `Speaker Output`, `Refresh Frequency`, and available `Streaming Channels`.
* **AI-Powered Product Recommendations:** Harnesses the power of the Gemini API to provide concise, relevant product suggestions tailored to your selected features, helping you choose the best fit for your home.
* **Modern & Responsive GUI:** Crafted using `CustomTkinter`, offering a visually appealing and easy-to-navigate interface that enhances the user experience.
* **Robust Machine Learning Backend:** Employs an ensemble of **Random Forest** and **XGBoost** regressors, fine-tuned with `RandomizedSearchCV` and validated using `cross-validation`, ensuring high prediction accuracy.
* **Modular & Maintainable Codebase:** The project is structured into logical modules, making it easy to understand, extend, and maintain.

---

## 🚀 How It Works

1.  **User Input:** You select desired features for either an Air Conditioner or a Smart TV through the interactive GUI.
2.  **Data Preprocessing:** Your selections, along with historical data, are meticulously cleaned and transformed (One-Hot Encoding for categorical data, Standardization for numerical data) to prepare them for the models.
3.  **Price Prediction:** The trained ensemble machine learning model takes your processed input and predicts an estimated market price for the product.
4.  **AI Recommendation:** Your chosen features are sent to the Google Gemini API, which processes the request and generates a tailored product recommendation, suggesting the most suitable brand/model based on the specified criteria.
5.  **Results Display:** Both the predicted price and the AI-generated recommendation are displayed clearly in the application, helping you make an informed decision.

---

## 📁 Project Structure

```

SmartShoppingAssistant/
├── data/                       \# Contains the raw dataset files
│   ├── Air\_condition\_dataset.csv
│   └── TELEVISION.csv
├── ac\_model.py                 \# Encapsulates all Air Conditioner (AC) model logic:
│                               \# - Data loading and preprocessing specific to ACs.
│                               \# - Training of RandomForest and XGBoost for ACs.
│                               \# - Ensemble model creation and evaluation for AC price prediction.
├── tv\_model.py                 \# Encapsulates all Smart TV (TV) model logic:
│                               \# - Data loading and preprocessing specific to TVs.
│                               \# - Training of RandomForest and XGBoost for TVs.
│                               \# - Ensemble model creation and evaluation for TV price prediction.
├── gui\_app.py                  \# Manages the CustomTkinter graphical user interface:
│                               \# - Defines the layout, widgets, and user interaction flow.
│                               \# - Calls methods from `ac_model.py` and `tv_model.py` for predictions.
│                               \# - Integrates with the Google Generative AI Client for recommendations.
├── main.py                     \# The primary entry point of the application:
│                               \# - Initializes the `ACModel` and `TVModel` instances.
│                               \# - Sets up the Google Generative AI client with your API key.
│                               \# - Launches the `ProductChoiceInterface` (GUI).
├── requirements.txt            \# Lists all Python libraries required to run the project.
└── README.md                   \# This comprehensive project overview.

````

---

## 🛠️ Setup and Installation

To get the Smart Shopping Assistant up and running on your local machine, follow these simple steps:

### 1. Clone the Repository

First, clone this repository to your local system using Git:

```bash
git clone https://github.com/Aaiz-Am17/Smart-Shopping-Assistan.git
cd Smart-Shopping-Assistant
````

### 2\. Place Dataset Files

**Create a `data` folder** inside the `Smart-Shopping-Assistant` directory, and **place your `Air_condition_dataset.csv` and `TELEVISION.csv` files inside it.**

The final structure should look like this:

```
Smart-Shopping-Assistant/
├── data/
│   ├── Air_condition_dataset.csv
│   └── TELEVISION.csv
├── ac_model.py
├── ...
```

Next, **update the file paths in `main.py`** to reflect this new structure. They should now be relative to the `main.py` file:

```python
# In main.py
import os

# Get the directory of the current script (main.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct paths relative to the script's directory
AC_FILE_PATH = os.path.join(BASE_DIR, "data", "Air_condition_dataset.csv")
TV_FILE_PATH = os.path.join(BASE_DIR, "data", "TELEVISION.csv")
```

Using `os.path.join` and `os.path.dirname(os.path.abspath(__file__))` makes your paths **operating system independent** and ensures they work whether your project is cloned or moved around\!

### 3\. Create a Virtual Environment (Recommended)

It's best practice to use a virtual environment to manage project dependencies.

```bash
python -m venv venv
```

**Activate the virtual environment:**

  * **On Windows:**
    ```bash
    .\venv\Scripts\activate
    ```
  * **On macOS/Linux:**
    ```bash
    source venv/bin/activate
    ```

### 4\. Install Dependencies

With your virtual environment activated, install all required Python packages:

```bash
pip install -r requirements.txt
```

### 5\. Get a Google Generative AI API Key

To enable the AI recommendation feature, you'll need an API key from Google AI Studio.

1.  Visit [Google AI Studio](https://aistudio.google.com/app/apikey).

2.  Follow the instructions to create a new API key.

3.  **Update the `API_KEY` variable in `main.py`** with your generated key:

    ```python
    # In main.py
    API_KEY = "YOUR_GENERATED_GOOGLE_GEMINI_API_KEY_HERE"
    ```

    ***Note:*** *Keep your API key secure and do not share it publicly in your code or commits. For production applications, it's recommended to use environment variables.*

### 6\. Run the Application

Once all the above steps are complete, you can run the application:

```bash
python main.py
```

This will launch the CustomTkinter GUI for the Smart Shopping Assistant.

-----

## 🚀 How to Use

1.  Upon launching, you'll be greeted by the main menu where you can choose between "Smart TV" and "Air Conditioner".
2.  Select your desired product category.
3.  For the chosen product, you'll be guided through a series of selections to define its features (e.g., OS, picture quality for TV; coil type, power consumption for AC).
4.  After making your selections, the application will display the predicted price and an AI-powered recommendation for the best suitable product based on your input.
5.  Click the "Restart" button to return to the main menu and explore other product predictions.

-----

## 🤝 Contributing

We welcome contributions to make this Smart Shopping Assistant even better\! If you have ideas for new features, bug fixes, or improvements, please feel free to:

1.  Fork this repository.
2.  Create a new branch (`git checkout -b feature/your-awesome-feature`).
3.  Make your changes and ensure the code adheres to best practices.
4.  Commit your changes (`git commit -m 'feat: Add amazing new feature'`).
5.  Push to the branch (`git push origin feature/your-awesome-feature`).
6.  Open a Pull Request, describing your changes in detail.

-----

## 📄 License

This project is open-sourced under the [MIT License](https://www.google.com/search?q=LICENSE). You are free to use, modify, and distribute this code, provided the original license and copyright notice are included.

-----

**Happy Shopping\!** 🛒✨
