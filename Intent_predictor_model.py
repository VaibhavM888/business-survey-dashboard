import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import warnings

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

def generate_market_data(n_samples=2000):
    """
    Generates synthetic data with STRONGER correlations for better calibration.
    """
    np.random.seed(42)
    
    # 1. Demographics
    ages = np.random.randint(18, 75, n_samples)
    incomes = np.random.randint(1, 14, n_samples) # 1=Low, 13=High
    trends = np.random.randint(1, 6, n_samples)   # 1=Low, 5=High
    
    # 2. Logic for PURCHASE INTENT (Y1)
    # Income is the biggest driver, Age has a sweet spot (30-50), Trend adds a boost
    # Base purchase probability around 15%
    base_intent = 15.0 
    
    # Income impact: Each level adds ~1.5%
    income_impact = (incomes * 1.5)
    
    # Age impact: People between 28 and 50 have higher intent
    age_impact = np.where((ages >= 28) & (ages <= 50), 5.0, -2.0)
    
    # Trend impact: High trend score adds significant intent
    trend_impact = (trends * 2.0)
    
    y_intent = base_intent + income_impact + age_impact + trend_impact + np.random.normal(0, 2, n_samples)
    y_intent = np.clip(y_intent, 0, 100) # Cap between 0-100%

    # 3. Logic for ECO-ADOPTION (Y2)
    # Age is negative driver (younger = higher), Trend is MASSIVE driver
    base_eco = 20.0
    
    # Age: Younger people get a bonus. For every year below 40, add score.
    age_factor = (40 - ages) * 0.4
    
    # Trend: This is the sensitivity fix. High trend score now multipliers heavily.
    # Score 5 adds ~25%, Score 1 adds ~5%
    trend_factor = (trends * 5.5) 
    
    # Income: Slight positive correlation (Eco products often cost more)
    income_factor = (incomes * 0.8)
    
    y_eco = base_eco + age_factor + trend_factor + income_factor + np.random.normal(0, 3, n_samples)
    y_eco = np.clip(y_eco, 0, 100)

    df = pd.DataFrame({
        'Age': ages,
        'Income_Code': incomes,
        'Trend_Score': trends,
        'Intent_Score': y_intent,
        'Eco_Score': y_eco
    })
    
    return df

def train_and_evaluate():
    print(">> [1/3] Generating High-Sensitivity Market Data...")
    df = generate_market_data()
    
    X = df[['Age', 'Income_Code', 'Trend_Score']]
    y_intent = df['Intent_Score']
    y_eco = df['Eco_Score']
    
    # Split data for validation
    X_train, X_test, yi_train, yi_test, ye_train, ye_test = train_test_split(X, y_intent, y_eco, test_size=0.2, random_state=42)
    
    print(">> [2/3] Training & Calibrating Models...")
    model_intent = LinearRegression()
    model_intent.fit(X_train, yi_train)
    
    model_eco = LinearRegression()
    model_eco.fit(X_train, ye_train)
    
    # Evaluate Accuracy
    acc_intent = r2_score(yi_test, model_intent.predict(X_test))
    acc_eco = r2_score(ye_test, model_eco.predict(X_test))
    
    # Market Averages
    avg_intent = df['Intent_Score'].mean()
    avg_eco = df['Eco_Score'].mean()
    
    print(">> [3/3] Calibration Complete.")
    print(f"   - Model Accuracy (Intent): {acc_intent:.2%} (R² Score)")
    print(f"   - Model Accuracy (Eco):    {acc_eco:.2%} (R² Score)")
    print(f"   - Market Avg Intent:       {avg_intent:.1f}%")
    print(f"   - Market Avg Eco-Adoption: {avg_eco:.1f}%")
    
    return model_intent, model_eco, avg_intent, avg_eco

def get_prediction(model_intent, model_eco, avg_intent, avg_eco):
    print("\n" + "="*60)
    print(" 🎯  CONSUMER PREDICTOR (v3.1 - Calibrated & Accuracy Check)  🎯")
    print("="*60)
    
    while True:
        try:
            print("\n--- Enter Persona Details ---")
            age = float(input("1. Age (e.g., 25): "))
            income = float(input("2. Income Code (1=Low, 7=Middle, 13=High): "))
            trend = float(input("3. Trend Score (1=Low, 3=Avg, 5=High): "))
            
            # Predict
            input_data = pd.DataFrame([[age, income, trend]], columns=['Age', 'Income_Code', 'Trend_Score'])
            
            pred_intent = model_intent.predict(input_data)[0]
            pred_eco = model_eco.predict(input_data)[0]
            
            # Interpretation Labels
            intent_label = "❄️ COLD LEAD"
            if pred_intent > avg_intent + 10: intent_label = "🔥 HOT LEAD"
            elif pred_intent > avg_intent: intent_label = "🤔 WARM LEAD"
            
            eco_label = "🛢️ TRADITIONAL"
            if pred_eco > avg_eco + 10: eco_label = "🌳 SUPER GREEN (Early Adopter)"
            elif pred_eco > avg_eco: eco_label = "🌿 ECO-FRIENDLY"
            
            print(f"\n" + "-"*40)
            print(f"PREDICTION FOR AGE {int(age)} | TREND {int(trend)}/5:")
            print(f"1. Purchase Intent:  {intent_label}")
            print(f"   (Score: {pred_intent:.1f}% vs Market Avg {avg_intent:.1f}%)")
            
            print(f"2. Likely Choice:    {eco_label}")
            print(f"   (Score: {pred_eco:.1f}% vs Market Avg {avg_eco:.1f}%)")
            print("-"*40)
            
            cont = input("\nTest another persona? (y/n): ")
            if cont.lower() != 'y':
                break
                
        except ValueError:
            print("Invalid input. Please enter numbers.")

if __name__ == "__main__":
    m_intent, m_eco, av_i, av_e = train_and_evaluate()
    get_prediction(m_intent, m_eco, av_i, av_e)