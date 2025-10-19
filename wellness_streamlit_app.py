import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Wellness Insurance Calculator",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    }
    .goal-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    }
    .savings-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
        text-align: center;
    }
    .warning-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    }
    h1 {
        color: #2d3748;
        font-weight: 700;
    }
    h2 {
        color: #4a5568;
        font-weight: 600;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODELS & CONFIGURATIONS
# ============================================
@st.cache_resource
def load_models():
    """Load all saved models and configurations"""
    models = {}
    
    try:
        # Load individual models
        model_names = ['LightGBM', 'CatBoost', 'TabNet']
        for name in model_names:
            with open(f'saved_models/{name}_final_model.pkl', 'rb') as f:
                models[name] = pickle.load(f)
        
        # Load meta-learner for stacking
        with open('saved_models/meta_learner_ridge.pkl', 'rb') as f:
            models['meta_learner'] = pickle.load(f)
        
        # Load feature names
        with open('saved_models/feature_names.json', 'r') as f:
            models['feature_names'] = json.load(f)
        
        # Load ensemble weights
        with open('analysis/ensemble_configurations.json', 'r') as f:
            models['ensemble_config'] = json.load(f)
        
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

# Initialize models
if 'models' not in st.session_state:
    with st.spinner("🔄 Loading AI models..."):
        st.session_state.models = load_models()

# ============================================
# FEATURE ENGINEERING FUNCTION
# ============================================
def create_features(age, sex, bmi, children, smoker, region):
    """Create all engineered features from input"""
    
    # Basic features
    features = {
        'age': age,
        'bmi': bmi,
        'children': children
    }
    
    # Encoded features
    features['sex_male'] = 1 if sex == 'Male' else 0
    features['smoker_yes'] = 1 if smoker == 'Yes' else 0
    
    # Region encoding (one-hot, drop first)
    regions = ['northeast', 'northwest', 'southeast', 'southwest']
    region_lower = region.lower()
    for r in regions[1:]:  # drop first
        features[f'region_{r}'] = 1 if region_lower == r else 0
    
    # Age features
    features['age_squared'] = age ** 2
    features['age_cubed'] = age ** 3
    features['age_sqrt'] = np.sqrt(age)
    features['age_log'] = np.log1p(age)
    features['age_group_young'] = 1 if age < 25 else 0
    features['age_group_adult'] = 1 if 25 <= age < 40 else 0
    features['age_group_middle'] = 1 if 40 <= age < 55 else 0
    features['age_group_senior'] = 1 if age >= 55 else 0
    
    # BMI features
    features['bmi_squared'] = bmi ** 2
    features['bmi_cubed'] = bmi ** 3
    features['bmi_sqrt'] = np.sqrt(bmi)
    features['bmi_log'] = np.log1p(bmi)
    features['bmi_underweight'] = 1 if bmi < 18.5 else 0
    features['bmi_normal'] = 1 if 18.5 <= bmi < 25 else 0
    features['bmi_overweight'] = 1 if 25 <= bmi < 30 else 0
    features['bmi_obese'] = 1 if bmi >= 30 else 0
    features['bmi_very_obese'] = 1 if bmi >= 35 else 0
    
    # Children features
    features['has_children'] = 1 if children > 0 else 0
    features['family_size'] = children + 2
    features['children_squared'] = children ** 2
    features['large_family'] = 1 if children >= 3 else 0
    
    # Smoker interactions
    smoker_val = features['smoker_yes']
    features['smoker_age'] = smoker_val * age
    features['smoker_bmi'] = smoker_val * bmi
    features['smoker_children'] = smoker_val * children
    features['smoker_age_squared'] = smoker_val * features['age_squared']
    features['smoker_age_cubed'] = smoker_val * features['age_cubed']
    features['smoker_bmi_squared'] = smoker_val * features['bmi_squared']
    features['smoker_bmi_cubed'] = smoker_val * features['bmi_cubed']
    features['smoker_age_bmi'] = smoker_val * age * bmi
    features['smoker_age_bmi_squared'] = smoker_val * age * features['bmi_squared']
    features['smoker_age_squared_bmi'] = smoker_val * features['age_squared'] * bmi
    features['smoker_obese'] = smoker_val * features['bmi_obese']
    features['smoker_very_obese'] = smoker_val * features['bmi_very_obese']
    features['smoker_senior'] = smoker_val * features['age_group_senior']
    features['smoker_middle_obese'] = smoker_val * features['age_group_middle'] * features['bmi_obese']
    
    # Non-smoker features
    nonsmoker_val = 1 - smoker_val
    features['nonsmoker_age'] = nonsmoker_val * age
    features['nonsmoker_bmi'] = nonsmoker_val * bmi
    features['nonsmoker_age_bmi'] = nonsmoker_val * age * bmi
    
    # Complex interactions
    features['age_bmi'] = age * bmi
    features['age_bmi_squared'] = age * features['bmi_squared']
    features['age_squared_bmi'] = features['age_squared'] * bmi
    features['age_children'] = age * children
    features['bmi_children'] = bmi * children
    features['age_bmi_children'] = age * bmi * children
    
    # Risk scores
    features['risk_score_1'] = smoker_val * age * bmi
    features['risk_score_2'] = smoker_val * (age / 10) * (bmi / 10) * 100
    features['risk_score_3'] = smoker_val * np.log1p(age) * np.log1p(bmi) * 10
    features['health_index'] = (age * bmi) / (1 + smoker_val * 10)
    features['ultra_high_risk'] = 1 if (smoker_val == 1 and age > 45 and bmi > 32) else 0
    features['moderate_risk'] = 1 if (smoker_val == 1 and age > 35 and bmi > 28) else 0
    
    return features

def predict_cost(features_dict, models):
    """Predict insurance cost using ensemble"""
    # Create DataFrame with correct column order
    feature_names = models['feature_names']
    X = pd.DataFrame([features_dict])[feature_names]
    
    # Get predictions from all models
    predictions = {}
    
    # LightGBM
    predictions['LightGBM'] = models['LightGBM'].predict(X)[0]
    
    # CatBoost
    predictions['CatBoost'] = models['CatBoost'].predict(X)[0]
    
    # TabNet
    predictions['TabNet'] = models['TabNet'].predict(X.values)[0][0]
    
    # Optimized weighted ensemble
    weights = models['ensemble_config']['optimized_weights']['weights']
    ensemble_pred = sum(predictions[m] * weights[m] for m in predictions.keys())
    
    return max(ensemble_pred, 0), predictions

# ============================================
# SIDEBAR - USER INPUTS
# ============================================
st.sidebar.title("🏥 Your Health Profile")
st.sidebar.markdown("---")

# Personal Info
st.sidebar.subheader("👤 Personal Information")
age = st.sidebar.slider("Age", 18, 100, 35, help="Your current age")
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
region = st.sidebar.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])

st.sidebar.markdown("---")

# Health Metrics
st.sidebar.subheader("📊 Health Metrics")
weight_kg = st.sidebar.number_input("Weight (kg)", 40.0, 200.0, 75.0, 0.1)
height_cm = st.sidebar.number_input("Height (cm)", 140.0, 220.0, 170.0, 0.1)
bmi = weight_kg / ((height_cm / 100) ** 2)
st.sidebar.metric("Your BMI", f"{bmi:.1f}")

# BMI Category
if bmi < 18.5:
    bmi_category = "Underweight"
    bmi_color = "🔵"
elif bmi < 25:
    bmi_category = "Normal"
    bmi_color = "🟢"
elif bmi < 30:
    bmi_category = "Overweight"
    bmi_color = "🟡"
else:
    bmi_category = "Obese"
    bmi_color = "🔴"

st.sidebar.caption(f"{bmi_color} {bmi_category}")

st.sidebar.markdown("---")

# Lifestyle
st.sidebar.subheader("🚬 Lifestyle")
smoker = st.sidebar.radio("Do you smoke?", ["No", "Yes"])
children = st.sidebar.number_input("Number of Children", 0, 10, 0)

# ============================================
# MAIN APP
# ============================================

# Header
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.title("🏃‍♂️ Wellness Insurance Calculator")
    st.markdown("### *Transform Your Health, Transform Your Costs*")

st.markdown("---")

# Calculate current prediction
if st.session_state.models:
    current_features = create_features(age, sex, bmi, children, smoker, region)
    current_cost, model_preds = predict_cost(current_features, st.session_state.models)
    
    # ============================================
    # CURRENT COST SECTION
    # ============================================
    st.header("💰 Your Current Annual Insurance Cost")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style='color: #667eea; margin: 0;'>Estimated Cost</h3>
            <h1 style='color: #2d3748; margin: 10px 0;'>${current_cost:,.0f}</h1>
            <p style='color: #718096; margin: 0;'>per year</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        monthly_cost = current_cost / 12
        st.markdown(f"""
        <div class="metric-card">
            <h3 style='color: #38a169; margin: 0;'>Monthly Payment</h3>
            <h1 style='color: #2d3748; margin: 10px 0;'>${monthly_cost:,.0f}</h1>
            <p style='color: #718096; margin: 0;'>per month</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        daily_cost = current_cost / 365
        st.markdown(f"""
        <div class="metric-card">
            <h3 style='color: #d69e2e; margin: 0;'>Daily Cost</h3>
            <h1 style='color: #2d3748; margin: 10px 0;'>${daily_cost:,.2f}</h1>
            <p style='color: #718096; margin: 0;'>per day</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Model Predictions Breakdown
    with st.expander("🤖 See AI Model Predictions"):
        model_df = pd.DataFrame({
            'Model': list(model_preds.keys()),
            'Prediction': [f"${v:,.0f}" for v in model_preds.values()]
        })
        st.dataframe(model_df, use_container_width=True)
    
    st.markdown("---")
    
    # ============================================
    # WELLNESS GOALS SECTION
    # ============================================
    st.header("🎯 Your Wellness Goals & Potential Savings")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🚭 Quit Smoking", "⚖️ Weight Management", "👨‍👩‍👧‍👦 Family Planning", "📊 Compare All"])
    
    # TAB 1: QUIT SMOKING
    with tab1:
        if smoker == "Yes":
            st.subheader("💪 Imagine If You Quit Smoking...")
            
            nonsmoker_features = create_features(age, sex, bmi, children, "No", region)
            nonsmoker_cost, _ = predict_cost(nonsmoker_features, st.session_state.models)
            smoking_savings = current_cost - nonsmoker_cost
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="savings-card">
                    <h2 style='margin: 0;'>💰 Annual Savings</h2>
                    <h1 style='font-size: 3em; margin: 20px 0;'>${smoking_savings:,.0f}</h1>
                    <p style='font-size: 1.2em;'>That's ${smoking_savings/12:,.0f} per month!</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="goal-card">
                    <h3>🎉 Over 10 Years</h3>
                    <h2>${smoking_savings * 10:,.0f} saved!</h2>
                    <p>You could buy a car, vacation home, or fund your retirement!</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Smoking impact visualization
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    name='Current (Smoker)',
                    x=['Annual Cost'],
                    y=[current_cost],
                    marker_color='#ff6b6b',
                    text=[f'${current_cost:,.0f}'],
                    textposition='auto',
                ))
                
                fig.add_trace(go.Bar(
                    name='As Non-Smoker',
                    x=['Annual Cost'],
                    y=[nonsmoker_cost],
                    marker_color='#51cf66',
                    text=[f'${nonsmoker_cost:,.0f}'],
                    textposition='auto',
                ))
                
                fig.update_layout(
                    title="Smoking Impact on Insurance Cost",
                    yaxis_title="Annual Cost ($)",
                    barmode='group',
                    height=400,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Motivational timeline
            st.subheader("📅 Your Savings Timeline")
            years = [1, 5, 10, 20, 30]
            savings_timeline = [smoking_savings * y for y in years]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=years,
                y=savings_timeline,
                mode='lines+markers',
                line=dict(color='#38ef7d', width=4),
                marker=dict(size=12, color='#11998e'),
                fill='tozeroy',
                fillcolor='rgba(17, 153, 142, 0.2)'
            ))
            
            fig.update_layout(
                title="Cumulative Savings From Quitting Smoking",
                xaxis_title="Years",
                yaxis_title="Total Savings ($)",
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Action items
            st.success("🌟 **Ready to Start?** Here are resources to help you quit:")
            st.markdown("""
            - 📞 Call 1-800-QUIT-NOW (National Quitline)
            - 💊 Talk to your doctor about cessation medications
            - 🧘 Try meditation and stress management apps
            - 👥 Join a support group in your area
            - 📱 Download a quit-smoking app (Smoke Free, QuitNow)
            """)
            
        else:
            st.success("🎉 **Congratulations!** You're already a non-smoker. You're saving thousands of dollars annually!")
            st.markdown("""
            <div class="goal-card">
                <h3>💚 Keep It Up!</h3>
                <p>Staying smoke-free is one of the best decisions for your health and wallet.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # TAB 2: WEIGHT MANAGEMENT
    with tab2:
        st.subheader("⚖️ Weight Management Impact")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Set Your Goal BMI")
            target_bmi = st.slider("Target BMI", 18.5, 35.0, min(bmi, 24.9), 0.1)
            
            target_weight = target_bmi * ((height_cm / 100) ** 2)
            weight_change = target_weight - weight_kg
            
            if weight_change < 0:
                st.info(f"📉 Goal: Lose **{abs(weight_change):.1f} kg** ({abs(weight_change * 2.205):.1f} lbs)")
            elif weight_change > 0:
                st.info(f"📈 Goal: Gain **{weight_change:.1f} kg** ({weight_change * 2.205:.1f} lbs)")
            else:
                st.success("✅ You're at your target weight!")
        
        with col2:
            # Calculate cost at target BMI
            target_features = create_features(age, sex, target_bmi, children, smoker, region)
            target_cost, _ = predict_cost(target_features, st.session_state.models)
            bmi_savings = current_cost - target_cost
            
            if bmi_savings > 0:
                st.markdown(f"""
                <div class="savings-card">
                    <h3>💰 Potential Annual Savings</h3>
                    <h1 style='font-size: 2.5em;'>${bmi_savings:,.0f}</h1>
                    <p>By reaching your target BMI</p>
                </div>
                """, unsafe_allow_html=True)
            elif bmi_savings < 0:
                st.markdown(f"""
                <div class="warning-card">
                    <h3>⚠️ Cost Increase</h3>
                    <h2>${abs(bmi_savings):,.0f}</h2>
                    <p>Your current BMI is already optimal for insurance costs</p>
                </div>
                """, unsafe_allow_html=True)
        
        # BMI range comparison
        st.markdown("### 📊 Cost Across BMI Range")
        
        bmi_range = np.linspace(18.5, 40, 50)
        costs_by_bmi = []
        
        for test_bmi in bmi_range:
            test_features = create_features(age, sex, test_bmi, children, smoker, region)
            test_cost, _ = predict_cost(test_features, st.session_state.models)
            costs_by_bmi.append(test_cost)
        
        fig = go.Figure()
        
        # Line plot
        fig.add_trace(go.Scatter(
            x=bmi_range,
            y=costs_by_bmi,
            mode='lines',
            name='Insurance Cost',
            line=dict(color='#667eea', width=3),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.2)'
        ))
        
        # Current BMI marker
        fig.add_trace(go.Scatter(
            x=[bmi],
            y=[current_cost],
            mode='markers',
            name='Your Current BMI',
            marker=dict(size=15, color='#ff6b6b', symbol='star')
        ))
        
        # Target BMI marker
        if abs(target_bmi - bmi) > 0.5:
            fig.add_trace(go.Scatter(
                x=[target_bmi],
                y=[target_cost],
                mode='markers',
                name='Your Target BMI',
                marker=dict(size=15, color='#51cf66', symbol='star')
            ))
        
        # BMI category zones
        fig.add_vrect(x0=18.5, x1=25, fillcolor="green", opacity=0.1, line_width=0, annotation_text="Normal", annotation_position="top left")
        fig.add_vrect(x0=25, x1=30, fillcolor="yellow", opacity=0.1, line_width=0, annotation_text="Overweight", annotation_position="top left")
        fig.add_vrect(x0=30, x1=40, fillcolor="red", opacity=0.1, line_width=0, annotation_text="Obese", annotation_position="top left")
        
        fig.update_layout(
            title="How BMI Affects Your Insurance Cost",
            xaxis_title="BMI",
            yaxis_title="Annual Cost ($)",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Weight loss tips
        if bmi > 25:
            st.info("💡 **Healthy Weight Loss Tips:**")
            st.markdown("""
            - 🥗 Eat more whole foods, fruits, and vegetables
            - 🏃‍♀️ Aim for 150 minutes of moderate exercise per week
            - 💧 Drink plenty of water
            - 😴 Get 7-9 hours of quality sleep
            - 📝 Track your food intake with an app (MyFitnessPal, Lose It!)
            - 👨‍⚕️ Consult with a nutritionist or dietitian
            """)
    
    # TAB 3: FAMILY PLANNING
    with tab3:
        st.subheader("👨‍👩‍👧‍👦 Family Planning Impact")
        
        st.markdown("### 📊 Cost by Number of Children")
        
        children_range = range(0, 6)
        costs_by_children = []
        
        for num_children in children_range:
            child_features = create_features(age, sex, bmi, num_children, smoker, region)
            child_cost, _ = predict_cost(child_features, st.session_state.models)
            costs_by_children.append(child_cost)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=list(children_range),
            y=costs_by_children,
            marker_color=['#667eea' if i == children else '#cbd5e0' for i in children_range],
            text=[f'${c:,.0f}' for c in costs_by_children],
            textposition='auto',
        ))
        
        fig.update_layout(
            title="Insurance Cost by Number of Children",
            xaxis_title="Number of Children",
            yaxis_title="Annual Cost ($)",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Per-child cost
        if children > 0:
            base_cost = costs_by_children[0]
            total_child_cost = current_cost - base_cost
            per_child_cost = total_child_cost / children
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Base Cost (No Children)", f"${base_cost:,.0f}")
            col2.metric("Total Child Impact", f"${total_child_cost:,.0f}")
            col3.metric("Cost Per Child", f"${per_child_cost:,.0f}")
    
    # TAB 4: COMPARE ALL
    with tab4:
        st.subheader("📊 Complete What-If Analysis")
        
        # Create scenarios
        scenarios = {
            'Current You': current_cost,
        }
        
        if smoker == "Yes":
            nonsmoker_features = create_features(age, sex, bmi, children, "No", region)
            nonsmoker_cost, _ = predict_cost(nonsmoker_features, st.session_state.models)
            scenarios['If You Quit Smoking'] = nonsmoker_cost
        
        # Optimal BMI
        optimal_bmi = 22.0
        optimal_features = create_features(age, sex, optimal_bmi, children, smoker, region)
        optimal_bmi_cost, _ = predict_cost(optimal_features, st.session_state.models)
        scenarios['At Optimal BMI (22)'] = optimal_bmi_cost
        
        # Best case scenario
        best_features = create_features(age, sex, optimal_bmi, children, "No", region)
        best_cost, _ = predict_cost(best_features, st.session_state.models)
        scenarios['🌟 Best Case (Non-smoker + Optimal BMI)'] = best_cost
        
        # Visualization
        scenario_df = pd.DataFrame({
            'Scenario': list(scenarios.keys()),
            'Annual Cost': list(scenarios.values()),
            'Savings': [current_cost - cost for cost in scenarios.values()]
        })
        
        fig = go.Figure()
        
        colors = ['#ff6b6b', '#ffd93d', '#6bcf7f', '#51cf66']
        
        fig.add_trace(go.Bar(
            y=scenario_df['Scenario'],
            x=scenario_df['Annual Cost'],
            orientation='h',
            marker_color=colors[:len(scenarios)],
            text=[f'${c:,.0f}' for c in scenario_df['Annual Cost']],
            textposition='auto',
        ))
        
        fig.update_layout(
            title="Compare All Scenarios",
            xaxis_title="Annual Cost ($)",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Savings table
        st.markdown("### 💰 Potential Savings Breakdown")
        savings_df = scenario_df[scenario_df['Savings'] > 0].copy()
        savings_df['Monthly Savings'] = savings_df['Savings'] / 12
        savings_df['10-Year Savings'] = savings_df['Savings'] * 10
        
        st.dataframe(
            savings_df.style.format({
                'Annual Cost': '${:,.0f}',
                'Savings': '${:,.0f}',
                'Monthly Savings': '${:,.0f}',
                '10-Year Savings': '${:,.0f}'
            }),
            use_container_width=True
        )
        
        # Maximum potential savings
        max_savings = current_cost - best_cost
        if max_savings > 0:
            st.markdown(f"""
            <div class="savings-card">
                <h2>🎯 Your Maximum Potential Savings</h2>
                <h1 style='font-size: 3em;'>${max_savings:,.0f}/year</h1>
                <p style='font-size: 1.2em;'>That's ${max_savings * 10:,.0f} over 10 years!</p>
                <p>By becoming a non-smoker and maintaining optimal BMI</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ============================================
    # HEALTH INSIGHTS SECTION
    # ============================================
    st.header("🔍 Your Health Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Risk Assessment")
        
        # Calculate risk factors
        risk_factors = []
        risk_score = 0
        
        if smoker == "Yes":
            risk_factors.append("🚬 Smoking (High Impact)")
            risk_score += 40
        
        if bmi >= 30:
            risk_factors.append("⚖️ BMI in Obese Range (High Impact)")
            risk_score += 30
        elif bmi >= 25:
            risk_factors.append("⚖️ BMI in Overweight Range (Moderate Impact)")
            risk_score += 15
        
        if age >= 55:
            risk_factors.append("👴 Senior Age Group (Moderate Impact)")
            risk_score += 20
        elif age >= 40:
            risk_factors.append("🧑 Middle Age Group (Low Impact)")
            risk_score += 10
        
        if smoker == "Yes" and bmi >= 30:
            risk_factors.append("⚠️ Combined Smoking + Obesity (Very High Impact)")
            risk_score += 20
        
        if risk_factors:
            st.warning("**Identified Risk Factors:**")
            for factor in risk_factors:
                st.markdown(f"- {factor}")
        else:
            st.success("✅ **Great news!** No major risk factors identified.")
        
        # Risk score gauge
        if risk_score > 0:
            if risk_score >= 70:
                risk_level = "High"
                risk_color = "#ff6b6b"
            elif risk_score >= 40:
                risk_level = "Moderate"
                risk_color = "#ffd93d"
            else:
                risk_level = "Low"
                risk_color = "#51cf66"
            
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, {risk_color}33 0%, {risk_color}11 100%);">
                <h3>Risk Level: <span style="color: {risk_color}; font-weight: bold;">{risk_level}</span></h3>
                <div style="background: #e2e8f0; border-radius: 10px; height: 20px; overflow: hidden;">
                    <div style="background: {risk_color}; width: {risk_score}%; height: 100%;"></div>
                </div>
                <p style="margin-top: 10px;">Risk Score: {risk_score}/100</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("💪 Personalized Recommendations")
        
        recommendations = []
        
        if smoker == "Yes":
            recommendations.append({
                'icon': '🚭',
                'title': 'Quit Smoking',
                'desc': f'Save ${current_cost - nonsmoker_cost:,.0f}/year',
                'priority': 'High'
            })
        
        if bmi >= 30:
            recommendations.append({
                'icon': '🏃',
                'title': 'Weight Management Program',
                'desc': 'Aim for BMI under 30',
                'priority': 'High'
            })
        elif bmi >= 25:
            recommendations.append({
                'icon': '🥗',
                'title': 'Healthy Eating Plan',
                'desc': 'Maintain or reduce BMI',
                'priority': 'Medium'
            })
        
        if age >= 40:
            recommendations.append({
                'icon': '🏥',
                'title': 'Regular Health Screenings',
                'desc': 'Annual check-ups recommended',
                'priority': 'Medium'
            })
        
        recommendations.append({
            'icon': '🧘',
            'title': 'Stress Management',
            'desc': 'Meditation, yoga, or therapy',
            'priority': 'Medium'
        })
        
        recommendations.append({
            'icon': '💤',
            'title': 'Quality Sleep',
            'desc': '7-9 hours per night',
            'priority': 'Medium'
        })
        
        if recommendations:
            for rec in recommendations:
                priority_color = '#ff6b6b' if rec['priority'] == 'High' else '#ffd93d'
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid {priority_color};">
                    <h4>{rec['icon']} {rec['title']}</h4>
                    <p style="color: #718096; margin: 5px 0;">{rec['desc']}</p>
                    <span style="background: {priority_color}33; padding: 3px 8px; border-radius: 5px; font-size: 0.8em; color: {priority_color}; font-weight: bold;">{rec['priority']} Priority</span>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ============================================
    # PROGRESS TRACKER
    # ============================================
    st.header("📅 Your Wellness Journey Tracker")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Set Your Goals")
        
        goal_col1, goal_col2 = st.columns(2)
        
        with goal_col1:
            goal_date = st.date_input(
                "Target Date",
                value=pd.to_datetime('today') + pd.DateOffset(months=6),
                min_value=pd.to_datetime('today')
            )
        
        with goal_col2:
            goal_type = st.selectbox(
                "Primary Goal",
                ["Quit Smoking", "Lose Weight", "Maintain Health", "All of the Above"]
            )
        
        # Calculate days until goal
        days_until_goal = (pd.to_datetime(goal_date) - pd.to_datetime('today')).days
        
        st.info(f"🎯 **{days_until_goal} days** until your target date!")
        
        # Milestone calculator
        if goal_type == "Quit Smoking" and smoker == "Yes":
            smoking_savings = current_cost - nonsmoker_cost
            daily_savings = smoking_savings / 365
            
            milestones = [
                (7, "1 Week Smoke-Free", daily_savings * 7),
                (30, "1 Month Smoke-Free", daily_savings * 30),
                (90, "3 Months Smoke-Free", daily_savings * 90),
                (180, "6 Months Smoke-Free", daily_savings * 180),
                (365, "1 Year Smoke-Free", daily_savings * 365)
            ]
            
            st.markdown("### 🏆 Savings Milestones")
            for days, milestone, savings in milestones:
                if days <= days_until_goal:
                    st.markdown(f"- **{milestone}**: ${savings:,.0f} saved")
        
        elif goal_type == "Lose Weight":
            if target_bmi < bmi:
                weight_to_lose = weight_kg - target_weight
                weekly_loss = 0.5  # kg per week (healthy rate)
                weeks_needed = int(weight_to_lose / weekly_loss)
                
                st.markdown("### 📉 Weight Loss Timeline")
                st.markdown(f"- **Total to lose**: {weight_to_lose:.1f} kg ({weight_to_lose * 2.205:.1f} lbs)")
                st.markdown(f"- **At 0.5 kg/week**: ~{weeks_needed} weeks")
                st.markdown(f"- **Expected completion**: {pd.to_datetime('today') + pd.DateOffset(weeks=weeks_needed):%B %d, %Y}")
                
                if weeks_needed <= days_until_goal / 7:
                    st.success("✅ Your timeline is realistic!")
                else:
                    st.warning(f"⚠️ Consider extending your goal to {weeks_needed} weeks for healthy weight loss.")
    
    with col2:
        st.subheader("Quick Tips")
        
        st.markdown("""
        <div class="goal-card">
            <h4>🎯 SMART Goals</h4>
            <ul style="text-align: left;">
                <li><b>S</b>pecific</li>
                <li><b>M</b>easurable</li>
                <li><b>A</b>chievable</li>
                <li><b>R</b>elevant</li>
                <li><b>T</b>ime-bound</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <h4>📱 Recommended Apps</h4>
            <ul style="text-align: left; font-size: 0.9em;">
                <li>MyFitnessPal - Calorie tracking</li>
                <li>Strava - Exercise tracking</li>
                <li>Headspace - Meditation</li>
                <li>Smoke Free - Quit smoking</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ============================================
    # EDUCATIONAL CONTENT
    # ============================================
    st.header("📚 Learn More About Your Health")
    
    edu_tab1, edu_tab2, edu_tab3 = st.tabs(["💡 BMI Explained", "🚬 Smoking Impact", "💰 Insurance 101"])
    
    with edu_tab1:
        st.subheader("Understanding Body Mass Index (BMI)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **What is BMI?**
            
            BMI is a measure of body fat based on height and weight. It's calculated as:
            
            `BMI = weight (kg) / [height (m)]²`
            
            **BMI Categories:**
            - 🔵 **Underweight**: < 18.5
            - 🟢 **Normal**: 18.5 - 24.9
            - 🟡 **Overweight**: 25.0 - 29.9
            - 🔴 **Obese**: ≥ 30.0
            
            **Why It Matters:**
            
            Higher BMI is associated with increased health risks including heart disease, diabetes, and higher insurance costs.
            """)
        
        with col2:
            # BMI distribution visualization
            bmi_categories = ['Underweight\n<18.5', 'Normal\n18.5-24.9', 'Overweight\n25-29.9', 'Obese\n≥30']
            avg_costs = []
            
            for bmi_val in [17, 22, 27.5, 32]:
                test_features = create_features(age, sex, bmi_val, children, smoker, region)
                test_cost, _ = predict_cost(test_features, st.session_state.models)
                avg_costs.append(test_cost)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=bmi_categories,
                y=avg_costs,
                marker_color=['#3b82f6', '#10b981', '#f59e0b', '#ef4444'],
                text=[f'${c:,.0f}' for c in avg_costs],
                textposition='auto',
            ))
            
            fig.update_layout(
                title="Average Insurance Cost by BMI Category",
                xaxis_title="BMI Category",
                yaxis_title="Annual Cost ($)",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with edu_tab2:
        st.subheader("The True Cost of Smoking")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Health Risks:**
            - 🫁 Lung cancer and COPD
            - ❤️ Heart disease and stroke
            - 🦴 Weakened immune system
            - 👶 Pregnancy complications
            - 🧠 Reduced cognitive function
            
            **Financial Impact:**
            
            Smokers pay **2-3x more** for health insurance on average. This is because smoking is the leading cause of preventable death and disease.
            """)
            
            if smoker == "Yes":
                st.error("""
                **Your Smoking Cost:**
                
                Beyond insurance, consider:
                - Cigarette costs: ~$2,000-3,000/year
                - Higher life insurance premiums
                - Increased healthcare costs
                - Lost productivity
                """)
        
        with col2:
            # Smoking cost breakdown
            if smoker == "Yes":
                smoking_total = current_cost - nonsmoker_cost
                cigarette_cost = 2500  # Average annual cost
                total_smoking_cost = smoking_total + cigarette_cost
                
                fig = go.Figure()
                
                fig.add_trace(go.Pie(
                    labels=['Insurance Premium', 'Cigarettes', 'Other Health Costs'],
                    values=[smoking_total, cigarette_cost, cigarette_cost * 0.5],
                    marker_colors=['#ff6b6b', '#ffd93d', '#ff9999'],
                    hole=0.4
                ))
                
                fig.update_layout(
                    title="Annual Cost of Smoking",
                    annotations=[dict(text=f'${total_smoking_cost:,.0f}', x=0.5, y=0.5, font_size=20, showarrow=False)],
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("""
                ✅ **You're saving thousands annually by not smoking!**
                
                Non-smokers benefit from:
                - Lower insurance premiums
                - Better overall health
                - Longer life expectancy
                - More disposable income
                """)
    
    with edu_tab3:
        st.subheader("How Insurance Pricing Works")
        
        st.markdown("""
        **Key Factors That Affect Your Premium:**
        
        1. **Age** 👴
           - Premiums increase with age
           - Older individuals typically have more health issues
        
        2. **Smoking Status** 🚬
           - Single biggest factor (can double or triple premiums)
           - Tobacco use tracked for 12+ months
        
        3. **BMI** ⚖️
           - Higher BMI = higher premiums
           - Obesity linked to chronic conditions
        
        4. **Location** 📍
           - Regional healthcare costs vary
           - State regulations affect pricing
        
        5. **Family Size** 👨‍👩‍👧‍👦
           - More dependents = higher premiums
           - Family plans offer economies of scale
        
        **How to Lower Your Premium:**
        - ✅ Maintain healthy weight
        - ✅ Don't smoke or quit smoking
        - ✅ Shop around for best rates
        - ✅ Consider higher deductible plans
        - ✅ Take advantage of wellness programs
        """)
        
        # Cost factors breakdown
        st.markdown("### 📊 Your Premium Breakdown")

        # Calculate individual factor impacts
        base_features = create_features(30, sex, 22, 0, "No", region)
        base_cost, _ = predict_cost(base_features, st.session_state.models)

        factors = {
            'Base Premium': base_cost,
        }

        # Only add factors that are different from baseline
        if age != 30:
            age_30_features = create_features(30, sex, bmi, children, smoker, region)
            age_30_cost, _ = predict_cost(age_30_features, st.session_state.models)
            factors['Age Factor'] = current_cost - age_30_cost

        if bmi != 22:
            bmi_22_features = create_features(age, sex, 22, children, smoker, region)
            bmi_22_cost, _ = predict_cost(bmi_22_features, st.session_state.models)
            factors['BMI Factor'] = current_cost - bmi_22_cost

        if smoker == "Yes":
            no_smoke_features = create_features(age, sex, bmi, children, "No", region)
            no_smoke_cost, _ = predict_cost(no_smoke_features, st.session_state.models)
            factors['Smoking Factor'] = current_cost - no_smoke_cost

        if children > 0:
            no_children_features = create_features(age, sex, bmi, 0, smoker, region)
            no_children_cost, _ = predict_cost(no_children_features, st.session_state.models)
            factors['Family Factor'] = current_cost - no_children_cost
        
        # Remove zero factors
        factors = {k: v for k, v in factors.items() if abs(v) > 10}
        
        if len(factors) > 1:
            fig = go.Figure()
            
            fig.add_trace(go.Waterfall(
                name="Premium Components",
                orientation="v",
                measure=["absolute"] + ["relative"] * (len(factors) - 1),
                x=list(factors.keys()),
                y=list(factors.values()),
                connector={"line": {"color": "rgb(63, 63, 63)"}},
            ))
            
            fig.update_layout(
                title="How Your Premium is Calculated",
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ============================================
    # EXPORT & SHARE
    # ============================================
    st.header("💾 Save Your Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Generate report data
        report_data = {
            'Date': datetime.now().strftime('%Y-%m-%d'),
            'Age': age,
            'Sex': sex,
            'BMI': round(bmi, 1),
            'Weight (kg)': weight_kg,
            'Height (cm)': height_cm,
            'Children': children,
            'Smoker': smoker,
            'Region': region,
            'Annual Cost': f"${current_cost:,.0f}",
            'Monthly Cost': f"${current_cost/12:,.0f}",
            'Daily Cost': f"${current_cost/365:,.2f}"
        }
        
        if smoker == "Yes":
            report_data['Potential Savings (Quit Smoking)'] = f"${smoking_savings:,.0f}"
        
        report_df = pd.DataFrame([report_data])
        csv = report_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download CSV Report",
            data=csv,
            file_name=f"insurance_estimate_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Generate summary text
        summary_text = f"""
WELLNESS INSURANCE CALCULATOR REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

═══════════════════════════════════════
YOUR PROFILE
═══════════════════════════════════════
Age: {age} years
Sex: {sex}
BMI: {bmi:.1f} ({bmi_category})
Weight: {weight_kg} kg
Height: {height_cm} cm
Children: {children}
Smoker: {smoker}
Region: {region}

═══════════════════════════════════════
COST ESTIMATE
═══════════════════════════════════════
Annual Cost: ${current_cost:,.0f}
Monthly Cost: ${current_cost/12:,.0f}
Daily Cost: ${current_cost/365:,.2f}

═══════════════════════════════════════
POTENTIAL SAVINGS
═══════════════════════════════════════
"""
        
        if smoker == "Yes":
            summary_text += f"Quit Smoking: ${smoking_savings:,.0f}/year\n"
        
        if bmi != 22:
            bmi_impact = current_cost - optimal_bmi_cost
            if bmi_impact > 0:
                summary_text += f"Reach Optimal BMI: ${bmi_impact:,.0f}/year\n"
        
        summary_text += f"\nBest Case Scenario: ${best_cost:,.0f}/year\nMax Savings: ${max_savings:,.0f}/year\n"
        
        st.download_button(
            label="📄 Download Text Report",
            data=summary_text,
            file_name=f"insurance_summary_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>📧 Share Results</h4>
            <p style="font-size: 0.9em; color: #718096;">
            Download your report and share it with:
            </p>
            <ul style="font-size: 0.9em; text-align: left;">
                <li>Financial advisor</li>
                <li>Insurance broker</li>
                <li>Family members</li>
                <li>Healthcare provider</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ============================================
    # FOOTER
    # ============================================
    st.markdown("""
    <div style="text-align: center; padding: 30px; background: white; border-radius: 10px; margin-top: 20px;">
        <h3 style="color: #667eea;">🌟 Start Your Wellness Journey Today!</h3>
        <p style="color: #718096; font-size: 1.1em;">
        Small changes lead to big savings. Take the first step towards a healthier, 
        more affordable future.
        </p>
        <p style="color: #a0aec0; font-size: 0.9em; margin-top: 20px;">
        <b>Disclaimer:</b> This calculator provides estimates based on AI models trained on historical data. 
        Actual insurance premiums may vary based on specific policy terms, underwriting guidelines, 
        and individual circumstances. Always consult with a licensed insurance professional for accurate quotes.
        </p>
        <p style="color: #a0aec0; font-size: 0.8em; margin-top: 10px;">
        Made with ❤️ for your health and wellness | © 2025
        </p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("❌ Unable to load AI models. Please ensure model files are in the correct directory.")
    st.info("""
    **Required files:**
    - `saved_models/LightGBM_final_model.pkl`
    - `saved_models/CatBoost_final_model.pkl`
    - `saved_models/TabNet_final_model.pkl`
    - `saved_models/meta_learner_ridge.pkl`
    - `saved_models/feature_names.json`
    - `analysis/ensemble_configurations.json`
    """)