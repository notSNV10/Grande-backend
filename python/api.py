"""
FastAPI Service for AI-Powered Business Analytics
Replaces Flask endpoints with FastAPI
"""
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import mysql.connector
from mysql.connector import Error

# Import AI Agent
from ai_agent import AIAgent
from ai.openrouter_chat import chat_llm

# Import forecast functions
from forecast import create_lstm_model, create_dense_model, prepare_sequence_data, train_tensorflow_model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# TensorFlow imports with fallback
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    try:
        from tensorflow.keras.optimizers import Adam
    except ImportError:
        from keras.optimizers import Adam
    try:
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError:
        from keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
    tf.get_logger().setLevel('ERROR')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.set_visible_devices([], 'GPU')
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Database configuration - use Settings class for environment variable support
from config import get_settings
settings = get_settings()

# Legacy DB_CONFIG for backward compatibility (uses Settings)
DB_CONFIG = {
    'host': settings.db_host,
    'user': settings.db_user,
    'password': settings.db_password,
    'database': settings.db_name
}

app = FastAPI(title="Adarna Grande AI Analytics API")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "service": "Adarna Grande AI Analytics API",
        "status": "online",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict_sales": "/predict_sales?branch_id=<id>",
            "predict_inventory": "/predict_inventory?branch_id=<id>",
            "ai_agent": "/ai_agent (POST with query)",
            "docs": "/docs",
            "redoc": "/redoc"
        },
        "message": "FastAPI service is running. Visit /docs for interactive API documentation."
    }

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response models
class SalesForecastResponse(BaseModel):
    branch_id: int
    branch_name: Optional[str] = None
    dates: List[str]
    actual_sales: List[float]
    predicted_sales: List[float]
    growth_rate: float
    confidence: float
    insights: List[str]

class InventoryItem(BaseModel):
    item_name: str
    current_stock: int
    predicted_stock: int
    days_until_out: int
    recommendation: str
    is_overstocked: Optional[bool] = False
    overstock_multiplier: Optional[float] = None
    excess_stock: Optional[int] = None
    overstock_recommendation: Optional[str] = None

class InventoryForecastResponse(BaseModel):
    branch_id: int
    branch_name: Optional[str] = None
    items: List[InventoryItem]

class AIAgentQuery(BaseModel):
    query: str
    branch_id: Optional[int] = None
    user_role: Optional[str] = "admin"

class AIAgentResponse(BaseModel):
    response: str
    reasoning: str
    recommendations: List[str]
    basis: Dict[str, Any]
    interpretation: str
    data_summary: Optional[Dict[str, Any]] = None


class ChatBotRequest(BaseModel):
    """Simple chat request for conversational assistant."""
    message: str
    branch_id: Optional[int] = None
    user_role: Optional[str] = "owner"


class ChatBotResponse(BaseModel):
    """Simple chat response."""
    reply: str
    reasoning: str
    prompt_snapshot: Optional[Dict[str, Any]] = None


# New AI Task Models
class CompetitorPricingRequest(BaseModel):
    """Request for competitor pricing comparison."""
    branch_id: Optional[int] = None
    service_category: Optional[str] = None  # e.g., "haircut", "nail services"


class CompetitorPricingResponse(BaseModel):
    """Response with competitor pricing analysis."""
    our_pricing: Dict[str, Any]
    competitor_analysis: List[Dict[str, Any]]
    recommendations: List[str]
    ai_model_used: str
    fallback_used: bool


class ChartExplanationRequest(BaseModel):
    """Request for chart/graph explanation."""
    chart_type: str  # "sales", "inventory", "revenue"
    chart_data: Dict[str, Any]  # Chart data points
    branch_id: Optional[int] = None
    time_period: Optional[str] = None  # "weekly", "monthly", "yearly"


class ChartExplanationResponse(BaseModel):
    """Response with chart explanation."""
    explanation: str
    key_insights: List[str]
    trends: List[str]
    recommendations: List[str]
    ai_model_used: str
    fallback_used: bool


class KPINarrativeRequest(BaseModel):
    """Request for KPI narrative generation."""
    kpi_data: Dict[str, Any]  # KPI metrics
    branch_id: Optional[int] = None
    time_period: Optional[str] = None


class KPINarrativeResponse(BaseModel):
    """Response with KPI narrative."""
    narrative: str
    summary: str
    highlights: List[str]
    concerns: List[str]
    ai_model_used: str
    fallback_used: bool


class InventoryInsightRequest(BaseModel):
    """Request for inventory forecast reasoning."""
    branch_id: int
    item_name: Optional[str] = None  # Specific item or all items


class InventoryInsightResponse(BaseModel):
    """Response with inventory forecast insights."""
    insights: List[Dict[str, Any]]
    reasoning: str
    recommendations: List[str]
    ai_model_used: str
    fallback_used: bool


class InteractiveExplainRequest(BaseModel):
    """Request for interactive explanation."""
    question: str
    context_data: Optional[Dict[str, Any]] = None
    branch_id: Optional[int] = None
    user_role: Optional[str] = "owner"


class InteractiveExplainResponse(BaseModel):
    """Response with interactive explanation."""
    answer: str
    reasoning: str
    related_insights: List[str]
    ai_model_used: str
    fallback_used: bool

def get_db_connection():
    """Get database connection"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except Error as e:
        print(f"Database connection error: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

def get_branch_name(branch_id: int) -> Optional[str]:
    """Get branch name by ID"""
    if branch_id is None or branch_id <= 0:
        return None
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT branch_name FROM branches WHERE id = %s", (branch_id,))
        result = cursor.fetchone()
        return result['branch_name'] if result and 'branch_name' in result else None
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

@app.get("/predict_sales", response_model=SalesForecastResponse)
async def predict_sales(branch_id: int = Query(..., description="Branch ID for sales prediction")):
    """
    Predict sales for the next 7 days using hybrid ML models
    Returns actual and predicted sales with insights
    """
    if branch_id is None or branch_id <= 0:
        raise HTTPException(status_code=400, detail="Invalid branch_id")
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        # Get historical sales data
        cursor = conn.cursor(dictionary=True)
        query = """
            SELECT DATE(s.date) as date, SUM(s.amount) as revenue
            FROM sales s
            WHERE s.branch_id = %s
            GROUP BY DATE(s.date)
            ORDER BY date DESC
            LIMIT 60
        """
        cursor.execute(query, (branch_id,))
        sales_data = cursor.fetchall()
        
        if not sales_data:
            # Return empty response if no data
            return SalesForecastResponse(
                branch_id=branch_id,
                branch_name=get_branch_name(branch_id),
                dates=[],
                actual_sales=[],
                predicted_sales=[],
                growth_rate=0.0,
                confidence=0.0,
                insights=["No historical sales data available for forecasting."]
            )
        
        # Prepare data for forecasting
        df = pd.DataFrame(sales_data)
        df['date'] = pd.to_datetime(df['date'])
        df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce').fillna(0)
        df = df[df['revenue'] > 0].copy()  # Only days with actual sales
        df = df.sort_values('date')
        
        if len(df) < 3:
            return SalesForecastResponse(
                branch_id=branch_id,
                branch_name=get_branch_name(branch_id),
                dates=[],
                actual_sales=[],
                predicted_sales=[],
                growth_rate=0.0,
                confidence=0.0,
                insights=["Insufficient data for accurate forecasting. Need at least 3 days of sales data."]
            )
        
        # Feature engineering
        df['t'] = np.arange(len(df))
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['revenue_ma7'] = df['revenue'].rolling(window=7, min_periods=1).mean()
        df['revenue_ma30'] = df['revenue'].rolling(window=30, min_periods=1).mean()
        df['revenue_diff'] = df['revenue'].diff().fillna(0)
        df['revenue_pct_change'] = df['revenue'].pct_change().fillna(0)
        df['revenue_std'] = df['revenue'].rolling(window=7, min_periods=1).std().fillna(0)
        df['revenue_trend'] = df['revenue'].rolling(window=3, min_periods=1).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
        )
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_month_start'] = (df['date'].dt.day <= 7).astype(int)
        df['is_month_end'] = (df['date'].dt.day >= 25).astype(int)
        
        # Prepare features
        feature_cols = ['t', 'day_of_week', 'month', 'revenue_ma7', 'revenue_ma30', 
                        'revenue_diff', 'revenue_std', 'revenue_trend', 'is_weekend', 
                        'is_month_start', 'is_month_end']
        X = df[feature_cols].values
        y = df['revenue'].values
        
        # Train models
        models = {
            'linear': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        if TENSORFLOW_AVAILABLE and len(X) >= 5:
            models['dense_nn'] = create_dense_model(X.shape[1])
        if TENSORFLOW_AVAILABLE and len(X) >= 7:
            models['lstm'] = create_lstm_model(X.shape[1])
        
        best_model = None
        best_score = -float('inf')
        best_model_name = None
        
        for name, model in models.items():
            try:
                if name in ['dense_nn', 'lstm'] and TENSORFLOW_AVAILABLE:
                    if len(X) < 5:
                        continue
                    
                    if name == 'lstm':
                        X_seq, y_seq = prepare_sequence_data(X, y)
                        if X_seq is None:
                            continue
                        split_idx = int(len(X_seq) * 0.8)
                        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
                        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
                    else:
                        split_idx = int(len(X) * 0.8)
                        X_train, X_test = X[:split_idx], X[split_idx:]
                        y_train, y_test = y[:split_idx], y[split_idx:]
                    
                    trained_model, score = train_tensorflow_model(model, X_train, y_train, X_test, y_test)
                    if trained_model is not None:
                        model = trained_model
                else:
                    if len(X) < 3:
                        model.fit(X, y)
                        avg_revenue = np.mean(y) if len(y) > 0 else 0
                        score = min(0.8, avg_revenue / 1000)
                    else:
                        split_idx = int(len(X) * 0.8)
                        X_train, X_test = X[:split_idx], X[split_idx:]
                        y_train, y_test = y[:split_idx], y[split_idx:]
                        
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        score = r2_score(y_test, y_pred)
                
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_model_name = name
            except Exception as e:
                print(f"Error training {name}: {e}", file=sys.stderr)
                continue
        
        # Generate predictions
        if best_model is not None:
            predictions = []
            last_features = X[-1:].copy()
            is_tensorflow_model = hasattr(best_model, 'predict') and hasattr(best_model, 'layers')
            
            for i in range(7):
                if is_tensorflow_model and best_model_name == 'lstm':
                    last_sequence = X[-7:]
                    if len(last_sequence) < 7:
                        padding = np.zeros((7 - len(last_sequence), X.shape[1]))
                        last_sequence = np.vstack([padding, last_sequence])
                    sequence_input = last_sequence.reshape(1, 7, X.shape[1])
                    pred = float(best_model.predict(sequence_input, verbose=0)[0][0])
                    predictions.append(max(0.0, round(pred, 2)))
                    
                    new_features = last_sequence[-1].copy()
                    new_features[0] += 1
                    new_features[1] = (new_features[1] + 1) % 7
                    if new_features[1] == 0:
                        new_features[2] = (new_features[2] % 12) + 1
                    last_sequence = np.vstack([last_sequence[1:], new_features])
                else:
                    if is_tensorflow_model:
                        pred = float(best_model.predict(last_features, verbose=0)[0][0])
                    else:
                        pred = float(best_model.predict(last_features)[0])
                    predictions.append(max(0.0, round(pred, 2)))
                    
                    last_features[0][0] += 1
                    last_features[0][1] = (last_features[0][1] + 1) % 7
                    if last_features[0][1] == 0:
                        last_features[0][2] = (last_features[0][2] % 12) + 1
            
            # Calculate confidence and growth rate
            avg_confidence = max(10, min(100, round((best_score + 1) * 50, 1)))
            current_avg = np.mean(y[-7:]) if len(y) >= 7 else np.mean(y)
            future_avg = np.mean(predictions)
            growth_rate = ((future_avg - current_avg) / current_avg) if current_avg > 0 else 0.0
            
            # Generate insights
            insights = []
            if is_tensorflow_model:
                if best_model_name == 'lstm':
                    insights.append("Using advanced LSTM neural network for time series forecasting.")
                else:
                    insights.append("Using deep neural network for enhanced pattern recognition.")
            else:
                if best_model_name:
                    insights.append(f"Using {best_model_name} model for revenue forecasting.")
                else:
                    insights.append("Using machine learning model for revenue forecasting.")
            
            if growth_rate > 0.05:
                insights.append(f"Positive growth trend detected ({growth_rate*100:.1f}% increase expected).")
            elif growth_rate < -0.05:
                insights.append(f"Declining trend observed ({abs(growth_rate)*100:.1f}% decrease expected).")
            else:
                insights.append("Stable business performance expected.")
            
            # Prepare dates and actual sales
            actual_dates = df['date'].dt.strftime('%Y-%m-%d').tolist()
            actual_sales = df['revenue'].tolist()
            last_actual_date = df['date'].iloc[-1]
            
            # Forecast dates (next 7 days after last actual date)
            forecast_dates = [(last_actual_date + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(7)]
            all_dates = actual_dates + forecast_dates
            
            # Pad actual sales with None for forecast dates
            actual_sales_extended = actual_sales + [None] * 7
            
            # Pad predictions with None for actual dates
            predicted_sales_extended = [None] * len(actual_sales) + predictions
            
            return SalesForecastResponse(
                branch_id=branch_id,
                branch_name=get_branch_name(branch_id),
                dates=all_dates,
                actual_sales=actual_sales_extended,
                predicted_sales=predicted_sales_extended,
                growth_rate=round(growth_rate, 4),
                confidence=round(avg_confidence, 1),
                insights=insights
            )
        else:
            # Fallback
            return SalesForecastResponse(
                branch_id=branch_id,
                branch_name=get_branch_name(branch_id),
                dates=[],
                actual_sales=[],
                predicted_sales=[],
                growth_rate=0.0,
                confidence=0.0,
                insights=["Unable to generate forecast with available data."]
            )
    except Exception as e:
        print(f"Error in predict_sales: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Sales prediction failed: {str(e)}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

@app.get("/predict_inventory", response_model=InventoryForecastResponse)
async def predict_inventory(branch_id: int = Query(..., description="Branch ID for inventory prediction")):
    """
    Predict inventory depletion and restock timing
    Returns items with predicted stock levels and recommendations
    """
    if branch_id is None or branch_id <= 0:
        raise HTTPException(status_code=400, detail="Invalid branch_id")
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get inventory items
        query = """
            SELECT id, item_name, stock_level, min_stock
            FROM inventory
            WHERE branch_id = %s
        """
        cursor.execute(query, (branch_id,))
        items = cursor.fetchall()
        
        if not items:
            return InventoryForecastResponse(
                branch_id=branch_id,
                branch_name=get_branch_name(branch_id),
                items=[]
            )
        
        # Get service inventory mappings and sales data
        # Note: service_inventory table links services to inventory items
        # We need to join with inventory table to filter by branch_id
        mappings = []
        try:
            service_inventory_query = """
                SELECT si.service_id, si.inventory_id, si.quantity_required
                FROM service_inventory si
                JOIN inventory i ON si.inventory_id = i.id
                WHERE i.branch_id = %s
            """
            cursor.execute(service_inventory_query, (branch_id,))
            mappings = cursor.fetchall()
        except Exception as e:
            # If table doesn't exist or query fails, continue without mappings
            # This allows the API to still work, just without consumption rate data
            print(f"Warning: Could not fetch service_inventory mappings: {e}", file=sys.stderr)
            mappings = []
        
        # Get recent sales to estimate consumption
        sales_query = """
            SELECT s.service_id, COUNT(*) as service_count, SUM(s.amount) as total_revenue
            FROM sales s
            WHERE s.branch_id = %s AND s.date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
            GROUP BY s.service_id
        """
        cursor.execute(sales_query, (branch_id,))
        sales_data = cursor.fetchall()
        
        # Calculate predicted stock for each item
        inventory_items = []
        for item in items:
            current_stock = item['stock_level']
            min_stock = item['min_stock']
            
            # Calculate daily consumption rate from mappings and sales
            # quantity_required in mappings = items per service sale
            # Need to convert to daily consumption rate
            total_consumption_30days = 0
            total_sales_30days = 0
            
            for mapping in mappings:
                if mapping['inventory_id'] == item['id']:
                    for sale in sales_data:
                        if sale['service_id'] == mapping['service_id']:
                            # quantity_required is items per sale, service_count is number of sales
                            total_consumption_30days += mapping['quantity_required'] * sale['service_count']
                            total_sales_30days += sale['service_count']
            
            # Calculate daily consumption rate (items per day)
            daily_consumption = total_consumption_30days / 30.0 if total_consumption_30days > 0 else 0
            
            # If no consumption data, estimate based on current stock and minimum stock
            if daily_consumption == 0:
                # Assume minimum stock is meant to last 30 days
                daily_consumption = max(0.1, min_stock / 30.0) if min_stock > 0 else max(0.1, current_stock / 30.0)
            
            # Predict days until out of stock
            days_until_out = int(current_stock / daily_consumption) if daily_consumption > 0 else 999
            
            # Predict stock in 7 days
            predicted_stock = max(0, int(current_stock - (daily_consumption * 7)))
            
            # Overstock detection - STRICT criteria to avoid false positives
            # Only flag items that are clearly overstocked:
            # 1. Stock > 10x minimum (clearly excessive)
            # 2. OR (stock > 5x minimum AND days remaining > 180 - 6+ months supply)
            # This prevents items with moderate stock from being flagged as overstocked
            is_overstocked = (current_stock > min_stock * 10) or \
                           (current_stock > min_stock * 5 and days_until_out > 180)
            
            overstock_multiplier = None
            excess_stock = None
            overstock_recommendation = None
            
            if is_overstocked:
                overstock_multiplier = round(current_stock / min_stock, 1) if min_stock > 0 else 0
                # Excess beyond optimal stock (3x minimum is optimal)
                optimal_stock = min_stock * 3
                excess_stock = max(0, current_stock - optimal_stock)
                overstock_recommendation = f"Overstocked: {current_stock} units ({overstock_multiplier}x minimum). Excess: {excess_stock} units. Consider reducing future orders or running promotions."
            
            # Generate recommendation
            if is_overstocked:
                recommendation = "Overstocked - Reduce future orders"
            elif predicted_stock <= min_stock:
                if days_until_out <= 2:
                    recommendation = "Reorder immediately - Critical stock level"
                elif days_until_out <= 5:
                    recommendation = "Reorder within 2 days - Low stock warning"
                else:
                    recommendation = "Reorder within 5 days - Approaching minimum"
            else:
                recommendation = "Stock level adequate - Monitor consumption"
            
            inventory_items.append(InventoryItem(
                item_name=item['item_name'],
                current_stock=current_stock,
                predicted_stock=predicted_stock,
                days_until_out=days_until_out,
                recommendation=recommendation,
                is_overstocked=is_overstocked,
                overstock_multiplier=overstock_multiplier,
                excess_stock=excess_stock,
                overstock_recommendation=overstock_recommendation
            ))
        
        return InventoryForecastResponse(
            branch_id=branch_id,
            branch_name=get_branch_name(branch_id),
            items=inventory_items
        )
    except Exception as e:
        print(f"Error in predict_inventory: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Inventory prediction failed: {str(e)}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

@app.post("/ai_agent", response_model=AIAgentResponse)
async def ai_agent_endpoint(request: AIAgentQuery):
    """
    Interactive AI Agent endpoint
    Provides explanations, reasoning, and interpretations for user queries
    
    Example queries:
    - "Analyze my sales"
    - "What inventory needs restocking?"
    - "Give me recommendations"
    - "Explain your reasoning"
    - "Why should I restock this item?"
    """
    try:
        # Log the incoming query for debugging
        print(f"\n{'='*80}", file=sys.stderr)
        print(f"FASTAPI ENDPOINT: Received query: {request.query[:200]}", file=sys.stderr)
        print(f"FASTAPI ENDPOINT: Query contains 'inventory': {'inventory' in request.query.lower()}", file=sys.stderr)
        print(f"FASTAPI ENDPOINT: Query contains 'stock': {'stock' in request.query.lower()}", file=sys.stderr)
        print(f"{'='*80}\n", file=sys.stderr)
        
        agent = AIAgent()
        result = agent.analyze_and_explain(
            query=request.query,
            branch_id=request.branch_id,
            user_role=request.user_role or "admin"
        )

        # Call OpenRouter chat to naturalize/explain the result (auditable + fallback)
        try:
            llm = chat_llm(query=request.query, data_summary=result or {})
            # Overlay LLM reply while keeping basis/recommendations from classical pipeline
            result["response"] = llm.get("reply", result.get("response"))
            result["reasoning"] = llm.get("reasoning", result.get("reasoning"))
            result["interpretation"] = llm.get("reply", result.get("interpretation"))
        except Exception as llm_err:
            print(f"OpenRouter chat error (ignored, fallback to classical result): {llm_err}", file=sys.stderr)
        
        # Verify the result is correct type
        if result and isinstance(result, dict) and 'response' in result:
            response_lower = result['response'].lower() if isinstance(result['response'], str) else ''
            query_lower = request.query.lower()
            if ('inventory' in query_lower or 'stock' in query_lower) and ('sales data' in response_lower or 'revenue' in response_lower):
                print(f"ERROR: FastAPI endpoint detected wrong response type!", file=sys.stderr)
                print(f"Query was inventory but response is sales!", file=sys.stderr)
                # Force inventory response
                try:
                    result = agent._analyze_inventory_with_explanation(request.query, request.branch_id)
                except Exception as inv_error:
                    print(f"Error forcing inventory: {inv_error}", file=sys.stderr)
        
        return AIAgentResponse(**result)
    except Exception as e:
        print(f"Error in AI agent: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"AI agent error: {str(e)}")


@app.post("/chatbot", response_model=ChatBotResponse)
async def chatbot_endpoint(request: ChatBotRequest):
    """
    Lightweight conversational chatbot endpoint.
    Uses the same OpenRouter-backed chat_llm but with minimal context.
    Suitable for a simple chat widget in the UI.
    """
    try:
        context: Dict[str, Any] = {
            "branch_id": request.branch_id,
            "user_role": request.user_role,
            "mode": "chatbot",
        }
        llm = chat_llm(query=request.message, data_summary=context)
        return ChatBotResponse(
            reply=llm.get("reply", ""),
            reasoning=llm.get("reasoning", ""),
            prompt_snapshot=llm.get("prompt_used"),
        )
    except Exception as e:
        print(f"Error in chatbot endpoint: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Chatbot error: {str(e)}")


# ============================================================================
# NEW AI TASK ENDPOINTS
# ============================================================================

@app.post("/ai/competitor_pricing", response_model=CompetitorPricingResponse)
async def competitor_pricing_endpoint(request: CompetitorPricingRequest):
    """
    Competitor Pricing Comparison using Mistral AI.
    
    Compares business pricing against industry competitors and provides strategic recommendations.
    """
    try:
        from ai.mistral_ops import _call_mistral_api
        
        # Get our pricing data from database
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        query = """
            SELECT service_name, price, category 
            FROM services 
            WHERE 1=1
        """
        params = []
        
        if request.service_category:
            query += " AND category = %s"
            params.append(request.service_category)
        
        cursor.execute(query, params)
        our_services = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # Calculate average prices
        avg_price = sum(s['price'] for s in our_services) / len(our_services) if our_services else 0
        
        # Build Mistral prompt
        system_prompt = (
            "You are a competitive pricing analyst for a hair & nail studio in the Philippines. "
            "Analyze pricing strategies and provide actionable recommendations. "
            "IMPORTANT: Always use Philippine Peso (₱) for all currency amounts. Never use dollars ($)."
        )
        
        user_prompt = f"""Analyze our pricing strategy:

Our Services & Prices:
{chr(10).join(f"- {s['service_name']}: ₱{s['price']:.2f} ({s['category']})" for s in our_services[:10])}

Our Average Price: ₱{avg_price:.2f}

Tasks:
1. Compare our pricing against typical industry competitors
2. Identify pricing advantages and weaknesses
3. Provide 3-5 strategic pricing recommendations

Format your response as JSON with:
- competitor_analysis: List of competitor comparisons
- recommendations: List of strategic recommendations
- pricing_strategy: Overall strategy assessment

Remember: Use Philippine Peso (₱) for all amounts."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        result = _call_mistral_api(messages, max_tokens=800, temperature=0.3)
        
        if result and "choices" in result:
            content = result["choices"][0]["message"]["content"]
            # Post-process: Replace $ with ₱
            content = content.replace("$", "₱").replace("USD", "PHP")
            
            return CompetitorPricingResponse(
                our_pricing={"average_price": avg_price, "services": our_services},
                competitor_analysis=[{"analysis": content}],
                recommendations=[content],  # Extract recommendations from content
                ai_model_used="mistral-large-latest",
                fallback_used=False
            )
        else:
            # Fallback
            return CompetitorPricingResponse(
                our_pricing={"average_price": avg_price, "services": our_services},
                competitor_analysis=[],
                recommendations=["Review competitor pricing regularly", "Adjust prices based on market demand"],
                ai_model_used="fallback",
                fallback_used=True
            )
    except Exception as e:
        print(f"Error in competitor_pricing: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Competitor pricing error: {str(e)}")


@app.post("/ai/explain_chart", response_model=ChartExplanationResponse)
async def explain_chart_endpoint(request: ChartExplanationRequest):
    """
    Chart/Graph Explanation using Gemini AI.
    
    Explains trends and patterns in sales, inventory, or revenue charts.
    """
    try:
        from ai.gemini_insight import _call_gemini_api
        
        # Build Gemini prompt
        prompt = f"""You are a business analytics expert for a hair & nail studio in the Philippines.

Chart Type: {request.chart_type}
Time Period: {request.time_period or 'N/A'}

Chart Data:
{json.dumps(request.chart_data, indent=2)}

Tasks:
1. Explain the key trends visible in this chart
2. Identify significant patterns or anomalies
3. Provide 2-3 actionable insights
4. Suggest recommendations based on the data

IMPORTANT: 
- Use Philippine Peso (₱) for all currency amounts
- Be concise and business-focused
- Highlight both positive trends and concerns

Format your response with:
- Explanation: Overall trend description
- Key Insights: List of important findings
- Trends: List of identified patterns
- Recommendations: List of actionable suggestions"""
        
        result = _call_gemini_api(prompt, max_tokens=1000, temperature=0.3)
        
        if result and "candidates" in result:
            content = result["candidates"][0]["content"]["parts"][0]["text"]
            content = content.replace("$", "₱").replace("USD", "PHP")
            
            return ChartExplanationResponse(
                explanation=content,
                key_insights=[content[:200]],  # Extract insights
                trends=[content[:200]],
                recommendations=[content[:200]],
                ai_model_used="gemini-1.5-flash",
                fallback_used=False
            )
        else:
            # Fallback
            return ChartExplanationResponse(
                explanation="Chart shows data trends. Review patterns and adjust strategy accordingly.",
                key_insights=["Monitor trends regularly"],
                trends=["Data analysis needed"],
                recommendations=["Continue tracking metrics"],
                ai_model_used="fallback",
                fallback_used=True
            )
    except Exception as e:
        print(f"Error in explain_chart: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Chart explanation error: {str(e)}")


@app.post("/ai/kpi_narrative", response_model=KPINarrativeResponse)
async def kpi_narrative_endpoint(request: KPINarrativeRequest):
    """
    KPI Narrative Generation using Gemini AI.
    
    Converts numeric KPIs into compelling business stories.
    """
    try:
        from ai.gemini_insight import generate_insight
        
        # Use existing Gemini insight function
        result = generate_insight(request.kpi_data)
        
        narrative = result.get("summary", "")
        narrative = narrative.replace("$", "₱").replace("USD", "PHP")
        
        return KPINarrativeResponse(
            narrative=narrative,
            summary=narrative[:300],
            highlights=[narrative[:200]],
            concerns=[],
            ai_model_used=result.get("ai_model_used", "gemini-1.5-flash"),
            fallback_used=result.get("fallback_used", False)
        )
    except Exception as e:
        print(f"Error in kpi_narrative: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"KPI narrative error: {str(e)}")


@app.post("/ai/inventory_insight", response_model=InventoryInsightResponse)
async def inventory_insight_endpoint(request: InventoryInsightRequest):
    """
    Inventory Forecast Reasoning using Gemini AI.
    
    Explains inventory forecast predictions and provides reasoning.
    """
    try:
        from ai.gemini_insight import _call_gemini_api
        
        # Get inventory forecast data
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        query = """
            SELECT item_name, stock_level, min_stock, 
                   (stock_level - min_stock) as margin
            FROM inventory 
            WHERE branch_id = %s
        """
        if request.item_name:
            query += " AND item_name = %s"
            cursor.execute(query, (request.branch_id, request.item_name))
        else:
            cursor.execute(query, (request.branch_id,))
        
        items = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # Build prompt
        prompt = f"""You are an inventory management expert for a hair & nail studio in the Philippines.

Inventory Data:
{json.dumps(items[:20], indent=2)}

Tasks:
1. Analyze current inventory levels vs minimum stock requirements
2. Explain forecast reasoning for each item
3. Identify items needing immediate attention
4. Provide restocking recommendations

IMPORTANT: Use Philippine Peso (₱) for any cost-related values."""
        
        result = _call_gemini_api(prompt, max_tokens=800, temperature=0.3)
        
        if result and "candidates" in result:
            content = result["candidates"][0]["content"]["parts"][0]["text"]
            content = content.replace("$", "₱").replace("USD", "PHP")
            
            return InventoryInsightResponse(
                insights=[{"item": item["item_name"], "insight": content[:200]} for item in items[:5]],
                reasoning=content,
                recommendations=[content[:200]],
                ai_model_used="gemini-1.5-flash",
                fallback_used=False
            )
        else:
            return InventoryInsightResponse(
                insights=[],
                reasoning="Inventory analysis: Monitor stock levels and restock when below minimum.",
                recommendations=["Review inventory regularly", "Restock low items"],
                ai_model_used="fallback",
                fallback_used=True
            )
    except Exception as e:
        print(f"Error in inventory_insight: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Inventory insight error: {str(e)}")


@app.post("/ai/explain", response_model=InteractiveExplainResponse)
async def interactive_explain_endpoint(request: InteractiveExplainRequest):
    """
    Interactive Explanations using OpenRouter (primary) or Gemini (fallback).
    
    Answers user questions about any business data with natural language explanations.
    """
    try:
        from ai.openrouter_chat import chat_llm
        from ai.gemini_insight import _call_gemini_api
        
        # Try OpenRouter first
        try:
            context = request.context_data or {}
            context["branch_id"] = request.branch_id
            context["user_role"] = request.user_role
            
            result = chat_llm(query=request.question, data_summary=context)
            
            if not result.get("fallback_used", True):
                answer = result.get("reply", "")
                answer = answer.replace("$", "₱").replace("USD", "PHP")
                
                return InteractiveExplainResponse(
                    answer=answer,
                    reasoning=result.get("reasoning", ""),
                    related_insights=[answer[:200]],
                    ai_model_used="openrouter-llama",
                    fallback_used=False
                )
        except Exception as openrouter_err:
            print(f"OpenRouter failed, trying Gemini: {openrouter_err}", file=sys.stderr)
        
        # Fallback to Gemini
        prompt = f"""You are a business analyst assistant for a hair & nail studio in the Philippines.

User Question: {request.question}

Context Data:
{json.dumps(request.context_data or {}, indent=2)}

Provide a clear, helpful answer. Use Philippine Peso (₱) for all currency amounts."""
        
        result = _call_gemini_api(prompt, max_tokens=1000, temperature=0.3)
        
        if result and "candidates" in result:
            content = result["candidates"][0]["content"]["parts"][0]["text"]
            content = content.replace("$", "₱").replace("USD", "PHP")
            
            return InteractiveExplainResponse(
                answer=content,
                reasoning="Generated by Gemini AI",
                related_insights=[content[:200]],
                ai_model_used="gemini-1.5-flash",
                fallback_used=True
            )
        else:
            return InteractiveExplainResponse(
                answer="I can help explain business data. Please ask a specific question about sales, inventory, or operations.",
                reasoning="Fallback response",
                related_insights=[],
                ai_model_used="fallback",
                fallback_used=True
            )
    except Exception as e:
        print(f"Error in interactive_explain: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Interactive explain error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "Adarna Grande AI Analytics API"}

@app.get("/test_db")
async def test_database():
    """Test database connection endpoint"""
    from config import get_settings
    settings = get_settings()
    
    result = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "db_host": settings.db_host,
            "db_user": settings.db_user,
            "db_name": settings.db_name,
            "db_port": settings.db_port,
            "db_password_set": bool(settings.db_password)
        },
        "connection_test": {
            "status": "unknown",
            "error": None,
            "details": None
        }
    }
    
    try:
        connection = mysql.connector.connect(
            host=settings.db_host,
            user=settings.db_user,
            password=settings.db_password,
            database=settings.db_name,
            port=settings.db_port,
            connection_timeout=5
        )
        
        # Test query
        cursor = connection.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        cursor.close()
        connection.close()
        
        result["connection_test"]["status"] = "success"
        result["connection_test"]["details"] = "Database connection successful"
        
    except Error as e:
        result["connection_test"]["status"] = "failed"
        result["connection_test"]["error"] = str(e)
        result["connection_test"]["error_code"] = e.errno if hasattr(e, 'errno') else None
        
    except Exception as e:
        result["connection_test"]["status"] = "failed"
        result["connection_test"]["error"] = str(e)
        result["connection_test"]["error_type"] = type(e).__name__
    
    return result

@app.get("/test_gemini")
async def test_gemini():
    """
    Test endpoint for Gemini NLP API.
    
    This tests the PRIMARY use of Gemini: converting structured analytics
    outputs into human-readable insights (NLP task only).
    
    Forecasting is handled locally - NO Gemini used for numerical predictions.
    """
    from ai.gemini_insight import summarize_insights
    from config import get_settings
    
    settings = get_settings()
    
    # Sample KPI data for testing
    sample_kpis = {
        "total_revenue": 150000.50,
        "average_daily": 5000.02,
        "growth_rate": 0.12,
        "top_service": "Hair Color",
        "low_stock_items": 3,
        "trend": "increasing"
    }
    
    result = summarize_insights(sample_kpis)
    
    # Determine if Gemini AI is actually working
    gemini_working = not result.get("fallback_used", True)
    ai_model = result.get("ai_model_used", "fallback-template")
    
    return {
        "success": True,
        "gemini_api_key_set": bool(settings.gemini_api_key),
        "purpose": "NLP: Convert structured KPIs to human-readable insights",
        "gemini_ai_working": gemini_working,  # True if API succeeded, False if using fallback
        "ai_model_used": ai_model,  # Shows which model generated the summary
        "fallback_used": result.get("fallback_used", False),
        "fallback_reason": result.get("fallback_reason"),
        "summary": result.get("summary", ""),
        "test_kpis": sample_kpis,
        "status": "✅ Gemini AI is working!" if gemini_working else "⚠️ Using intelligent fallback (API unavailable)",
        "note": result.get("note", "Gemini API integration is implemented and ready.")
    }

@app.get("/test_gemini_status")
async def test_gemini_status():
    """
    Comprehensive diagnostic endpoint to check Gemini AI status.
    Shows if the API is working, key status, and model availability.
    """
    from ai.gemini_insight import summarize_insights
    from config import get_settings
    import requests
    
    settings = get_settings()
    
    # Test 1: Key validation (try a simple API call)
    key_valid = False
    if settings.gemini_api_key:
        try:
            # Try v1 first, then v1beta
            url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={settings.gemini_api_key}"
            test_response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{
                        "parts": [{"text": "test"}]
                    }],
                    "generationConfig": {"maxOutputTokens": 5}
                },
                timeout=5
            )
            if test_response.status_code == 200:
                key_valid = True
        except:
            pass
    
    # Test 2: Try actual API call
    sample_kpis = {
        "total_revenue": 150000.50,
        "average_daily": 5000.02,
        "growth_rate": 0.12,
        "top_service": "Hair Color",
        "low_stock_items": 3,
        "trend": "increasing"
    }
    
    result = summarize_insights(sample_kpis)
    gemini_working = not result.get("fallback_used", True)
    ai_model = result.get("ai_model_used", "fallback-template")
    
    return {
        "success": True,
        "diagnostic_summary": {
            "api_key_configured": bool(settings.gemini_api_key),
            "api_key_valid": key_valid,
            "gemini_ai_working": gemini_working,
            "ai_model_used": ai_model,
        },
        "status": "✅ Gemini AI is WORKING!" if gemini_working else "⚠️ Gemini AI is UNAVAILABLE (using fallback)",
        "recommendation": "System is working correctly" if not gemini_working else "Gemini API is responding successfully",
        "test_result": {
            "fallback_used": result.get("fallback_used", False),
            "fallback_reason": result.get("fallback_reason"),
            "summary": result.get("summary", ""),
        },
        "test_kpis": sample_kpis,
    }

@app.get("/test_mistral")
async def test_mistral():
    """
    Test endpoint for Mistral AI API.
    
    This tests Mistral's use for inventory and workforce operations recommendations.
    """
    from ai.mistral_ops import recommend_ops
    from config import get_settings
    
    settings = get_settings()
    
    # Sample inventory and staffing data for testing
    sample_inventory = {
        "low_stock_count": 5,
        "urgent_items": [
            {"name": "Hair Color Dye", "current_stock": 2, "min_stock": 10},
            {"name": "Nail Polish Remover", "current_stock": 1, "min_stock": 5}
        ],
        "total_items": 50,
        "overstocked_count": 2
    }
    
    sample_staffing = {
        "peak_hours": ["10:00-12:00", "14:00-16:00", "18:00-20:00"],
        "understaffed_branches": [
            {"name": "Main Branch", "current_staff": 3, "recommended_staff": 5},
            {"name": "Mall Branch", "current_staff": 2, "recommended_staff": 4}
        ],
        "total_staff": 15
    }
    
    result = recommend_ops(sample_inventory, sample_staffing)
    
    # Determine if Mistral AI is actually working
    mistral_working = not result.get("fallback_used", True)
    ai_model = result.get("ai_model_used", "fallback-template")
    
    return {
        "success": True,
        "mistral_api_key_set": bool(settings.mistral_api_key),
        "purpose": "Operations: Inventory & Workforce Recommendations",
        "mistral_ai_working": mistral_working,  # True if API succeeded, False if using fallback
        "ai_model_used": ai_model,  # Shows which model generated the recommendations
        "fallback_used": result.get("fallback_used", False),
        "recommendations": result.get("recommendations", []),
        "test_data": {
            "inventory": sample_inventory,
            "staffing": sample_staffing
        },
        "status": "✅ Mistral AI is working!" if mistral_working else "⚠️ Using intelligent fallback (API unavailable)",
        "note": "Mistral API integration is implemented and ready for inventory & workforce operations."
    }


@app.get("/test_mistral_status")
async def test_mistral_status():
    """
    Comprehensive diagnostic endpoint to check Mistral AI status.
    Shows if the API is working, key status, and model availability.
    """
    from ai.mistral_ops import recommend_ops
    from config import get_settings
    import requests
    
    settings = get_settings()
    
    # Test 1: Key validation (try a simple API call)
    key_valid = False
    if settings.mistral_api_key:
        try:
            url = "https://api.mistral.ai/v1/chat/completions"
            test_response = requests.post(
                url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {settings.mistral_api_key}"
                },
                json={
                    "model": "mistral-small-latest",
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 5
                },
                timeout=5
            )
            if test_response.status_code == 200:
                key_valid = True
        except:
            pass
    
    # Test 2: Try actual API call
    sample_inventory = {
        "low_stock_count": 3,
        "urgent_items": [{"name": "Test Item", "current_stock": 1, "min_stock": 5}],
    }
    
    sample_staffing = {
        "peak_hours": ["10:00-12:00"],
        "understaffed_branches": [{"name": "Test Branch", "current_staff": 2, "recommended_staff": 4}],
    }
    
    result = recommend_ops(sample_inventory, sample_staffing)
    mistral_working = not result.get("fallback_used", True)
    ai_model = result.get("ai_model_used", "fallback-template")
    
    return {
        "success": True,
        "diagnostic_summary": {
            "api_key_configured": bool(settings.mistral_api_key),
            "api_key_valid": key_valid,
            "mistral_ai_working": mistral_working,
            "ai_model_used": ai_model,
        },
        "status": "✅ Mistral AI is WORKING!" if mistral_working else "⚠️ Mistral AI is UNAVAILABLE (using fallback)",
        "recommendation": "System is working correctly" if mistral_working else "Mistral API is responding successfully",
        "test_result": {
            "fallback_used": result.get("fallback_used", False),
            "recommendations": result.get("recommendations", []),
        }
    }


@app.post("/mistral_ops_recommendations")
async def mistral_ops_recommendations(
    inventory_data: Dict[str, Any] = Body(...),
    staffing_data: Dict[str, Any] = Body(...)
):
    """
    Get inventory and workforce recommendations from Mistral AI.
    
    This endpoint uses Mistral AI to generate intelligent recommendations
    for inventory restocking and workforce optimization.
    """
    try:
        from ai.mistral_ops import recommend_ops
        
        result = recommend_ops(inventory_data, staffing_data)
        
        return {
            "success": True,
            "recommendations": result.get("recommendations", []),
            "mistral_ai_working": result.get("mistral_ai_working", False),
            "ai_model_used": result.get("ai_model_used", "fallback-template"),
            "fallback_used": result.get("fallback_used", False),
        }
    except Exception as e:
        print(f"Error in mistral_ops_recommendations: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Mistral ops error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

