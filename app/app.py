import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, classification_report
)
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────
st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --red:     #FF3B3B;
    --red-dim: rgba(255,59,59,0.13);
    --green:   #00D68F;
    --green-dim:rgba(0,214,143,0.13);
    --amber:   #FFB830;
    --blue:    #4D9FFF;
    --bg:      #0A0E1A;
    --bg2:     #111827;
    --bg3:     #1C2333;
    --border:  #FFFFFF12;
    --text:    #E8EDF5;
    --muted:   #8892A4;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

[data-testid="collapsedControl"] { display: flex !important; }
.block-container { padding: 1.5rem 2rem !important; max-width: 1400px; }

[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

[data-testid="metric-container"] {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricLabel"] { color: #8892A4 !important; font-size: 12px !important; text-transform: uppercase; letter-spacing: 1px; }
[data-testid="stMetricValue"] { color: #FFFFFF !important; font-family: 'Space Mono', monospace !important; font-size: 1.6rem !important; font-weight: 700 !important; -webkit-text-fill-color: #FFFFFF !important; }
[data-testid="stMetricValue"] > div { color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important; }
[data-testid="stMetric"] { background: #1C2333 !important; border: 1px solid rgba(255,255,255,0.07) !important; border-radius: 12px !important; padding: 1rem !important; }
[data-testid="metric-container"] { background: #1C2333 !important; border: 1px solid rgba(255,255,255,0.07) !important; border-radius: 12px !important; }
            
.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--muted);
    margin: 2rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}

.kpi-card {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
}
.kpi-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--muted);
    margin-bottom: 0.3rem;
}
.kpi-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
}
.kpi-sub { font-size: 12px; color: var(--muted); margin-top: 0.3rem; }

.hero {
    background: linear-gradient(135deg, #0A0E1A 0%, #111827 50%, #1a1040 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: var(--text);
    margin: 0;
}
.hero-title span { color: var(--red); }
.hero-sub { color: var(--muted); margin-top: 0.5rem; font-size: 14px; }

.status-live {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--green-dim);
    color: var(--green);
    border: 1px solid #00D68F33;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 1rem;
}
.dot { width: 7px; height: 7px; border-radius: 50%; background: var(--green); display:inline-block; }

.stTabs [data-baseweb="tab-list"] {
    background: var(--bg2) !important;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    padding: 6px 16px !important;
}
.stTabs [aria-selected="true"] {
    background: var(--bg3) !important;
    color: var(--text) !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem !important; }
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

PLOT_THEME = dict(
    paper_bgcolor='rgba(17,24,39,0)',
    plot_bgcolor='rgba(17,24,39,0)',
    font=dict(family='DM Sans', color='#8892A4', size=12),
    margin=dict(t=30, b=30, l=10, r=10),
    xaxis=dict(gridcolor='rgba(255,255,255,0.04)', zerolinecolor='rgba(255,255,255,0.08)'),
    yaxis=dict(gridcolor='rgba(255,255,255,0.04)', zerolinecolor='rgba(255,255,255,0.08)'),
)

# ── Load ──────────────────────────────────────────────
@st.cache_resource
def load_models():
    base = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(base)
    return {
        'lr'       : joblib.load(f'{root}/models/baseline_lr.pkl'),
        'xgb'      : joblib.load(f'{root}/models/final_xgb.pkl'),
        'lgbm'     : joblib.load(f'{root}/models/final_lgbm.pkl'),
        'threshold': joblib.load(f'{root}/models/ensemble_threshold.pkl'),
        'deploy'   : joblib.load(f'{root}/models/deploy_package.pkl'),
    }

@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(base)
    return {
        'X_test'       : joblib.load(f'{root}/data/X_test.pkl'),
        'y_test'       : joblib.load(f'{root}/data/y_test.pkl'),
        'X_train'      : joblib.load(f'{root}/data/X_train.pkl'),
        'prob_xgb'     : joblib.load(f'{root}/data/y_prob_final_xgb.pkl'),
        'prob_lgbm'    : joblib.load(f'{root}/data/y_prob_final_lgbm.pkl'),
        'prob_ensemble': joblib.load(f'{root}/data/y_prob_ensemble.pkl'),
        'pred_ensemble': joblib.load(f'{root}/data/y_pred_ensemble.pkl'),
    }

models = load_models()
data   = load_data()

X_test        = data['X_test']
y_test        = data['y_test']
X_train       = data['X_train']
prob_lr       = models['lr'].predict_proba(X_test)[:, 1]
prob_xgb      = data['prob_xgb']
prob_lgbm     = data['prob_lgbm']
prob_ensemble = data['prob_ensemble']
pred_ensemble = data['pred_ensemble']
threshold     = models['threshold']
deploy        = models['deploy']

cm             = confusion_matrix(y_test, pred_ensemble)
tn, fp, fn, tp = cm.ravel()
catch_rate     = tp / (tp + fn) * 100
false_alarm    = fp / (fp + tn) * 100

# ── Sidebar ───────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 0.5rem">
        <div style="font-family:'Space Mono',monospace;font-size:1.1rem;font-weight:700;color:#E8EDF5">
            🛡️ FraudShield
        </div>
        <div style="font-size:11px;color:#8892A4;letter-spacing:2px;text-transform:uppercase;margin-top:4px">
            AI Detection System
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="status-live"><span class="dot"></span> &nbsp;System Active</div>', unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio("", [
        "🏠  Overview",
        "📈  Model Performance",
        "🔬  Feature Lab",
        "⚡  Live Detector",
        "📋  Transaction Log",
        "💼  Business Impact",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div class="kpi-label">Model Stats</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    col1.metric("ROC-AUC", f"{deploy['roc_auc']}")
    col2.metric("PR-AUC",  f"{deploy['pr_auc']}")
    st.markdown(f"""
    <div style="font-size:12px;color:#8892A4;margin-top:0.8rem;line-height:2">
        🤖 <b style="color:#E8EDF5">{deploy['best_model_name']}</b> ensemble<br>
        🎯 Threshold: <b style="color:#E8EDF5">{deploy['threshold']}</b><br>
        📊 284,807 transactions<br>
        🏦 Kaggle CC Fraud dataset
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    confidence_filter = st.slider("Min fraud prob filter (%)", 0, 100, 50, 5)


# ══════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════
if "Overview" in page:
    st.markdown("""
    <div class="hero">
        <div class="hero-title">Fraud<span>Shield</span> AI Dashboard</div>
        <div class="hero-sub">Real-time credit card fraud detection · XGBoost + LightGBM ensemble · 284,807 transactions analyzed</div>
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Total Transactions", f"{len(y_test):,}")
    c2.metric("Actual Frauds",      f"{int(y_test.sum()):,}")
    c3.metric("Frauds Caught",      f"{tp:,}", delta=f"{catch_rate:.1f}%")
    c4.metric("Frauds Missed",      f"{fn:,}", delta=f"-{fn}", delta_color="inverse")
    c5.metric("False Alarms",       f"{fp:,}", delta_color="inverse")
    c6.metric("Money Saved",        f"${deploy['money_saved']:,.0f}")

    st.markdown("---")
    col_a, col_b, col_c = st.columns([1.2, 1.2, 1])

    with col_a:
        st.markdown('<div class="section-title">Class Distribution</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Pie(
            labels=['Legitimate', 'Fraud'],
            values=[(y_test==0).sum(), (y_test==1).sum()],
            hole=0.65,
            marker=dict(colors=['#4D9FFF','#FF3B3B'],
                        line=dict(color='#0A0E1A', width=3)),
            textinfo='label+percent',
            textfont=dict(color='#E8EDF5', size=12),
        ))
        fig.add_annotation(text="<b>0.17%</b><br>fraud rate",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=14, color='#E8EDF5'))
        fig.update_layout(**PLOT_THEME, height=260, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">Fraud Probability Distribution</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=prob_ensemble[y_test==0], name='Legitimate',
                                   marker_color='#4D9FFF', opacity=0.7,
                                   nbinsx=50, histnorm='probability'))
        fig.add_trace(go.Histogram(x=prob_ensemble[y_test==1], name='Fraud',
                                   marker_color='#FF3B3B', opacity=0.7,
                                   nbinsx=50, histnorm='probability'))
        fig.add_vline(x=threshold, line_dash='dash', line_color='#FFB830',
                      annotation_text=f'Threshold {threshold:.2f}',
                      annotation_font_color='#FFB830')
        fig.update_layout(**PLOT_THEME, height=260, barmode='overlay',
                          legend=dict(bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig, use_container_width=True)

    with col_c:
        st.markdown('<div class="section-title">Detection Scorecard</div>', unsafe_allow_html=True)
        for label, val, color in [
            ("Catch Rate",  f"{catch_rate:.1f}%",    "#00D68F"),
            ("Precision",   f"{tp/(tp+fp)*100:.1f}%","#4D9FFF"),
            ("False Alarms",f"{false_alarm:.2f}%",   "#FFB830"),
            ("ROC-AUC",     f"{deploy['roc_auc']}",  "#FF3B3B"),
            ("PR-AUC",      f"{deploy['pr_auc']}",   "#A855F7"),
        ]:
            st.markdown(f"""
            <div class="kpi-card" style="border-left:3px solid {color};padding:0.7rem 1rem;margin-bottom:0.5rem">
                <div class="kpi-label">{label}</div>
                <div style="font-family:'Space Mono',monospace;font-size:1.3rem;color:{color};font-weight:700">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Prediction Breakdown</div>', unsafe_allow_html=True)
    for col, (label, val, color, desc) in zip(
        st.columns(4),
        [("✅ True Negatives", f"{tn:,}", "#00D68F", "Legit correctly identified"),
         ("⚠️ False Positives",f"{fp:,}", "#FFB830", "Legit flagged as fraud"),
         ("❌ False Negatives",f"{fn:,}", "#FF3B3B", "Actual fraud missed"),
         ("🎯 True Positives", f"{tp:,}", "#4D9FFF", "Fraud correctly caught")]
    ):
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="border-top:3px solid {color};text-align:center">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value" style="color:{color}">{val}</div>
                <div class="kpi-sub">{desc}</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# PAGE 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════
elif "Model Performance" in page:
    st.markdown('<h2 style="font-family:Space Mono,monospace;color:#FFB830">📈 Model Performance</h2>', unsafe_allow_html=True)

    lr_roc     = roc_auc_score(y_test, prob_lr)
    lr_prauc   = average_precision_score(y_test, prob_lr)
    xgb_roc    = roc_auc_score(y_test, prob_xgb)
    xgb_prauc  = average_precision_score(y_test, prob_xgb)
    lgbm_roc   = roc_auc_score(y_test, prob_lgbm)
    lgbm_prauc = average_precision_score(y_test, prob_lgbm)
    ens_roc    = roc_auc_score(y_test, prob_ensemble)
    ens_prauc  = average_precision_score(y_test, prob_ensemble)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Comparison","📉 ROC Curves","📈 PR Curves","🔲 Confusion Matrix"])

    with tab1:
        col_a, col_b = st.columns([1.5, 1])
        with col_a:
            df_m = pd.DataFrame({
                'Model'  : ['Logistic Reg.','XGBoost','LightGBM','Ensemble'],
                'ROC-AUC': [lr_roc, xgb_roc, lgbm_roc, ens_roc],
                'PR-AUC' : [lr_prauc, xgb_prauc, lgbm_prauc, ens_prauc],
            })
            fig = go.Figure()
            fig.add_trace(go.Bar(name='ROC-AUC', x=df_m['Model'], y=df_m['ROC-AUC'],
                marker_color=['rgba(77,159,255,0.3)','rgba(77,159,255,0.3)','rgba(77,159,255,0.3)','#4D9FFF'],
                text=[f"{v:.4f}" for v in df_m['ROC-AUC']],
                textposition='outside', textfont=dict(color='#E8EDF5')))
            fig.add_trace(go.Bar(name='PR-AUC', x=df_m['Model'], y=df_m['PR-AUC'],
                marker_color=['rgba(255,59,59,0.3)','rgba(255,59,59,0.3)','rgba(255,59,59,0.3)','#FF3B3B'],
                text=[f"{v:.4f}" for v in df_m['PR-AUC']],
                textposition='outside', textfont=dict(color='#E8EDF5')))
            fig.update_layout(template='plotly_dark',height=380,
                barmode='group',yaxis=dict(range=[0.8, 1.01],gridcolor='rgba(255,255,255,0.04)'),
                legend=dict(bgcolor='rgba(0,0,0,0)'))
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.markdown('<div class="section-title">Score Table</div>', unsafe_allow_html=True)
            # Build all cards as one html string — avoids per-loop rendering bug
            cards_html = ""
            for _, row in df_m.iterrows():
                best   = row['Model'] == 'Ensemble'
                border = '#FF3B3B' if best else 'rgba(255,255,255,0.07)'
                best_badge = '<span style="font-size:10px;color:#FF3B3B;margin-left:6px">★ BEST</span>' if best else ''
                cards_html += f"""
                <div style="background:#1C2333;border:1px solid rgba(255,255,255,0.07);
                            border-left:3px solid {border};border-radius:12px;
                            padding:0.8rem 1rem;margin-bottom:0.5rem">
                    <div style="font-size:13px;font-weight:600;color:#E8EDF5">
                        {row['Model']}{best_badge}
                    </div>
                    <div style="display:flex;gap:1rem;margin-top:6px">
                        <div>
                            <span style="color:#8892A4;font-size:11px">ROC </span>
                            <span style="font-family:Space Mono,monospace;color:#4D9FFF">{row['ROC-AUC']:.4f}</span>
                        </div>
                        <div>
                            <span style="color:#8892A4;font-size:11px">PR </span>
                            <span style="font-family:Space Mono,monospace;color:#FF3B3B">{row['PR-AUC']:.4f}</span>
                        </div>
                    </div>
                </div>"""
            st.markdown(cards_html, unsafe_allow_html=True)

    with tab2:
        fpr_lr,  tpr_lr,  _ = roc_curve(y_test, prob_lr)
        fpr_xgb, tpr_xgb, _ = roc_curve(y_test, prob_xgb)
        fpr_lgbm,tpr_lgbm,_ = roc_curve(y_test, prob_lgbm)
        fpr_ens, tpr_ens, _ = roc_curve(y_test, prob_ensemble)
        fig = go.Figure()
        for fpr,tpr,name,color,dash in [
            (fpr_lr, tpr_lr, f'Logistic Reg. (AUC={lr_roc:.4f})',  '#8892A4','dash'),
            (fpr_xgb,tpr_xgb,f'XGBoost (AUC={xgb_roc:.4f})',      '#4D9FFF','solid'),
            (fpr_lgbm,tpr_lgbm,f'LightGBM (AUC={lgbm_roc:.4f})',  '#00D68F','solid'),
            (fpr_ens,tpr_ens,f'Ensemble (AUC={ens_roc:.4f})',      '#FF3B3B','solid'),
        ]:
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name,
                line=dict(color=color, dash=dash, width=3 if 'Ensemble' in name else 1.5)))
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],showlegend=False,
                                 line=dict(color='rgba(255,255,255,0.13)',dash='dot')))
        fig.update_layout(**PLOT_THEME, height=440,
                          xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
                          legend=dict(bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        p_lr,  r_lr,  _ = precision_recall_curve(y_test, prob_lr)
        p_xgb, r_xgb, _ = precision_recall_curve(y_test, prob_xgb)
        p_lgbm,r_lgbm,_ = precision_recall_curve(y_test, prob_lgbm)
        p_ens, r_ens, _ = precision_recall_curve(y_test, prob_ensemble)
        fig = go.Figure()
        for r,p,name,color,dash in [
            (r_lr, p_lr, f'Logistic Reg. (PR={lr_prauc:.4f})',  '#8892A4','dash'),
            (r_xgb,p_xgb,f'XGBoost (PR={xgb_prauc:.4f})',      '#4D9FFF','solid'),
            (r_lgbm,p_lgbm,f'LightGBM (PR={lgbm_prauc:.4f})',  '#00D68F','solid'),
            (r_ens,p_ens,f'Ensemble (PR={ens_prauc:.4f})',      '#FF3B3B','solid'),
        ]:
            fig.add_trace(go.Scatter(x=r, y=p, name=name,
                line=dict(color=color, dash=dash, width=3 if 'Ensemble' in name else 1.5)))
        fig.add_hline(y=y_test.mean(), line_dash='dot', line_color='rgba(255,255,255,0.2)',
                      annotation_text='Random baseline', annotation_font_color='#8892A4')
        fig.update_layout(**PLOT_THEME, height=440,
                          xaxis_title='Recall', yaxis_title='Precision',
                          legend=dict(bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.imshow(cm, text_auto=True,
                x=['Predicted Legit','Predicted Fraud'],
                y=['Actual Legit','Actual Fraud'],
                color_continuous_scale=[[0,'#111827'],[0.5,'#1a3a5c'],[1,'#4D9FFF']],
                aspect='auto')
            fig.update_traces(textfont=dict(size=18, color='#E8EDF5'))
            fig.update_layout(**PLOT_THEME, height=380, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            report = classification_report(y_test, pred_ensemble,
                         target_names=['Legitimate','Fraud'], output_dict=True)
            rows = [{'Class':cls,'Precision':f"{report[cls]['precision']:.4f}",
                     'Recall':f"{report[cls]['recall']:.4f}",
                     'F1':f"{report[cls]['f1-score']:.4f}",
                     'Support':f"{int(report[cls]['support']):,}"}
                    for cls in ['Legitimate','Fraud']]
            st.markdown('<div class="section-title">Classification Report</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=110)
            st.markdown('<div class="section-title" style="margin-top:1.5rem">Key Metrics</div>', unsafe_allow_html=True)
            st.metric("Fraud Catch Rate", f"{catch_rate:.1f}%")
            st.metric("False Alarm Rate", f"{false_alarm:.3f}%")
            st.metric("Ensemble ROC-AUC", f"{ens_roc:.4f}")
            st.metric("Ensemble PR-AUC",  f"{ens_prauc:.4f}")


# ══════════════════════════════════════════════════════
# PAGE 3 — FEATURE LAB
# ══════════════════════════════════════════════════════
elif "Feature Lab" in page:
    st.markdown('<h2 style="font-family:Space Mono,monospace;color:#FFB830">🔬 Feature Lab</h2>', unsafe_allow_html=True)

    feat_imp = pd.Series(
        models['xgb'].feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)

    tab1, tab2, tab3 = st.tabs(["📊 Importance","🔍 Distribution","🌡️ Correlation"])

    with tab1:
        col_a, col_b = st.columns([2, 1])
        with col_a:
            n    = st.slider("Features to display", 5, len(feat_imp), 20)
            top  = feat_imp.head(n)
            cols = ['#FF3B3B' if i < 3 else '#4D9FFF' if i < 8 else '#8892A4'
                    for i in range(len(top))]
            fig = go.Figure(go.Bar(
                x=top.values, y=top.index, orientation='h',
                marker=dict(color=cols),
                text=[f"{v:.4f}" for v in top.values],
                textposition='outside', textfont=dict(color='#E8EDF5', size=11)
            ))
            fig.update_layout(**PLOT_THEME, height=max(300, n*28),
                  xaxis_title='Importance score')
            fig.update_yaxes(autorange='reversed')
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            st.markdown('<div class="section-title">Top 8 Features</div>', unsafe_allow_html=True)
            feat_cards = ""
            ranks = ['🥇','🥈','🥉','4️⃣','5️⃣','6️⃣','7️⃣','8️⃣']
            clrs  = ['#FF3B3B']*3 + ['#4D9FFF']*5
            for i,(feat,score) in enumerate(feat_imp.head(8).items()):
                feat_cards += f"""
                <div style="background:#1C2333;border:1px solid rgba(255,255,255,0.07);
                            border-left:2px solid {clrs[i]};border-radius:12px;
                            padding:0.6rem 1rem;margin-bottom:0.4rem">
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <div style="font-size:12px;color:#E8EDF5">{ranks[i]} {feat}</div>
                        <div style="font-family:Space Mono,monospace;font-size:12px;color:{clrs[i]}">{score:.4f}</div>
                    </div>
                </div>"""
            st.markdown(feat_cards, unsafe_allow_html=True)

    with tab2:
        col_a, col_b = st.columns([1, 3])
        with col_a:
            feature   = st.selectbox("Feature", X_test.columns.tolist())
            plot_type = st.radio("Chart type", ["Histogram","Box","Violin"])
            normalize = st.checkbox("Normalize", value=True)
        with col_b:
            lv = X_test[y_test==0][feature].values
            fv = X_test[y_test==1][feature].values
            if plot_type == "Histogram":
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=lv, name='Legitimate', marker_color='#4D9FFF',
                    opacity=0.7, nbinsx=60, histnorm='probability' if normalize else None))
                fig.add_trace(go.Histogram(x=fv, name='Fraud', marker_color='#FF3B3B',
                    opacity=0.7, nbinsx=60, histnorm='probability' if normalize else None))
                fig.update_layout(**PLOT_THEME, barmode='overlay', height=360,
                                  legend=dict(bgcolor='rgba(0,0,0,0)'))
            elif plot_type == "Box":
                fig = go.Figure()
                fig.add_trace(go.Box(y=lv, name='Legitimate', marker_color='#4D9FFF', line_color='#4D9FFF'))
                fig.add_trace(go.Box(y=fv, name='Fraud',      marker_color='#FF3B3B', line_color='#FF3B3B'))
                fig.update_layout(**PLOT_THEME, height=360, legend=dict(bgcolor='rgba(0,0,0,0)'))
            else:
                fig = go.Figure()
                fig.add_trace(go.Violin(y=lv, name='Legitimate', fillcolor='rgba(77,159,255,0.2)',
                    line_color='#4D9FFF', box_visible=True, meanline_visible=True))
                fig.add_trace(go.Violin(y=fv, name='Fraud', fillcolor='rgba(255,59,59,0.2)',
                    line_color='#FF3B3B', box_visible=True, meanline_visible=True))
                fig.update_layout(**PLOT_THEME, height=360, legend=dict(bgcolor='rgba(0,0,0,0)'))
            st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        for col, vals, color, label in [
            (c1, lv, '#4D9FFF', 'Legitimate'),
            (c2, fv, '#FF3B3B', 'Fraud'),
        ]:
            with col:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">{label} — {feature}</div>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px">
                        <div><span style="color:#8892A4;font-size:11px">Mean</span><br>
                             <span style="font-family:Space Mono,monospace;color:{color}">{vals.mean():.4f}</span></div>
                        <div><span style="color:#8892A4;font-size:11px">Std</span><br>
                             <span style="font-family:Space Mono,monospace;color:{color}">{vals.std():.4f}</span></div>
                        <div><span style="color:#8892A4;font-size:11px">Min</span><br>
                             <span style="font-family:Space Mono,monospace;color:{color}">{vals.min():.4f}</span></div>
                        <div><span style="color:#8892A4;font-size:11px">Max</span><br>
                             <span style="font-family:Space Mono,monospace;color:{color}">{vals.max():.4f}</span></div>
                    </div>
                </div>""", unsafe_allow_html=True)

    with tab3:
        n_feats   = st.slider("Features in heatmap", 5, 20, 12)
        top_feats = feat_imp.head(n_feats).index.tolist()
        corr      = X_test[top_feats].corr()
        fig = px.imshow(corr, text_auto='.2f',
                        color_continuous_scale='RdBu_r',
                        zmin=-1, zmax=1, aspect='auto')
        fig.update_traces(textfont=dict(size=10, color='#E8EDF5'))
        fig.update_layout(**PLOT_THEME, height=480)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════
# PAGE 4 — LIVE DETECTOR
# ══════════════════════════════════════════════════════
elif "Live Detector" in page:
    st.markdown('<h2 style="font-family:Space Mono,monospace;color:#FFB830">⚡ Live Transaction Detector</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8892A4">Adjust parameters to simulate a transaction and get an instant fraud prediction.</p>', unsafe_allow_html=True)

    col_input, col_result = st.columns(2)

    with col_input:
        st.markdown('<div class="section-title">Transaction Parameters</div>', unsafe_allow_html=True)
        amount   = st.number_input("💰 Amount ($)", 0.01, 25000.0, 150.0, 0.01)
        hour     = st.slider("🕐 Hour of Day", 0, 23, 14)
        is_night = 1 if (hour >= 22 or hour <= 6) else 0

        st.markdown(f"""
        <div class="kpi-card" style="margin:0.8rem 0">
            <div style="display:flex;gap:1.5rem">
                <div><span class="kpi-label">Night Tx</span><br>
                     <span style="color:{'#FF3B3B' if is_night else '#00D68F'};font-weight:600">
                     {'⚠️ Yes' if is_night else '✅ No'}</span></div>
                <div><span class="kpi-label">High Value</span><br>
                     <span style="color:{'#FFB830' if amount > 77 else '#00D68F'};font-weight:600">
                     {'⚠️ Yes' if amount > 77 else '✅ No'}</span></div>
                <div><span class="kpi-label">Log Amount</span><br>
                     <span style="font-family:Space Mono,monospace;color:#4D9FFF">{np.log1p(amount):.3f}</span></div>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title">V-Feature Adjustments</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            v1  = st.slider("V1",  -5.0, 5.0, 0.0, 0.1)
            v3  = st.slider("V3",  -5.0, 5.0, 0.0, 0.1)
            v14 = st.slider("V14", -5.0, 5.0, 0.0, 0.1)
        with c2:
            v2  = st.slider("V2",  -5.0, 5.0, 0.0, 0.1)
            v4  = st.slider("V4",  -5.0, 5.0, 0.0, 0.1)
            v17 = st.slider("V17", -5.0, 5.0, 0.0, 0.1)

    with col_result:
        st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)

        feature_names = X_train.columns.tolist()
        input_dict    = {f: 0.0 for f in feature_names}
        input_dict.update({
            'Amount_scaled': (amount - 88.35) / 250.12,
            'Amount_log'   : np.log1p(amount),
            'Hour'         : float(hour),
            'Is_night'     : float(is_night),
            'Is_highval'   : 1.0 if amount > 77.17 else 0.0,
            'Night_highval': float(is_night) * (1.0 if amount > 77.17 else 0.0),
            'V1':v1,'V2':v2,'V3':v3,'V4':v4,'V14':v14,'V17':v17,
        })
        input_df = pd.DataFrame([input_dict])[feature_names]

        p_lr_live   = models['lr'].predict_proba(input_df)[0][1]
        p_xgb_live  = models['xgb'].predict_proba(input_df)[0][1]
        p_lgbm_live = models['lgbm'].predict_proba(input_df)[0][1]
        p_ens_live  = 0.4*p_xgb_live + 0.4*p_lgbm_live + 0.2*p_lr_live
        is_fraud    = p_ens_live >= threshold
        risk_pct    = p_ens_live * 100
        vc          = '#FF3B3B' if is_fraud else '#00D68F'

        st.markdown(f"""
        <div class="kpi-card" style="border:2px solid {vc};text-align:center;padding:1.5rem">
            <div style="font-family:Space Mono,monospace;font-size:1.2rem;font-weight:700;color:{vc}">
                {'🚨 FRAUD DETECTED' if is_fraud else '✅ LEGITIMATE'}
            </div>
            <div style="font-family:Space Mono,monospace;font-size:3rem;font-weight:700;color:{vc};margin:0.5rem 0">
                {risk_pct:.1f}%
            </div>
            <div style="color:#8892A4;font-size:12px">fraud probability</div>
        </div>""", unsafe_allow_html=True)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_pct,
            gauge={
                'axis':{'range':[0,100],'tickcolor':'#8892A4'},
                'bar':{'color':vc},
                'bgcolor':'#111827',
                'steps':[
                    {'range':[0,30],'color':'rgba(0,214,143,0.08)'},
                    {'range':[30,60],'color':'rgba(255,184,48,0.08)'},
                    {'range':[60,100],'color':'rgba(255,59,59,0.08)'},
                ],
                'threshold':{'line':{'color':'#FFB830','width':3},'value':threshold*100}
            },
            number={'suffix':'%','font':{'size':28,'color':'#E8EDF5'}}
        ))
        fig.update_layout(**PLOT_THEME, height=240)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-title">Per-Model Breakdown</div>', unsafe_allow_html=True)
        for name, prob in [('Logistic Reg.',p_lr_live),('XGBoost',p_xgb_live),
                           ('LightGBM',p_lgbm_live),('Ensemble',p_ens_live)]:
            pct   = prob * 100
            color = '#FF3B3B' if pct >= threshold*100 else '#00D68F'
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
                <div style="width:100px;font-size:12px;color:#8892A4">{name}</div>
                <div style="flex:1;background:#1C2333;border-radius:4px;height:8px;overflow:hidden">
                    <div style="width:{pct}%;height:100%;background:{color};border-radius:4px"></div>
                </div>
                <div style="width:48px;text-align:right;font-family:Space Mono,monospace;font-size:12px;color:{color}">{pct:.1f}%</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# PAGE 5 — TRANSACTION LOG
# ══════════════════════════════════════════════════════
elif "Transaction Log" in page:
    st.markdown('<h2 style="font-family:Space Mono,monospace;color:#FFB830">📋 Transaction Log</h2>', unsafe_allow_html=True)

    log_df = X_test.copy()
    log_df['Actual']     = y_test.values
    log_df['Predicted']  = pred_ensemble
    log_df['Fraud_Prob'] = (prob_ensemble * 100).round(2)
    log_df['Risk'] = pd.cut(log_df['Fraud_Prob'],
                            bins=[-1,30,60,80,101],
                            labels=['Low','Medium','High','Critical'])
    log_df['Status'] = log_df.apply(lambda r:
        'True Fraud'   if r['Actual']==1 and r['Predicted']==1 else
        'Missed Fraud' if r['Actual']==1 and r['Predicted']==0 else
        'False Alarm'  if r['Actual']==0 and r['Predicted']==1 else
        'Legitimate', axis=1)

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        status_filter = st.multiselect("Status",
            ['True Fraud','Missed Fraud','False Alarm','Legitimate'],
            default=['True Fraud','Missed Fraud','False Alarm'])
    with col_f2:
        risk_filter = st.multiselect("Risk level",
            ['Critical','High','Medium','Low'],
            default=['Critical','High'])
    with col_f3:
        min_prob = st.slider("Min fraud prob (%)", 0, 100, int(confidence_filter))

    filtered = log_df[
        (log_df['Status'].isin(status_filter)) &
        (log_df['Risk'].isin(risk_filter)) &
        (log_df['Fraud_Prob'] >= min_prob)
    ].sort_values('Fraud_Prob', ascending=False)

    st.markdown(f'<div class="section-title">{len(filtered):,} transactions matched</div>',
                unsafe_allow_html=True)

    display_cols = ['Fraud_Prob','Risk','Status','Amount_scaled','Hour','Is_night','Amount_log']
    st.dataframe(filtered[display_cols].head(200).reset_index(drop=True),
                 use_container_width=True, height=420)

    st.download_button("⬇️ Download filtered CSV",
        data=filtered[display_cols].to_csv(index=False),
        file_name='fraud_alerts.csv', mime='text/csv')

    st.markdown('<div class="section-title">Status Breakdown</div>', unsafe_allow_html=True)
    sc = log_df['Status'].value_counts()
    cmap = {'Legitimate':'#4D9FFF','True Fraud':'#00D68F',
            'False Alarm':'#FFB830','Missed Fraud':'#FF3B3B'}
    fig = go.Figure(go.Bar(
        x=sc.index, y=sc.values,
        marker_color=[cmap.get(s,'#8892A4') for s in sc.index],
        text=sc.values, textposition='outside',
        textfont=dict(color='#E8EDF5')
    ))
    fig.update_layout(**PLOT_THEME, height=300)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════
# PAGE 6 — BUSINESS IMPACT
# ══════════════════════════════════════════════════════
elif "Business Impact" in page:
    st.markdown('<h2 style="font-family:Space Mono,monospace;color:#FFB830">💼 Business Impact Analysis</h2>', unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.markdown('<div class="section-title">Assumptions</div>', unsafe_allow_html=True)
        avg_fraud_amt = st.number_input("Avg fraud amount ($)", 50, 5000, 122)
        investigation = st.number_input("Cost per investigation ($)", 1, 500, 25)
        fraud_scale   = st.number_input("Scale factor (× test set)", 1, 1000, 100,
                                        help="Estimate real-world annual impact")

    with col_b:
        money_saved  = tp * avg_fraud_amt * fraud_scale
        money_lost   = fn * avg_fraud_amt * fraud_scale
        invest_cost  = fp * investigation * fraud_scale
        net_benefit  = money_saved - money_lost - invest_cost

        st.markdown('<div class="section-title">Projected Annual Impact</div>', unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Fraud Prevented",    f"${money_saved:,.0f}", delta="saved")
        c2.metric("Fraud Undetected",   f"${money_lost:,.0f}",  delta=f"-${money_lost:,.0f}", delta_color="inverse")
        c3.metric("Investigation Cost", f"${invest_cost:,.0f}", delta_color="inverse")
        c4.metric("Net Benefit",        f"${net_benefit:,.0f}",
                  delta="profit" if net_benefit > 0 else "loss",
                  delta_color="normal" if net_benefit > 0 else "inverse")

    st.markdown("---")

    # Waterfall
    st.markdown('<div class="section-title">Financial Waterfall</div>', unsafe_allow_html=True)
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute","relative","relative","total"],
        x=["Fraud Prevented","Fraud Missed","Investigation Cost","Net Benefit"],
        y=[money_saved, -money_lost, -invest_cost, 0],
        connector={"line":{"color":"rgba(255,255,255,0.13)"}},
        decreasing={"marker":{"color":"#FF3B3B"}},
        increasing={"marker":{"color":"#00D68F"}},
        totals={"marker":{"color":"#4D9FFF"}},
        text=[f"${v:,.0f}" for v in [money_saved,-money_lost,-invest_cost,net_benefit]],
        textposition="outside", textfont=dict(color='#E8EDF5')
    ))
    fig.update_layout(**PLOT_THEME, height=380, yaxis_title="USD ($)")
    st.plotly_chart(fig, use_container_width=True)

    # Threshold sensitivity
    st.markdown('<div class="section-title">Threshold Sensitivity Analysis</div>', unsafe_allow_html=True)
    thresholds = np.linspace(0.01, 0.99, 100)
    savings, losses, inv_costs, nets = [], [], [], []
    for t in thresholds:
        pt = (prob_ensemble >= t).astype(int)
        cmt = confusion_matrix(y_test, pt)
        tnt,fpt,fnt,tpt = cmt.ravel()
        savings.append(tpt * avg_fraud_amt * fraud_scale)
        losses.append(fnt * avg_fraud_amt * fraud_scale)
        inv_costs.append(fpt * investigation * fraud_scale)
        nets.append(tpt*avg_fraud_amt*fraud_scale - fnt*avg_fraud_amt*fraud_scale - fpt*investigation*fraud_scale)

    fig = go.Figure()
    for y_vals, name, color in [
        (savings,   'Fraud Prevented',    '#00D68F'),
        (losses,    'Fraud Missed',       '#FF3B3B'),
        (inv_costs, 'Investigation Cost', '#FFB830'),
        (nets,      'Net Benefit',        '#4D9FFF'),
    ]:
        fig.add_trace(go.Scatter(x=thresholds, y=y_vals, name=name,
                                 line=dict(color=color, width=3 if name=='Net Benefit' else 1.5)))
    fig.add_vline(x=threshold, line_dash='dash', line_color='rgba(255,255,255,0.4)',
                  annotation_text=f'Current ({threshold:.2f})',
                  annotation_font_color='#E8EDF5')
    fig.update_layout(**PLOT_THEME, height=380,
                      xaxis_title='Decision Threshold', yaxis_title='USD ($)',
                      legend=dict(bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig, use_container_width=True)