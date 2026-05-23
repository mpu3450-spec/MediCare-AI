#MATERNAL HEALTH RISK PREDICTOR
# The project aims to assist in the early identification of high-risk
# pregnancies by using machine learning techniques to analyze patient health
# data and provide risk predictions.
import streamlit as st
import re
import pandas as pd
import os
import numpy as np
import joblib
from reportlab.platypus import SimpleDocTemplate ,Paragraph
from reportlab.lib.styles import getSampleStyleSheet


translations = {
    "en": {
        "maternal health risk predictor": "Maternal Health Risk Predictor",
        "Blood Pressure": "Blood Pressure",
        "Systolic BP (upper number)": "Systolic BP (upper number)",
        "Diastolic BP (lower number)": "Diastolic BP (lower number)",
        "Hemoglobin Level": "Hemoglobin Level",
        "Enter Hemoglobin (Hb) value": "Enter Hemoglobin (Hb) value",
        "Pregnancy Details": "Pregnancy Details",
        "Weeks of Pregnancy": "Weeks of Pregnancy",
        "Any previous complications?": "Any previous complications?",
        "Yes": "Yes",
        "No": "No",
        "Bleeding / Hemorrhage": "Bleeding / Hemorrhage",
        "Are you experiencing bleeding?": "Are you experiencing bleeding?",
        "Select symptoms if any:": "Select symptoms if any:",
        "Heavy bleeding": "Heavy bleeding",
        "Dizziness or weakness": "Dizziness or weakness",
        "Low BP feeling": "Low BP feeling",
        "Fast heartbeat": "Fast heartbeat",
        "Cold or pale skin": "Cold or pale skin",
        "Check My Risk": "Check My Risk",
        "Your Results": "Your Results",
        "BP Status": "BP Status",
        "Anemia Status": "Anemia Status",
        "Hemorrhage Risk": "Hemorrhage Risk",
        "Suggestions": "Suggestions",
        "Go to hospital NOW!": "Go to hospital NOW!",
        "Overall Risk": "Overall Risk",
    },
    "hi": {
        "maternal health risk predictor": "मातृ स्वास्थ्य जोखिम पूर्वानुमानक",
        "Blood Pressure": "रक्तचाप",
        "Systolic BP (upper number)": "सिस्टोलिक BP (ऊपरी संख्या)",
        "Diastolic BP (lower number)": "डायस्टोलिक BP (निचली संख्या)",
        "Hemoglobin Level": "हीमोग्लोबिन स्तर",
        "Enter Hemoglobin (Hb) value": "हीमोग्लोबिन दर्ज करें",
        "Pregnancy Details": "गर्भावस्था विवरण",
        "Weeks of Pregnancy": "गर्भावस्था के सप्ताह",
        "Any previous complications?": "क्या पहले कोई समस्या हुई थी?",
        "Yes": "हाँ",
        "No": "नहीं",
        "Bleeding / Hemorrhage": "रक्तस्राव",
        "Are you experiencing bleeding?": "क्या रक्तस्राव हो रहा है?",
        "Select symptoms if any:": "यदि कोई लक्षण हों तो चुनें:",
        "Heavy bleeding": "अधिक रक्तस्राव",
        "Dizziness or weakness": "चक्कर या कमजोरी",
        "Low BP feeling": "लो BP महसूस होना",
        "Fast heartbeat": "तेज धड़कन",
        "Cold or pale skin": "ठंडी या पीली त्वचा",
        "Check My Risk": "जोखिम जाँचें",
        "Your Results": "आपके परिणाम",
        "BP Status": "BP स्थिति",
        "Anemia Status": "एनीमिया स्थिति",
        "Hemorrhage Risk": "रक्तस्राव जोखिम",
        "Suggestions": "सुझाव",
        "Go to hospital NOW!": "अभी अस्पताल जाएं!",
        "Overall Risk": "कुल जोखिम",
    }
}


def smart_text(text, lang):
    return translations.get(lang, {}).get(text, text)

import joblib
import numpy as np

# Model load karo — ek baar
@st.cache_resource
def load_model():
    try:
        return joblib.load("maternal_model.pkl")
    except:
        return None

# ML prediction function
def ml_predict(systolic, diastolic, hb, 
               weeks, history, bleeding, lang):
    
    model = load_model()
    
    if model is None:
        return None  # model nahi mila
    
    features = [[
        systolic, diastolic, hb, weeks,
        1 if history == "Yes" else 0,
        1 if bleeding == "Yes" else 0
    ]]
    
    prediction  = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    confidence  = round(max(probability) * 100, 1)
    
    risk_map = {
        0: {"en": "✅ Low Risk",      "hi": "✅ कम खतरा"},
        1: {"en": "⚠️ Moderate Risk", "hi": "⚠️ मध्यम खतरा"},
        2: {"en": "🔴 High Risk",     "hi": "🔴 अधिक खतरा"},
    }
    
    return {
        "risk":       risk_map[prediction][lang],
        "confidence": confidence,
        "level":      prediction
    }

# ── LABEL FUNCTION ─────────────────────────
def get_label(score):
    if score >= 8:
        return "High"
    elif score >= 4:
        return "Moderate"
    else:
        return "Low"

# ── SAVE DATA ─────────────────────────
def save_data(systolic, diastolic, hb, weeks, total):
    label = get_label(total)

    data = {
        "Systolic": systolic,
        "Diastolic": diastolic,
        "HB": hb,
        "Weeks": weeks,
        "Risk Score": total,
        "Risk_Label": label
    }

    df = pd.DataFrame([data])

    if os.path.exists("patient_data.csv"):
        df.to_csv("patient_data.csv", mode='a', header=False, index=False)
    else:
        df.to_csv("patient_data.csv", index=False)

# ── PDF GENERATION ─────────────────────────
def create_pdf(systolic, diastolic, hb, total, advice):
    doc = SimpleDocTemplate("report.pdf")
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph("Maternal Health Report", styles["Title"]))
    content.append(Paragraph(f"Systolic BP: {systolic}", styles["Normal"]))
    content.append(Paragraph(f"Diastolic BP: {diastolic}", styles["Normal"]))
    content.append(Paragraph(f"Hemoglobin: {hb}", styles["Normal"]))
    content.append(Paragraph(f"Risk Score: {total}", styles["Normal"]))
    content.append(Paragraph(f"Advice: {advice}", styles["Normal"]))

    doc.build(content)

# ── MODULES ───────────────────────────────────────────────────
def bp_module(systolic, diastolic, lang):
    RESULTS = {
        "Low":      {"en": "✅ BP Normal",      "hi": "✅ BP सामान्य है"},
        "Moderate": {"en": "⚠️ BP At Risk",     "hi": "⚠️ BP थोड़ा अधिक है"},
        "High":     {"en": "🔴 BP High Risk",   "hi": "🔴 BP बहुत अधिक है"},
    }
    SUGG = {
        "Low":      {"en": ["Monitor BP monthly", "Stay hydrated"],
                     "hi": ["हर महीने BP जाँचें", "पानी ज़्यादा पिएं"]},
        "Moderate": {"en": ["Reduce salt", "Take rest"],
                     "hi": ["नमक कम खाएं", "आराम करें"]},
        "High":     {"en": ["Go to hospital immediately"],
                     "hi": ["तुरंत अस्पताल जाएं"]},
    }
    if systolic < 120 and diastolic < 80:
        risk, score = "Low", 0
    elif (120 <= systolic <= 139) or (80 <= diastolic <= 89):
        risk, score = "Moderate", 2
    else:
        risk, score = "High", 4
    return {"risk": risk, "score": score,
            "status": RESULTS[risk][lang],
            "suggestions": SUGG[risk][lang]}

def anemia_module(hb, lang):
    RESULTS = {
        "Low":      {"en": "✅ Hemoglobin Normal", "hi": "✅ हीमोग्लोबिन सामान्य है"},
        "Moderate": {"en": "⚠️ Mild Anemia",       "hi": "⚠️ हल्की खून की कमी"},
        "High":     {"en": "🔴 Severe Anemia",      "hi": "🔴 बहुत ज़्यादा खून की कमी"},
    }
    SUGG = {
        "Low":      {"en": ["Maintain iron-rich diet"],
                     "hi": ["आयरन से भरपूर खाना खाएं"]},
        "Moderate": {"en": ["Eat spinach, jaggery", "Take iron tablets"],
                     "hi": ["पालक, गुड़ खाएं", "आयरन गोली लें"]},
        "High":     {"en": ["Go to doctor immediately"],
                     "hi": ["तुरंत डॉक्टर के पास जाएं"]},
    }
    if hb >= 11:
        risk, score = "Low", 0
    elif 9 <= hb < 11:
        risk, score = "Moderate", 2
    else:
        risk, score = "High", 4
    return {"risk": risk, "score": score,
            "status": RESULTS[risk][lang],
            "suggestions": SUGG[risk][lang]}

def hemorrhage_module(bleeding, weeks, history, symptoms, lang):
    if bleeding == "No":
        return {
            "risk": "Low", "score": 0,
            "status":      {"en": "✅ No Risk",         "hi": "✅ कोई खतरा नहीं"}[lang],
            "suggestions": {"en": ["Monitor normally"], "hi": ["सामान्य देखभाल करें"]}[lang],
            "steps": [], "is_emergency": False
        }
    score = 0
    if "heavy_bleeding" in symptoms: score += 3
    if "dizziness"      in symptoms: score += 2
    if "low_bp"         in symptoms: score += 3
    if "fast_heartbeat" in symptoms: score += 3
    if "cold_pale_skin" in symptoms: score += 3
    if history == "Yes": score += 3
    if weeks > 28:       score += 1
    is_emergency = score >= 10

    if is_emergency:
        risk   = "EMERGENCY"
        status = {"en": "🚨 EMERGENCY",     "hi": "🚨 आपातकाल"}[lang]
        steps  = {"en": ["Call 108 NOW", "Go to hospital immediately", "Keep someone with you"],
                  "hi": ["अभी 108 call करें", "तुरंत अस्पताल जाएं", "किसी को साथ लेकर जाएं"]}[lang]
    elif score >= 6:
        risk   = "High"
        status = {"en": "🔴 High Risk",     "hi": "🔴 अधिक खतरा"}[lang]
        steps  = {"en": ["Contact doctor immediately", "Do not leave patient alone"],
                  "hi": ["तुरंत डॉक्टर को बुलाएं", "मरीज़ को अकेला न छोड़ें"]}[lang]
    elif score >= 3:
        risk   = "Moderate"
        status = {"en": "⚠️ Moderate Risk", "hi": "⚠️ मध्यम खतरा"}[lang]
        steps  = {"en": ["Inform ASHA worker"],
                  "hi": ["ASHA कार्यकर्ता को बताएं"]}[lang]
    else:
        risk   = "Low"
        status = {"en": "✅ Low Risk",      "hi": "✅ कम खतरा"}[lang]
        steps  = {"en": ["Monitor normally"],
                  "hi": ["सामान्य देखभाल करें"]}[lang]

    return {"risk": risk, "score": score, "status": status,
            "suggestions": steps, "steps": steps,
            "is_emergency": is_emergency}

def save_data(systolic, diastolic, hb, weeks, total):

    data = {
        "Systolic": systolic,
        "Diastolic": diastolic,
        "HB": hb,
        "Weeks": weeks,
        "Risk Score": total
    }

    df = pd.DataFrame([data])

    if os.path.exists("patient_data.csv"):
        df.to_csv("patient_data.csv", mode='a', header=False, index=False)
    else:
        df.to_csv("patient_data.csv", index=False)
# ══════════════════════════════════════════════════════════════
#                        UI STARTS HERE
# ══════════════════════════════════════════════════════════════
if "result_ready" not in st.session_state:
    st.session_state.result_ready = False
lang = st.selectbox("भाषा / Language", ["hi", "en"])
st.title(smart_text("maternal health risk predictor", lang))
st.divider()

# ── BP ────────────────────────────────────────────────────────
st.subheader(smart_text("Blood Pressure", lang))

systolic = st.number_input(
    smart_text("Systolic BP (upper number)", lang),
    min_value=60, max_value=250, value=120
)
diastolic = st.number_input(
    smart_text("Diastolic BP (lower number)", lang),
    min_value=40, max_value=150, value=80
)

st.divider()

# ── Anemia ────────────────────────────────────────────────────
st.subheader(smart_text("Hemoglobin Level", lang))

hb = st.number_input(
    smart_text("Enter Hemoglobin (Hb) value", lang),
    min_value=1.0, max_value=20.0, value=11.0
)

st.divider()

# ── Pregnancy ─────────────────────────────────────────────────
st.subheader(smart_text("Pregnancy Details", lang))

weeks = st.number_input(
    smart_text("Weeks of Pregnancy", lang),
    min_value=1, max_value=42, value=20
)

history = st.radio(
    smart_text("Any previous complications?", lang),
    [smart_text("Yes", lang), smart_text("No", lang)],
    key="history_radio"
)

st.divider()

# ── Hemorrhage ────────────────────────────────────────────────
st.subheader(smart_text("Bleeding / Hemorrhage", lang))

bleeding = st.radio(
    smart_text("Are you experiencing bleeding?", lang),
    [smart_text("Yes", lang), smart_text("No", lang)],
    key="bleeding_radio"
)

st.write(smart_text("Select symptoms if any:", lang))

sym1 = st.checkbox(smart_text("Heavy bleeding",        lang), key="cb1")
sym2 = st.checkbox(smart_text("Dizziness or weakness", lang), key="cb2")
sym3 = st.checkbox(smart_text("Low BP feeling",        lang), key="cb3")
sym4 = st.checkbox(smart_text("Fast heartbeat",        lang), key="cb4")
sym5 = st.checkbox(smart_text("Cold or pale skin",     lang), key="cb5")

symptoms = []
if sym1: symptoms.append("heavy_bleeding")
if sym2: symptoms.append("dizziness")
if sym3: symptoms.append("low_bp")
if sym4: symptoms.append("fast_heartbeat")
if sym5: symptoms.append("cold_pale_skin")

st.divider()

def get_ai_advice(bp_r, ana_r, hem_r, lang):

    # Emergency sabse pehle
    if hem_r["is_emergency"]:
        return ("🚨 तुरंत अस्पताल जाएं! देरी खतरनाक हो सकती है"
                if lang=="hi"
                else "🚨 Go to hospital immediately! Delay can be dangerous")

    advice = []

    # BP high
    if bp_r["risk"] == "High":
        advice.append("BP बहुत ज्यादा है — डॉक्टर से तुरंत मिलें"
                      if lang=="hi"
                      else "BP is very high — consult a doctor immediately")

    # Anemia high
    if ana_r["risk"] == "High":
        advice.append("गंभीर खून की कमी — आयरन उपचार जरूरी है"
                      if lang=="hi"
                      else "Severe anemia — iron treatment required urgently")

    # Combined danger
    if bp_r["risk"] == "High" and ana_r["risk"] == "High":
        advice.append("⚠️ BP + Anemia दोनों high हैं — high risk pregnancy"
                      if lang=="hi"
                      else "⚠️ Both BP and anemia are high — high risk pregnancy")

    # Moderate cases
    if bp_r["risk"] == "Moderate":
        advice.append("नमक कम करें और आराम करें"
                      if lang=="hi"
                      else "Reduce salt and take proper rest")

    if ana_r["risk"] == "Moderate":
        advice.append("आयरन युक्त भोजन लें (पालक, गुड़)"
                      if lang=="hi"
                      else "Take iron-rich diet (spinach, jaggery)")

    # Safe case
    if not advice:
        advice.append("सब सामान्य है — नियमित जांच कराते रहें"
                      if lang=="hi"
                      else "Everything looks normal — continue regular checkups")

    return "\n".join(advice)



# ── Submit ────────────────────────────────────────────────────
if st.button(smart_text("Check My Risk", lang)):

    bleeding_val = "Yes" if bleeding == smart_text("Yes", lang) else "No"
    history_val  = "Yes" if history  == smart_text("Yes", lang) else "No"

    bp_r  = bp_module(systolic, diastolic, lang)
    ana_r = anemia_module(hb, lang)
    hem_r = hemorrhage_module(bleeding_val, weeks, history_val, symptoms, lang)

    total = bp_r["score"] + ana_r["score"] + hem_r["score"]

    # ✅ SAVE DATA YAHAN
    save_data(systolic, diastolic, hb, weeks, total)

    # Overall Risk
    if total >= 8 or hem_r["is_emergency"]:
        overall_label = {"en": "🔴 HIGH RISK — See doctor now!",
                         "hi": "🔴 अधिक खतरा — तुरंत डॉक्टर से मिलें!"}[lang]
        overall_level = "High"
    elif total >= 4:
        overall_label = {"en": "⚠️ MODERATE RISK — Monitor closely",
                         "hi": "⚠️ मध्यम खतरा — ध्यान दें"}[lang]
        overall_level = "Moderate"
    else:
        overall_label = {"en": "✅ LOW RISK — All good",
                         "hi": "✅ सब ठीक है — नियमित जाँच करें"}[lang]
        overall_level = "Low"

    # Results
    st.subheader(smart_text("Your Results", lang))

    col1, col2, col3 = st.columns(3)
    with col1: st.metric(smart_text("BP Status",       lang), bp_r["risk"])
    with col2: st.metric(smart_text("Anemia Status",   lang), ana_r["risk"])
    with col3: st.metric(smart_text("Hemorrhage Risk", lang), hem_r["risk"])

    if overall_level == "High":
        st.error(overall_label)
    elif overall_level == "Moderate":
        st.warning(overall_label)
    else:
        st.success(overall_label)

    # Status messages
    st.info(bp_r["status"])
    st.info(ana_r["status"])
    st.info(hem_r["status"])

    # Suggestions
    st.subheader(smart_text("Suggestions", lang))
    all_tips = (bp_r["suggestions"]
              + ana_r["suggestions"]
              + hem_r["suggestions"])
    for tip in all_tips:
        st.write(f"• {tip}")

    # Emergency
    if hem_r["is_emergency"]:
        st.error("🚨 " + smart_text("Go to hospital NOW!", lang))
        for step in hem_r["steps"]:
            st.write(f"➡️ {step}")

    # ✅ AI Advice YAHAN SHIFT KIYA
    st.subheader("🤖 AI Advice" if lang=="en" else "🤖 AI सलाह")

    with st.spinner("AI soch raha hai..." if lang=="hi" else "AI thinking..."):
        ai_advice = get_ai_advice(bp_r, ana_r, hem_r, lang)
        st.info(ai_advice)
     # 👉 STORE
    st.session_state.bp_r = bp_r
    st.session_state.ana_r = ana_r
    st.session_state.hem_r = hem_r
    st.session_state.total = total
    st.session_state.ai_advice = ai_advice
    st.session_state.result_ready = True

 # PDF BUTTON
if st.session_state.result_ready:

    st.subheader("Results")
    st.write("Total:", st.session_state.total)

    st.info(st.session_state.ai_advice)

    # ✅ PDF button
    if st.button("📄 Generate Report"):

        create_pdf(
            systolic,
            diastolic,
            hb,
            st.session_state.total,
            st.session_state.ai_advice
        )

        with open("report.pdf", "rb") as f:
            st.download_button("Download PDF", f, "report.pdf")
    
    ml_result = ml_predict(
        systolic, diastolic, hb,
        weeks, history_val, bleeding_val, lang
    )
    
    if ml_result:
        st.divider()
        st.subheader(
            "🤖 ML Prediction" if lang=="en"
            else "🤖 ML अनुमान"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "ML Risk" if lang=="en" else "ML जोखिम",
                ml_result["risk"]
            )
        with col2:
            st.metric(
                "Confidence" if lang=="en" else "विश्वास",
                f"{ml_result['confidence']}%"
            )
        
        # Rule based vs ML compare karo
        st.info(
            f"Rule Based: {overall_level} | ML: {ml_result['risk']}"
        )
        if lang == "en":
            st.warning(
        "⚠️ This tool is for educational purposes only. "
        "Please consult a qualified doctor for medical advice."
                )

        else:
            st.warning(
        "⚠️ यह टूल केवल शैक्षिक उद्देश्य के लिए है। "
        "कृपया सही चिकित्सा सलाह के लिए डॉक्टर से संपर्क करें।"
            )