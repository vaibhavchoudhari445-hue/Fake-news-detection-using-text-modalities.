import streamlit as st
import numpy as np
import pickle, re, os, joblib, nltk, requests
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

st.set_page_config(page_title="Fake News Detector", page_icon="🔬", layout="centered")

@st.cache_resource
def download_nltk():
    nltk.download("stopwords", quiet=True)
download_nltk()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Unbounded:wght@300;400;600;700;800&family=DM+Sans:wght@300;400;500;600&family=Inconsolata:wght@400;500;600&display=swap');

:root {
    --bg:       #f4faf5;
    --surface:  #ffffff;
    --surface2: #edf8ef;
    --border:   #c8eace;
    --border2:  #8dd498;
    --lime:     #16c940;
    --lime2:    #00e855;
    --emerald:  #0a8c30;
    --text:     #071a0e;
    --text2:    #2a5c38;
    --text3:    #6a9a74;
    --real:     #0a7a30;
    --real-bg:  #f0fdf4;
    --real-bd:  #86efac;
    --fake:     #dc2626;
    --fake-bg:  #fef2f2;
    --fake-bd:  #fca5a5;
    --warn:     #d97706;
    --warn-bg:  #fffbeb;
    --warn-bd:  #fcd34d;
    --radius:   12px;
    --shadow:   0 4px 20px rgba(22,201,64,0.12), 0 1px 4px rgba(0,0,0,0.04);
}
*,*::before,*::after{box-sizing:border-box;}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
#MainMenu,footer,header{visibility:hidden;}

.stApp{
    background:var(--bg);
    background-image:radial-gradient(ellipse at 0% 0%,rgba(22,201,64,0.1) 0%,transparent 45%),
                     radial-gradient(ellipse at 100% 100%,rgba(10,140,48,0.07) 0%,transparent 40%);
}
.block-container{padding-top:0!important;padding-bottom:4rem!important;max-width:880px!important;}

/* HERO */
.hero{padding:3.5rem 1rem 2.5rem;text-align:center;}
.hero-eyebrow{
    font-family:'Inconsolata',monospace;font-size:0.65rem;letter-spacing:4px;
    color:var(--lime);text-transform:uppercase;margin-bottom:1rem;
    display:inline-flex;align-items:center;gap:0.5rem;
    background:rgba(22,201,64,0.08);border:1px solid rgba(22,201,64,0.2);
    border-radius:6px;padding:0.3rem 0.8rem;
}
.hero-eyebrow::before{content:'●';font-size:0.5rem;animation:blink 1.5s infinite;}
@keyframes blink{0%,100%{opacity:1;}50%{opacity:0.2;}}
.hero-title{
    font-family:'Unbounded',sans-serif;font-weight:800;
    font-size:clamp(2.2rem,6.5vw,4.2rem);line-height:1;
    letter-spacing:-1.5px;color:var(--text);margin-bottom:0.6rem;
}
.hero-title .hl{color:var(--lime);}
.hero-sub{font-size:0.88rem;color:var(--text3);font-weight:300;max-width:460px;margin:0 auto 1.5rem;line-height:1.65;}
.hero-pills{display:flex;gap:0.45rem;justify-content:center;flex-wrap:wrap;}
.hero-pill{
    background:var(--surface);border:1.5px solid var(--border);
    border-radius:8px;padding:0.28rem 0.7rem;
    font-family:'Inconsolata',monospace;font-size:0.62rem;color:var(--text3);
    transition:all 0.2s;
}
.hero-pill:hover{border-color:var(--lime2);color:var(--emerald);}

.rule{height:1px;background:linear-gradient(90deg,transparent,var(--border2),transparent);margin:0 0 2rem;}

/* INPUT */
.input-shell{
    background:var(--surface);border:1.5px solid var(--border2);
    border-radius:var(--radius);padding:1.6rem 1.6rem 1.2rem;
    margin-bottom:1.3rem;box-shadow:var(--shadow);position:relative;
}
.input-shell::before{
    content:'';position:absolute;top:0;left:0;right:0;height:3px;
    background:linear-gradient(90deg,var(--lime),var(--emerald),var(--lime));
    border-radius:var(--radius) var(--radius) 0 0;
}
.field-label{
    font-family:'Inconsolata',monospace;font-size:0.62rem;
    letter-spacing:2.5px;color:var(--emerald);text-transform:uppercase;margin-bottom:0.65rem;
}

.stTextArea textarea{
    background:var(--surface2)!important;border:1.5px solid var(--border)!important;
    border-radius:8px!important;color:var(--text)!important;
    font-family:'DM Sans',sans-serif!important;font-size:0.9rem!important;line-height:1.65!important;
    transition:all 0.2s!important;
}
.stTextArea textarea:focus{border-color:var(--lime)!important;box-shadow:0 0 0 3px rgba(22,201,64,0.15)!important;}
.stTextArea textarea::placeholder{color:var(--text3)!important;font-style:italic;}
label[data-testid="stWidgetLabel"]{display:none!important;}
.stButton>button{
    background:linear-gradient(135deg,var(--lime),var(--emerald))!important;
    border:none!important;border-radius:10px!important;color:#fff!important;
    font-family:'Unbounded',sans-serif!important;font-weight:600!important;
    font-size:0.75rem!important;letter-spacing:1px!important;
    padding:0.78rem 2rem!important;width:100%!important;
    box-shadow:0 6px 24px rgba(22,201,64,0.3)!important;transition:all 0.2s!important;
}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 10px 32px rgba(22,201,64,0.42)!important;}

/* SECTION */
.sec-header{font-family:'Inconsolata',monospace;font-size:0.62rem;letter-spacing:2.5px;color:var(--text3);text-transform:uppercase;margin-bottom:0.8rem;display:flex;align-items:center;gap:0.5rem;}
.sec-line{flex:1;height:1px;background:var(--border);}

/* VERDICT */
.verdict-real{background:var(--real-bg);border:1.5px solid var(--real-bd);border-radius:var(--radius);padding:1.3rem;margin-bottom:0.9rem;box-shadow:0 4px 16px rgba(10,122,48,0.1);animation:slideUp 0.4s ease;}
.verdict-fake{background:var(--fake-bg);border:1.5px solid var(--fake-bd);border-radius:var(--radius);padding:1.3rem;margin-bottom:0.9rem;box-shadow:0 4px 16px rgba(220,38,38,0.1);animation:slideUp 0.4s ease;}
@keyframes slideUp{from{opacity:0;transform:translateY(10px);}to{opacity:1;transform:translateY(0);}}
.verdict-tag{font-family:'Inconsolata',monospace;font-size:0.57rem;letter-spacing:2.5px;text-transform:uppercase;margin-bottom:0.35rem;}
.verdict-real .verdict-tag{color:var(--real);}
.verdict-fake .verdict-tag{color:var(--fake);}
.verdict-label{font-family:'Unbounded',sans-serif;font-size:1.7rem;font-weight:700;letter-spacing:-1px;line-height:1;margin-bottom:0.3rem;}
.verdict-real .verdict-label{color:var(--real);}
.verdict-fake .verdict-label{color:var(--fake);}
.verdict-meta{font-family:'Inconsolata',monospace;font-size:0.6rem;color:var(--text3);letter-spacing:1px;}

/* MODELS */
.models-card{background:var(--surface);border:1.5px solid var(--border);border-radius:var(--radius);padding:1.1rem 1.2rem;margin-bottom:0.9rem;box-shadow:var(--shadow);animation:slideUp 0.5s ease;}
.models-title{font-family:'Inconsolata',monospace;font-size:0.58rem;letter-spacing:2.5px;color:var(--text3);text-transform:uppercase;margin-bottom:0.9rem;padding-bottom:0.6rem;border-bottom:1.5px solid var(--border);display:flex;align-items:center;gap:0.5rem;}
.models-title::before{content:'';display:block;width:3px;height:14px;background:linear-gradient(180deg,var(--lime),var(--emerald));border-radius:999px;}
.mrow{display:flex;align-items:center;gap:0.7rem;margin-bottom:0.75rem;}
.mrow:last-of-type{margin-bottom:0;}
.mname{font-family:'Inconsolata',monospace;font-size:0.62rem;color:var(--text2);width:130px;flex-shrink:0;}
.mbar-track{flex:1;background:var(--surface2);border-radius:999px;height:7px;overflow:hidden;border:1px solid var(--border);}
.mbar-fake{background:linear-gradient(90deg,#fca5a5,var(--fake));height:100%;border-radius:999px;}
.mbar-real{background:linear-gradient(90deg,#86efac,var(--real));height:100%;border-radius:999px;}
.mpct{font-family:'Inconsolata',monospace;font-size:0.67rem;font-weight:600;width:40px;text-align:right;flex-shrink:0;}
.mpct-real{color:var(--real);}.mpct-fake{color:var(--fake);}
.mbadge{font-family:'Inconsolata',monospace;font-size:0.54rem;letter-spacing:1px;padding:0.12rem 0.45rem;border-radius:5px;flex-shrink:0;}
.mbadge-real{background:var(--real-bg);color:var(--real);border:1px solid var(--real-bd);}
.mbadge-fake{background:var(--fake-bg);color:var(--fake);border:1px solid var(--fake-bd);}
.mbadge-na{background:var(--surface2);color:var(--text3);border:1px solid var(--border);}
.mdetail{font-family:'Inconsolata',monospace;font-size:0.54rem;color:var(--text3);margin:-0.45rem 0 0.6rem 138px;}

/* API */
.api-card{background:var(--surface);border:1.5px solid var(--border);border-radius:var(--radius);padding:1.1rem 1.2rem;margin-bottom:0.9rem;box-shadow:var(--shadow);animation:slideUp 0.6s ease;}
.score-block{display:flex;align-items:center;gap:1rem;background:var(--surface2);border:1.5px solid var(--border);border-radius:8px;padding:0.8rem;margin-bottom:0.8rem;}
.score-num{font-family:'Unbounded',sans-serif;font-size:2.4rem;font-weight:700;line-height:1;flex-shrink:0;letter-spacing:-2px;}
.score-high{color:var(--real);}.score-mid{color:var(--warn);}.score-low{color:var(--fake);}
.score-detail{flex:1;}
.score-lbl{font-family:'Inconsolata',monospace;font-size:0.55rem;letter-spacing:2px;color:var(--text3);margin-bottom:0.3rem;text-transform:uppercase;}
.sbar-track{background:var(--bg);border-radius:999px;height:8px;border:1px solid var(--border);overflow:hidden;margin-bottom:0.3rem;}
.sbar-high{background:linear-gradient(90deg,#86efac,var(--real));height:100%;border-radius:999px;}
.sbar-mid{background:linear-gradient(90deg,#fcd34d,var(--warn));height:100%;border-radius:999px;}
.sbar-low{background:linear-gradient(90deg,#fca5a5,var(--fake));height:100%;border-radius:999px;}
.score-sum{font-family:'DM Sans',sans-serif;font-size:0.78rem;color:var(--text2);line-height:1.45;}
.src-item{background:var(--surface2);border:1.5px solid var(--border);border-radius:8px;padding:0.7rem 0.85rem;margin-bottom:0.4rem;transition:all 0.15s;}
.src-item:hover{border-color:var(--border2);}
.src-ttl{font-family:'DM Sans',sans-serif;font-size:0.82rem;color:var(--text);line-height:1.4;margin-bottom:0.3rem;font-weight:500;}
.src-foot{display:flex;justify-content:space-between;align-items:center;}
.src-id{font-family:'Inconsolata',monospace;font-size:0.56rem;color:var(--text3);}
.sim-chip{font-family:'Inconsolata',monospace;font-size:0.56rem;padding:0.1rem 0.4rem;border-radius:4px;}
.sim-high{background:var(--real-bg);color:var(--real);border:1px solid var(--real-bd);}
.sim-mid{background:var(--warn-bg);color:var(--warn);border:1px solid var(--warn-bd);}
.sim-low{background:var(--bg);color:var(--text3);border:1px solid var(--border2);}
.src-lnk{color:var(--lime);font-family:'Inconsolata',monospace;font-size:0.57rem;text-decoration:none;margin-left:0.4rem;opacity:0.7;}
.src-lnk:hover{opacity:1;}
.no-results{text-align:center;padding:1.1rem;background:var(--surface2);border:1.5px dashed var(--border2);border-radius:8px;font-family:'Inconsolata',monospace;font-size:0.6rem;color:var(--text3);letter-spacing:1.5px;}

/* FINAL */
.final-real{background:var(--real-bg);border:1.5px solid var(--real-bd);}
.final-fake{background:var(--fake-bg);border:1.5px solid var(--fake-bd);}
.final-conflict{background:var(--warn-bg);border:1.5px solid var(--warn-bd);}
.final-real,.final-fake,.final-conflict{border-radius:10px;padding:0.85rem 1rem;text-align:center;margin-top:0.7rem;animation:slideUp 0.7s ease;}
.final-lbl{font-family:'Unbounded',sans-serif;font-size:1rem;font-weight:600;letter-spacing:-0.3px;margin-bottom:0.15rem;}
.final-real .final-lbl{font-size:1rem;font-weight:800;letter-spacing:0.5px;color:var(--real);}
.final-fake .final-lbl{color:var(--fake);}
.final-conflict .final-lbl{color:var(--warn);}
.final-sub{font-family:'Inconsolata',monospace;font-size:0.55rem;letter-spacing:1px;color:var(--text3);text-transform:uppercase;}

/* FOOTER */
.footer{text-align:center;padding:2rem 1rem 1rem;border-top:1.5px solid var(--border);margin-top:1.5rem;}
.footer-logo{font-family:'Unbounded',sans-serif;font-size:0.85rem;font-weight:700;color:var(--emerald);letter-spacing:-0.5px;margin-bottom:0.25rem;}
.footer-sub{font-family:'Inconsolata',monospace;font-size:0.54rem;letter-spacing:1.5px;color:var(--text3);}
[data-testid="column"]{padding:0 0.5rem!important;}
</style>
""", unsafe_allow_html=True)

MODEL_DIR = "models"
def safe_load(path):
    if not os.path.exists(path): return None
    for fn in [lambda p: joblib.load(p), lambda p: pickle.load(open(p,"rb")), lambda p: pickle.load(open(p,"rb"), fix_imports=True, encoding="latin1")]:
        try: return fn(path)
        except: pass
    return None

@st.cache_resource
def load_models():
    return {"vectorizer": safe_load(os.path.join(MODEL_DIR,"vectorizer.pkl")), "lr": safe_load(os.path.join(MODEL_DIR,"linear_regression_model.pkl")), "rf": safe_load(os.path.join(MODEL_DIR,"random_forest_model.pkl")), "lstm": safe_load(os.path.join(MODEL_DIR,"lstm_model.pkl"))}
models = load_models()

ps = PorterStemmer()
def clean(t): return " ".join(re.sub(r"[^a-zA-Z\s]"," ",t).lower().split())
def clean_sw(t):
    sw = set(stopwords.words("english"))
    return " ".join(w for w in clean(t).split() if w not in sw)
def clean_stem(t):
    sw = set(stopwords.words("english"))
    return " ".join(ps.stem(w) for w in clean(t).split() if w not in sw)
def best_clean(text, vectorizer):
    candidates = [clean(text), clean_sw(text), clean_stem(text)]; best, best_nnz = candidates[0], -1
    try:
        for c in candidates:
            nnz = vectorizer.transform([c]).nnz
            if nnz > best_nnz: best_nnz, best = nnz, c
    except: pass
    return best

def predict(model, vectorizer, text):
    if model is None or vectorizer is None: return None, None, None
    try:
        vec = vectorizer.transform([best_clean(text, vectorizer)]); pred = int(model.predict(vec)[0])
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(vec)[0]; classes = list(model.classes_)
            real_idx = classes.index(1) if 1 in classes else 1; fake_idx = 1 - real_idx
            real_pct = round(float(proba[real_idx])*100, 1); fake_pct = round(float(proba[fake_idx])*100, 1)
        else: real_pct = 100.0 if pred==1 else 0.0; fake_pct = 100.0 - real_pct
        return pred, real_pct, fake_pct
    except: return None, None, None

NEWS_API_KEY = "pub_3847b7c7931f4f589c2f958b054f38c8"
NAMED_KEEP = {"india","cricket","t20","world","cup","ipl","won","win","final","match","series","odi","test","championship","trophy","election","government","minister","president","parliament","covid","ukraine","russia","israel","gaza","china","usa","uk"}

def extract_keywords(text, n=8):
    sw = set(stopwords.words("english")) - NAMED_KEEP
    proper = [w for w in text.split() if w[0].isupper() and len(w)>2 and w.lower() not in {"the","a","an","in","on","at","is","was"}]
    lower_words = [w for w in re.sub(r"[^a-zA-Z\s]"," ",text).lower().split() if (w not in sw or w in NAMED_KEEP) and len(w)>2]
    seen, unique = set(), []
    for w in proper + lower_words:
        wl = w.lower()
        if wl not in seen: seen.add(wl); unique.append(w)
    return " ".join(unique[:n])

def make_queries(text):
    words = text.strip().split(); short_phrase = " ".join(words[:4]) if len(words)>=4 else text
    kw = extract_keywords(text, n=5); broad = extract_keywords(text, n=3); named = " ".join(w for w in words if len(w)>2 and w[0].isupper())
    seen, uq = set(), []
    for q in [short_phrase, kw, broad, named]:
        q = q.strip()
        if q and q not in seen: seen.add(q); uq.append(q)
    return uq

def verify_with_newsdata(text):
    queries = make_queries(text); all_results = []; last_error = None
    for query in queries:
        if not query.strip(): continue
        try:
            resp = requests.get("https://newsdata.io/api/1/news", params={"q":query,"language":"en","apikey":NEWS_API_KEY,"size":10}, timeout=12); data = resp.json()
            if data.get("status") != "success": last_error = data.get("message","API error"); continue
            all_results.extend(data.get("results",[]));
            if len(all_results)>=15: break
        except Exception as e: last_error=str(e); continue
    if not all_results:
        return {"found":False,"matches":[],"credibility_score":20,"summary":f"API error: {last_error}" if last_error else "No matching news found."}
    seen_t, unique_r = set(), []
    for r in all_results:
        t=(r.get("title") or "").strip().lower()
        if t and t not in seen_t: seen_t.add(t); unique_r.append(r)
    iw = set(re.sub(r"[^a-zA-Z\s]"," ",text).lower().split()); ip = set(); wl2=list(iw)
    for i in range(len(wl2)-1): ip.add(f"{wl2[i]} {wl2[i+1]}")
    matches = []
    for r in unique_r[:10]:
        title = (r.get("title") or "").strip()
        if not title: continue
        tw = set(re.sub(r"[^a-zA-Z\s]"," ",title).lower().split()); tp = set(); twl=list(tw)
        for i in range(len(twl)-1): tp.add(f"{twl[i]} {twl[i+1]}")
        wo=len(iw&tw); po=len(ip&tp)*2; total=max(len(iw|tw),1); similarity=round(min((wo+po)/total*100,100),1)
        matches.append({"title":title,"source":r.get("source_id","Unknown"),"url":r.get("link","#"),"date":(r.get("pubDate") or "")[:10],"similarity":similarity})
    matches.sort(key=lambda x:x["similarity"],reverse=True)
    top_sim=matches[0]["similarity"] if matches else 0; num_sources=len(set(m["source"] for m in matches))
    if top_sim>=40: credibility=min(95,65+int(top_sim//3)+num_sources*2); summary=f"Strong match across {num_sources} source(s) — story is verified."
    elif top_sim>=20: credibility=min(65,45+int(top_sim)); summary="Partial match — related stories exist but not fully confirmed."
    elif top_sim>=8: credibility=35; summary="Weak match — story may be real but not widely covered yet."
    else: credibility=20; summary="No strong match — story may be fabricated."
    return {"found":top_sim>=8,"matches":matches[:3],"credibility_score":int(credibility),"summary":summary}

def show_ml_results(text):
    vec = models["vectorizer"]
    lr_pred,lr_real,lr_fake = predict(models["lr"],vec,text)
    rf_pred,rf_real,rf_fake = predict(models["rf"],vec,text)
    lstm_pred,lstm_real,lstm_fake = predict(models["lstm"],vec,text)
    valid = [p for p in [lr_pred,rf_pred,lstm_pred] if p is not None]
    if not valid: st.error("⚠️ No ML models loaded."); return None
    real_votes = sum(1 for p in valid if p==1); is_real = real_votes > len(valid)/2
    if is_real:
        confs=[c for p,c in [(lr_pred,lr_real),(rf_pred,rf_real),(lstm_pred,lstm_real)] if p==1 and c is not None]; overall=round(np.mean(confs),1) if confs else 0
        st.markdown(f"""<div class="verdict-real"><div class="verdict-tag">✓ Verified</div><div class="verdict-label">Real News</div><div class="verdict-meta">{overall}% CONF · {real_votes}/{len(valid)} MODELS</div></div>""", unsafe_allow_html=True)
    else:
        confs=[c for p,c in [(lr_pred,lr_fake),(rf_pred,rf_fake),(lstm_pred,lstm_fake)] if p==0 and c is not None]; overall=round(np.mean(confs),1) if confs else 0
        st.markdown(f"""<div class="verdict-fake"><div class="verdict-tag">✗ Fabricated</div><div class="verdict-label">Fake News</div><div class="verdict-meta">{overall}% CONF · {len(valid)-real_votes}/{len(valid)} MODELS</div></div>""", unsafe_allow_html=True)
    st.markdown('<div class="models-card"><div class="models-title">Model Breakdown</div>', unsafe_allow_html=True)
    for name, pred_v, real_pct, fake_pct in [("Logistic Reg.", lr_pred,lr_real,lr_fake),("Random Forest",rf_pred,rf_real,rf_fake),("LSTM Network",lstm_pred,lstm_real,lstm_fake)]:
        if pred_v is None:
            st.markdown(f"""<div class="mrow"><span class="mname">{name}</span><div class="mbar-track"></div><span class="mpct mpct-fake">—</span><span class="mbadge mbadge-na">N/A</span></div>""", unsafe_allow_html=True)
        else:
            bar_cls="mbar-real" if pred_v==1 else "mbar-fake"; pct_cls="mpct-real" if pred_v==1 else "mpct-fake"
            badge="<span class='mbadge mbadge-real'>REAL</span>" if pred_v==1 else "<span class='mbadge mbadge-fake'>FAKE</span>"
            bar_w=real_pct if pred_v==1 else fake_pct; pct_val=real_pct if pred_v==1 else fake_pct
            st.markdown(f"""<div class="mrow"><span class="mname">{name}</span><div class="mbar-track"><div class="{bar_cls}" style="width:{bar_w}%"></div></div><span class="mpct {pct_cls}">{pct_val}%</span>{badge}</div><div class="mdetail">Real {real_pct}% / Fake {fake_pct}%</div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    return is_real

def show_api_result(result):
    score=result["credibility_score"]
    if score>=60: num_cls,bar_cls="score-high","sbar-high"
    elif score>=40: num_cls,bar_cls="score-mid","sbar-mid"
    else: num_cls,bar_cls="score-low","sbar-low"
    st.markdown(f"""<div class="api-card"><div class="models-title">Live Verification</div><div class="score-block"><div class="score-num {num_cls}">{score}</div><div class="score-detail"><div class="score-lbl">Credibility Score / 100</div><div class="sbar-track"><div class="{bar_cls}" style="width:{score}%"></div></div><div class="score-sum">{result['summary']}</div></div></div>""", unsafe_allow_html=True)
    if result["matches"]:
        for m in result["matches"]:
            sc="sim-high" if m["similarity"]>=30 else "sim-mid" if m["similarity"]>=15 else "sim-low"
            st.markdown(f"""<div class="src-item"><div class="src-ttl">{m['title']}</div><div class="src-foot"><span class="src-id">📰 {m['source']} · {m['date']}</span><span><span class="sim-chip {sc}">{m['similarity']}%</span><a href="{m['url']}" target="_blank" class="src-lnk">↗</a></span></div></div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div class="no-results">NO MATCHING ARTICLES FOUND</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# UI
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Fake News Detection System</div>
    <div class="hero-title">Detect <span class="hl">Fake</span><br>News Instantly</div>
    <div class="hero-sub">Multi-model AI combining Logistic Regression, Random Forest & LSTM with real-time news API verification.</div>
    <div class="hero-pills">
        <span class="hero-pill">🤖 Logistic Regression</span>
        <span class="hero-pill">🌲 Random Forest</span>
        <span class="hero-pill">🧠 LSTM Network</span>
        <span class="hero-pill">🌐 NewsData.io</span>
    </div>
</div>
<div class="rule"></div>
""", unsafe_allow_html=True)

st.markdown("""<div class="input-shell"><div class="field-label">News Article Input</div></div>""", unsafe_allow_html=True)
news_input = st.text_area("n", placeholder="Paste a news headline or full article here for analysis…", height=155, label_visibility="collapsed")
analyse = st.button("🔬  Analyse Article")

if analyse:
    if not news_input.strip(): st.warning("Please paste some news text first.")
    else:
        text = news_input.strip()
        st.markdown("<div class='rule' style='margin-top:0.5rem'></div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="sec-header">ML Analysis<div class="sec-line"></div></div>', unsafe_allow_html=True)
            with st.spinner(""): ml_is_real = show_ml_results(text)
        with col2:
            st.markdown('<div class="sec-header">Live Verification<div class="sec-line"></div></div>', unsafe_allow_html=True)
            with st.spinner(""): api_result = verify_with_newsdata(text)
            show_api_result(api_result)

            # Combined final verdict banner
            if ml_is_real is not None:
                score = api_result["credibility_score"]
                api_has_data = api_result["found"]
                api_says_real = score >= 50 and api_has_data
                api_says_fake = api_has_data and not api_says_real

                st.markdown("<hr style='border:none;border-top:1px solid var(--border);margin:0.8rem 0'>", unsafe_allow_html=True)

                if not api_has_data:
                    # API returned nothing — show ML verdict only, no conflict
                    if ml_is_real:
                        st.markdown("""<div class="final-real"><div class="final-lbl">✅ REAL NEWS</div><div class="final-sub">Based on ML Models · API: No data</div></div>""", unsafe_allow_html=True)
                    else:
                        st.markdown("""<div class="final-fake"><div class="final-lbl">🚨 FAKE NEWS</div><div class="final-sub">Based on ML Models · API: No data</div></div>""", unsafe_allow_html=True)
                elif ml_is_real and api_says_real:
                    # Both agree REAL
                    st.markdown(f"""<div class="final-real"><div class="final-lbl">✅ BOTH AGREE: REAL</div><div class="final-sub">ML Models + NewsData.io · {score}/100 credibility</div></div>""", unsafe_allow_html=True)
                elif not ml_is_real and api_says_fake:
                    # Both agree FAKE
                    st.markdown(f"""<div class="final-fake"><div class="final-lbl">🚨 BOTH AGREE: FAKE</div><div class="final-sub">ML Models + NewsData.io · {score}/100 credibility</div></div>""", unsafe_allow_html=True)
                else:
                    # API has data but ML and API disagree
                    st.markdown(f"""<div class="final-conflict"><div class="final-lbl">⚠️ CONFLICTING</div><div class="final-sub">ML vs API disagree · Review manually · {score}/100</div></div>""", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    <div class="footer-logo">Fake News Detector</div>
    <div class="footer-sub">LR · RF · LSTM · NewsData.io</div>
</div>""", unsafe_allow_html=True)
