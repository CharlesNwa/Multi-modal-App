"""
Streamlit web app — Hand Gesture AI
Artistic UI inspired by pixel-trail interactive design.
Run:  streamlit run app.py
"""

import time
import streamlit as st
import mediapipe as mp
import cv2
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import streamlit.components.v1 as components

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hand Gesture AI",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 0 2rem 2rem; max-width: 100%; }

  .stApp {
    background: #0d0d0d;
    color: #f0ede6;
    font-family: 'Georgia', serif;
  }

  /* ── Hero ── */
  .hero-wrap {
    position: relative;
    width: 100%;
    min-height: 220px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    overflow: hidden;
    padding: 2.5rem 1rem 1.5rem;
  }
  .hero-title {
    font-size: clamp(2.5rem, 8vw, 6rem);
    font-weight: 900;
    letter-spacing: -0.03em;
    color: #f0ede6;
    margin: 0;
    z-index: 1;
    text-align: center;
    line-height: 1;
  }
  .hero-title span {
    color: #ffa04f;
  }
  .hero-desc {
    font-size: 1rem;
    color: #555;
    max-width: 540px;
    line-height: 1.7;
    margin: 0.8rem 0 0;
    z-index: 1;
    text-align: center;
  }
  .hero-sub {
    font-size: 0.78rem;
    color: #444;
    margin-top: 0.4rem;
    z-index: 1;
    text-align: center;
    letter-spacing: 0.1em;
    text-transform: uppercase;
  }

  /* ── Divider ── */
  .divider { border: none; border-top: 1px solid #222; margin: 1rem 0; }

  /* ── Gesture chips ── */
  .chips {
    display: flex; gap: 0.6rem; flex-wrap: wrap;
    justify-content: center; margin: 1rem 0 1.5rem;
  }
  .chip {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 99px;
    padding: 0.45rem 1.1rem;
    font-size: 0.9rem;
    color: #aaa;
    letter-spacing: 0.04em;
    transition: all 0.2s;
  }
  .chip:hover { border-color: #ffa04f; color: #ffa04f; }

  /* ── Result box ── */
  .result-wrap {
    background: #111;
    border: 1px solid #222;
    border-radius: 20px;
    padding: 2rem 1.5rem;
    text-align: center;
    min-height: 160px;
    display: flex; flex-direction: column;
    justify-content: center; align-items: center;
  }
  .result-gesture {
    font-size: clamp(1.8rem, 4vw, 3.2rem);
    font-weight: 900;
    color: #ffa04f;
    letter-spacing: -0.02em;
  }
  .result-idle { font-size: 1rem; color: #333; font-style: italic; }

  .conf-label { font-size: 0.78rem; color: #444; margin-top: 0.6rem; letter-spacing: 0.1em; text-transform: uppercase; }
  .conf-track {
    background: #1e1e1e; border-radius: 99px;
    height: 6px; width: 200px; margin: 0.4rem auto 0; overflow: hidden;
  }
  .conf-fill {
    height: 100%; border-radius: 99px;
    background: linear-gradient(90deg, #ffa04f, #ff6b35);
  }

  /* ── Section label ── */
  .sec-label {
    font-size: 0.7rem; color: #444;
    letter-spacing: 0.18em; text-transform: uppercase;
    margin-bottom: 0.5rem; text-align: center;
  }

  /* ── Footer ── */
  .footer {
    text-align: right; color: #2a2a2a;
    font-size: 0.75rem; padding: 1.5rem 0 0;
    letter-spacing: 0.06em;
  }
</style>
""", unsafe_allow_html=True)

# ── Pixel trail canvas (full-page interactive background) ─────────────────────
components.html("""
<canvas id="c" style="
  position:fixed; top:0; left:0;
  width:100vw; height:100vh;
  pointer-events:none; z-index:0; opacity:0.55;
"></canvas>
<script>
const canvas = document.getElementById('c');
const ctx    = canvas.getContext('2d');
canvas.width  = window.innerWidth;
canvas.height = window.innerHeight;

const PIXEL = 52, FADE = 0.04, DELAY = 1400;
const pixels = [];
let mx = -999, my = -999;

window.addEventListener('resize', () => {
  canvas.width  = window.innerWidth;
  canvas.height = window.innerHeight;
});

window.addEventListener('mousemove', e => { mx = e.clientX; my = e.clientY; });

function addPixel() {
  if (mx < 0) return;
  pixels.push({ x: Math.round(mx/PIXEL)*PIXEL, y: Math.round(my/PIXEL)*PIXEL, a: 1 });
}

setInterval(addPixel, 60);

function loop() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (let i = pixels.length - 1; i >= 0; i--) {
    const p = pixels[i];
    p.a -= FADE;
    if (p.a <= 0) { pixels.splice(i, 1); continue; }
    ctx.globalAlpha = p.a;
    ctx.fillStyle   = '#ffa04f';
    ctx.beginPath();
    ctx.arc(p.x + PIXEL/2, p.y + PIXEL/2, PIXEL/2, 0, Math.PI*2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
  requestAnimationFrame(loop);
}
loop();
</script>
""", height=0)


# ── Gesture classifier ────────────────────────────────────────────────────────
def classify_gesture(lm):
    idx = lm[8].y  < lm[6].y  - 0.02
    mid = lm[12].y < lm[10].y - 0.02
    rng = lm[16].y < lm[14].y - 0.02
    pky = lm[20].y < lm[18].y - 0.02

    t_up   = lm[4].y < lm[2].y - 0.04
    t_down = lm[4].y > lm[2].y + 0.04
    curled = not idx and not mid and not rng and not pky
    all4   = idx and mid and rng and pky
    ok_d   = ((lm[4].x-lm[8].x)**2 + (lm[4].y-lm[8].y)**2)**0.5

    if curled and t_up:                          return "👍 Thumbs Up",   0.93
    if curled and t_down:                        return "👎 Thumbs Down", 0.93
    if curled:                                   return "✊ Fist",         0.88
    if idx and mid and not rng and not pky:      return "✌️ Peace",        0.92
    if all4:                                     return "🖐️ Stop",         0.88
    if ok_d < 0.08 and mid and rng and pky:      return "👌 OK",           0.88
    return "", 0.0


# ── WebRTC video processor ────────────────────────────────────────────────────
class GestureProcessor(VideoProcessorBase):
    def __init__(self):
        self._h = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.6, min_tracking_confidence=0.5,
        )
        self._d  = mp.solutions.drawing_utils
        self._s  = mp.solutions.drawing_styles
        self._mp = mp.solutions.hands
        self.label = ""
        self.conf  = 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img    = frame.to_ndarray(format="bgr24")
        result = self._h.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        label, conf = "", 0.0
        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0]
            self._d.draw_landmarks(img, lm, self._mp.HAND_CONNECTIONS,
                self._s.get_default_hand_landmarks_style(),
                self._s.get_default_hand_connections_style())
            label, conf = classify_gesture(lm.landmark)

        self.label, self.conf = label, conf

        h, w = img.shape[:2]
        # dark strip
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 72), (13, 13, 13), -1)
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)

        txt   = label if label else "show a gesture..."
        color = (80, 160, 255) if label else (60, 60, 60)
        fs    = 1.05
        (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        if tw > w - 20:
            fs = fs * (w - 20) / tw
        cv2.putText(img, txt, (12, 52), cv2.FONT_HERSHEY_SIMPLEX, fs, color, 2, cv2.LINE_AA)

        if label:
            bw = int(conf * 160)
            cv2.rectangle(img, (10, 60), (170, 68), (30, 30, 30), -1)
            cv2.rectangle(img, (10, 60), (10 + bw, 68), (80, 160, 255), -1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ── Layout ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <h1 class="hero-title">hand gesture<span> ✽</span> ai</h1>
  <p class="hero-desc">
    A computer vision system that recognizes hand gestures in real time using MediaPipe
    for hand tracking and a Bidirectional LSTM for gesture classification —
    no keyboard or touchscreen required.
  </p>
  <p class="hero-sub">real-time · offline · built by Charles Nwachukwu 🇳🇬</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="chips">
  <span class="chip">👍 Thumbs Up</span>
  <span class="chip">👎 Thumbs Down</span>
  <span class="chip">✊ Fist</span>
  <span class="chip">✌️ Peace</span>
  <span class="chip">🖐️ Stop</span>
  <span class="chip">👌 OK</span>
</div>
<hr class="divider"/>
""", unsafe_allow_html=True)

cam_col, res_col = st.columns([3, 2], gap="large")

with cam_col:
    st.markdown('<div class="sec-label">📷 live camera</div>', unsafe_allow_html=True)
    ctx = webrtc_streamer(
        key="gesture-ai",
        video_processor_factory=GestureProcessor,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with res_col:
    st.markdown('<div class="sec-label">🎯 detected gesture</div>', unsafe_allow_html=True)
    result_ph = st.empty()

    if ctx.video_processor:
        while True:
            lbl  = ctx.video_processor.label
            conf = ctx.video_processor.conf
            pct  = int(conf * 100)
            if lbl:
                result_ph.markdown(f"""
                <div class="result-wrap">
                  <div class="result-gesture">{lbl}</div>
                  <div class="conf-label">confidence</div>
                  <div class="conf-track">
                    <div class="conf-fill" style="width:{pct}%"></div>
                  </div>
                  <div class="conf-label">{pct}%</div>
                </div>""", unsafe_allow_html=True)
            else:
                result_ph.markdown("""
                <div class="result-wrap">
                  <div class="result-idle">waiting for gesture…</div>
                </div>""", unsafe_allow_html=True)
            time.sleep(0.05)
    else:
        result_ph.markdown("""
        <div class="result-wrap">
          <div class="result-idle">start the camera to begin</div>
        </div>""", unsafe_allow_html=True)

st.markdown("""
<hr class="divider"/>


<style>
  /* ── Animations ── */
  @keyframes fadeUp {
    from { opacity:0; transform:translateY(24px); }
    to   { opacity:1; transform:translateY(0); }
  }
  @keyframes scaleIn {
    from { opacity:0; transform:scale(0.6); }
    to   { opacity:0.35; transform:scale(1); }
  }
  .animate-fade-up  { opacity:0; animation: fadeUp  0.7s ease forwards; }
  .animate-scale-in { opacity:0; animation: scaleIn 1.2s ease forwards; }

  /* ── CTA wrapper ── */
  .cta-section {
    position: relative;
    overflow: hidden;
    padding: 4rem 1rem 5rem;
    text-align: center;
  }
  .cta-inner {
    position: relative;
    max-width: 680px;
    margin: 0 auto;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1.4rem;
    z-index: 1;
  }

  /* Badge */
  .cta-badge {
    display: inline-block;
    border: 1px solid #2a2a2a;
    border-radius: 99px;
    padding: 0.35rem 1rem;
    font-size: 0.78rem;
    color: #666;
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }

  /* Title */
  .cta-title {
    font-size: clamp(2rem, 5vw, 3.6rem);
    font-weight: 700;
    line-height: 1.1;
    letter-spacing: -0.03em;
    color: #f0ede6;
    margin: 0;
  }

  /* Description */
  .cta-desc {
    font-size: 1rem;
    color: #555;
    line-height: 1.7;
    max-width: 520px;
    margin: 0;
  }

  /* Button */
  .cta-btn {
    display: inline-block;
    background: #ffa04f;
    color: #0d0d0d;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.04em;
    padding: 0.75rem 2rem;
    border-radius: 99px;
    text-decoration: none;
    transition: background 0.2s, transform 0.2s, box-shadow 0.2s;
    box-shadow: 0 0 0 0 rgba(255,160,79,0);
  }
  .cta-btn:hover {
    background: #ffb36b;
    transform: translateY(-2px);
    box-shadow: 0 0 32px 8px rgba(255,160,79,0.25);
  }

  /* Glow blob */
  .cta-glow {
    position: absolute;
    width: 500px; height: 500px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(255,160,79,0.18) 0%, transparent 70%);
    top: 50%; left: 50%;
    transform: translate(-50%, -50%) scale(1);
    pointer-events: none;
    z-index: 0;
  }
</style>

<div class="footer">make the web fun again ✽</div>
""", unsafe_allow_html=True)
