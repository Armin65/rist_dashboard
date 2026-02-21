# Deployment-Anleitung: Risk Dashboard

## ⚠️ WICHTIG: Vercel ist NICHT ideal für Streamlit-Apps!

Streamlit benötigt einen **lang laufenden Prozess**, während Vercel für **Serverless Functions** designed ist. Die beste Lösung ist **Streamlit Cloud** (kostenlos und einfach).

---

## ✅ Option 1: Streamlit Cloud (EMPFOHLEN)

### Vorteile:
- ✅ Kostenlos
- ✅ Einfaches Deployment
- ✅ Automatische Updates bei Git-Push
- ✅ Perfekt für Streamlit-Apps

### Schritte:

1. **GitHub Repository erstellen** (falls noch nicht vorhanden):
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/DEIN-USERNAME/risk_dashboard.git
   git push -u origin main
   ```

2. **Streamlit Cloud Account erstellen**:
   - Gehe zu https://streamlit.io/cloud
   - Melde dich mit deinem GitHub-Account an

3. **App deployen**:
   - Klicke auf "New app"
   - Wähle dein Repository: `risk_dashboard`
   - Wähle Branch: `main`
   - Main file path: `app.py`
   - Klicke auf "Deploy!"

4. **Fertig!** Deine App ist unter `https://DEIN-APP-NAME.streamlit.app` verfügbar.

---

## 🚀 Option 2: Railway.app (Alternative zu Vercel)

### Vorteile:
- ✅ Einfaches Deployment
- ✅ Automatische Updates
- ✅ Kostenloser Plan verfügbar

### Schritte:

1. **Railway Account erstellen**: https://railway.app

2. **Neues Projekt erstellen**:
   - Klicke auf "New Project"
   - Wähle "Deploy from GitHub repo"
   - Wähle dein Repository

3. **Konfiguration**:
   - Railway erkennt automatisch Python
   - Start Command: `streamlit run app.py --server.port $PORT`
   - Railway setzt automatisch die Umgebungsvariable `PORT`

4. **Deploy!** Deine App ist unter `https://DEIN-PROJEKT.up.railway.app` verfügbar.

---

## 🌐 Option 3: Render.com (Alternative zu Vercel)

### Vorteile:
- ✅ Kostenloser Plan verfügbar
- ✅ Einfaches Deployment

### Schritte:

1. **Render Account erstellen**: https://render.com

2. **Neues Web Service erstellen**:
   - Klicke auf "New +" → "Web Service"
   - Verbinde dein GitHub Repository

3. **Konfiguration**:
   - Name: `risk-dashboard`
   - Environment: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
   - Plan: Free

4. **Deploy!** Deine App ist unter `https://risk-dashboard.onrender.com` verfügbar.

---

## ❌ Option 4: Vercel (NICHT EMPFOHLEN)

**Warum Vercel nicht ideal ist:**
- Vercel ist für Serverless Functions designed
- Streamlit braucht einen lang laufenden Prozess
- Die `vercel.json` und `api/streamlit.js` Dateien sind nur Platzhalter
- **Diese Lösung wird nicht funktionieren!**

**Wenn du trotzdem Vercel verwenden willst:**
Du müsstest einen separaten Streamlit-Server auf Railway/Render laufen lassen und dann über Vercel als Proxy darauf zugreifen. Das ist kompliziert und nicht sinnvoll.

---

## 📋 Voraussetzungen für alle Optionen

1. **GitHub Repository** mit deinem Code
2. **requirements.txt** (bereits vorhanden ✅)
3. **app.py** als Hauptdatei (bereits vorhanden ✅)

---

## 🔧 Lokales Testen vor Deployment

```bash
# Virtuelles Environment aktivieren
source .venv/bin/activate

# Dependencies installieren
pip install -r requirements.txt

# App lokal starten
streamlit run app.py
```

---

## 📝 Zusammenfassung

| Plattform | Kosten | Einfachheit | Empfehlung |
|-----------|--------|-------------|------------|
| **Streamlit Cloud** | Kostenlos | ⭐⭐⭐⭐⭐ | ✅ **BESTE WAHL** |
| Railway.app | Kostenlos (mit Limits) | ⭐⭐⭐⭐ | ✅ Gut |
| Render.com | Kostenlos (mit Limits) | ⭐⭐⭐⭐ | ✅ Gut |
| Vercel | Kostenlos | ⭐ | ❌ Nicht empfohlen |

**Fazit:** Nutze **Streamlit Cloud** für die einfachste und beste Erfahrung! 🎉
