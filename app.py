from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np

app = Flask(__name__)

# Change this path if your CSV is elsewhere
CSV_PATH = "player_stats.csv"

# Load dataset once at startup for responsiveness
df = pd.read_csv(CSV_PATH)

# Normalize column names (lowercase) for safety
df.columns = [c.strip() for c in df.columns]

# Helper functions
def player_summary(player_name):
    # Try several column names that likely exist in dataset
    # We'll search for a 'player' or 'player_id' or 'player_name' column
    players_cols = [c for c in df.columns if c.lower() in ("player", "player_name", "player_id", "playerid")]
    
    if players_cols:
        pname_col = players_cols[0]
    else:
        # fallback: try 'player' literal
        pname_col = "player"
    
    # CASE-INSENSITIVE search by 'player' column
    player_df = df[df[pname_col].astype(str).str.lower() == str(player_name).lower()]
    
    if player_df.empty:
        # attempt partial match
        player_df = df[df[pname_col].astype(str).str.lower().str.contains(str(player_name).lower(), na=False)]
    
    if player_df.empty:
        return None
    
    kills = player_df.get('kill', player_df.get('kills', player_df.get('kills ', pd.Series([0]*len(player_df))))).astype(float).sum()
    deaths = player_df.get('death', player_df.get('deaths', pd.Series([0]*len(player_df)))).astype(float).sum()
    assists = player_df.get('assist', player_df.get('assists', pd.Series([0]*len(player_df)))).astype(float).sum()
    matches = player_df.shape[0]
    
    # ACS column (average combat score) might be 'acs' or 'rating' etc.
    acs_col = None
    for c in df.columns:
        if c.lower() in ("acs", "avgcombat", "averagecombat", "rating"):
            acs_col = c
            break
    
    acs = None
    if acs_col:
        acs = player_df[acs_col].astype(float).mean()
    
    kast_col = next((c for c in df.columns if c.lower().startswith("kast")), None)
    kast = None
    if kast_col:
        # try to parse percent if present like "75%"
        kast_vals = player_df[kast_col].astype(str).str.replace('%','').astype(float)
        kast = kast_vals.mean()
    
    hs_col = next((c for c in df.columns if 'hs' in c.lower()), None)
    hs = None
    if hs_col:
        hs_vals = player_df[hs_col].astype(str).str.replace('%','').astype(float, errors='ignore')
        hs = pd.to_numeric(hs_vals, errors='coerce').mean()
    
    # K/D ratio
    kd = round(kills / deaths, 2) if deaths > 0 else float('inf')
    
    # top agents and maps
    agent_col = next((c for c in df.columns if 'agent' in c.lower()), None)
    top_agent = None
    if agent_col:
        top_agent = player_df[agent_col].value_counts().idxmax()
    
    map_col = next((c for c in df.columns if c.lower() in ('map','maps')), None)
    top_map = None
    if map_col:
        top_map = player_df[map_col].value_counts().idxmax()
    
    # simple consistency metric: std of ACS or kills
    consistency = None
    if acs_col:
        consistency = round(player_df[acs_col].astype(float).std(), 2)
    
    # Build summary
    summary = {
        "player": player_name,
        "matches": int(matches),
        "kills": int(kills),
        "deaths": int(deaths),
        "assists": int(assists),
        "kd_ratio": kd,
        "acs": round(float(acs),2) if acs is not None and not np.isnan(acs) else None,
        "kast_pct": round(float(kast),2) if kast is not None and not np.isnan(kast) else None,
        "hs_pct": round(float(hs),2) if hs is not None and not np.isnan(hs) else None,
        "top_agent": top_agent,
        "top_map": top_map,
        "consistency": consistency
    }
    
    # small time series sample: ACS per match (if acs_col exists) or kills per match
    timeseries = []
    if acs_col:
        timeseries = list(player_df[acs_col].astype(float).fillna(0).head(50))
    else:
        timeseries = list(player_df.get('kill', player_df.get('kills', pd.Series([0]))).astype(float).fillna(0).head(50))
    
    summary['timeseries'] = timeseries
    
    # agent distribution (top 5)
    if agent_col:
        agent_counts = player_df[agent_col].value_counts().nlargest(6).to_dict()
        summary['agent_counts'] = agent_counts
    else:
        summary['agent_counts'] = {}
    
    return summary

@app.route("/")
def home():
    # Build player list for dropdown. Try to find a 'player' column in df
    players_cols = [c for c in df.columns if c.lower() in ("player", "player_name", "playerid", "player_id")]
    
    if players_cols:
        pname_col = players_cols[0]
    else:
        pname_col = df.columns[0]
    
    players = sorted(df[pname_col].astype(str).unique().tolist())
    return render_template("home.html", players=players)

@app.route("/showdown")
def showdown():
    players_cols = [c for c in df.columns if c.lower() in ("player","player_name","playerid","player_id")]
    pname_col = players_cols[0] if players_cols else df.columns[0]
    players = sorted(df[pname_col].astype(str).unique().tolist())
    return render_template("showdown.html", players=players)

@app.route("/champions")
def champions():
    return render_template("champions.html")

# CORRECTED API ENDPOINTS FOR CASE-INSENSITIVE SEARCH BY 'player' COLUMN

@app.route("/api/search")
def api_search():
    # Case-insensitive search by 'player' column
    q = request.args.get("q", "").strip().lower()
    
    if not q:
        return jsonify({"players": []})
    
    # Find the 'player' column
    players_cols = [c for c in df.columns if c.lower() in ("player", "player_name", "playerid", "player_id")]
    
    if players_cols:
        pname_col = players_cols[0]
    else:
        pname_col = "player"
    
    # Search for players containing the query (case-insensitive)
    matching_players = df[df[pname_col].astype(str).str.lower().str.contains(q, na=False)][pname_col].unique().tolist()
    
    # Limit to top 10 matches
    matching_players = sorted(matching_players)[:10]
    
    return jsonify({"players": matching_players})

@app.route("/api/player")
def api_player():
    # Case-insensitive player lookup by 'player' column
    name = request.args.get("name", "").strip()
    
    if not name:
        return jsonify({"error": "no player specified"}), 400
    
    summary = player_summary(name)
    
    if summary is None:
        return jsonify({"error": "player not found"}), 404
    
    return jsonify(summary)

@app.route("/api/compare")
def api_compare():
    # Case-insensitive comparison of two players
    player1 = request.args.get("player1", "").strip()
    player2 = request.args.get("player2", "").strip()
    
    if not player1 or not player2:
        return jsonify({"error": "both players must be specified"}), 400
    
    summary1 = player_summary(player1)
    summary2 = player_summary(player2)
    
    if summary1 is None or summary2 is None:
        return jsonify({"error": "one or both players not found"}), 404
    
    return jsonify({
        "player1": summary1,
        "player2": summary2
    })

if __name__ == "__main__":
    # debug True for development; switch off for production
    app.run(debug=True, port=5000)
