"""
Bristol Rovers Set Piece Analysis App - WITH DIAGNOSTICS
Complete version with diagnostic tools at the bottom
"""

import streamlit as st
import pandas as pd
import numpy as np
from statsbombpy import sb
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mplsoccer import VerticalPitch
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D

# Page config
st.set_page_config(
    page_title="Set Piece Analysis",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

BR_BLUE = "#0066CC"
BR_DARK_BLUE = "#003D7A"

st.markdown(f"""
<style>
    .main-header {{
        font-size: 2.5rem;
        font-weight: bold;
        color: {BR_BLUE};
        text-align: center;
        padding: 1rem 0;
    }}
    .metric-card {{
        background: linear-gradient(135deg, {BR_BLUE} 0%, {BR_DARK_BLUE} 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    .metric-value {{
        font-size: 2rem;
        font-weight: bold;
    }}
    .metric-label {{
        font-size: 0.9rem;
        opacity: 0.9;
    }}
    .section-header {{
        font-size: 1.5rem;
        font-weight: bold;
        color: {BR_BLUE};
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }}
    .goal-box {{
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid {BR_BLUE};
        margin: 0.5rem 0;
        color: #000000;
    }}
    .goal-box strong {{
        color: {BR_DARK_BLUE};
        font-weight: bold;
    }}
</style>
""", unsafe_allow_html=True)

# Constants
LEAGUES = {
    "League Two 2025/26": {"competition_id": 5, "season_id": 318},
    "League One 2025/26": {"competition_id": 4, "season_id": 318}
}

SB_USERNAME = "dhillon.gil@bristolrovers.co.uk"
SB_PASSWORD = "004laVPb"

ZONE_Y_BINS = [18, 30, 44, 50, 62]
ZONE_Y_LABELS = ['Near Post (18-30)', 'Near-Central (30-44)', 'Central (44-50)', 'Far Post (50-62)']
ZONE_X_BINS = [102, 108, 114, 120]
ZONE_X_LABELS = ['Edge of Box (102-108)', 'Penalty Spot (108-114)', '6-Yard Box (114-120)']

# Data loaders
@st.cache_data(ttl=3600)
def load_teams_for_league(competition_id: int, season_id: int):
    try:
        matches = sb.matches(
            competition_id=competition_id,
            season_id=season_id,
            creds={"user": SB_USERNAME, "passwd": SB_PASSWORD}
        )
        if matches.empty:
            return []
        teams = set(matches['home_team'].unique()) | set(matches['away_team'].unique())
        return sorted(list(teams))
    except Exception as e:
        st.error(f"Error loading teams: {str(e)}")
        return []

@st.cache_data(ttl=3600)
def load_matches(team_name: str, competition_id: int, season_id: int):
    try:
        matches = sb.matches(
            competition_id=competition_id,
            season_id=season_id,
            creds={"user": SB_USERNAME, "passwd": SB_PASSWORD}
        )
        team_matches = matches[
            ((matches['home_team'] == team_name) | (matches['away_team'] == team_name)) &
            (matches['match_status'] == 'available')
        ].copy()
        team_matches['match_label'] = team_matches.apply(
            lambda x: f"{x['match_date']} - {x['home_team']} vs {x['away_team']}", axis=1
        )
        return team_matches.sort_values('match_date', ascending=False)
    except Exception as e:
        st.error(f"Error loading matches: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_events(match_id: int):
    try:
        return sb.events(
            match_id=match_id,
            creds={"user": SB_USERNAME, "passwd": SB_PASSWORD}
        )
    except Exception as e:
        st.error(f"Error loading events: {str(e)}")
        return pd.DataFrame()

def extract_corner_data(events_df: pd.DataFrame, team_name: str = None) -> pd.DataFrame:
    """Extract corner kicks with comprehensive tracking."""
    if events_df.empty:
        return pd.DataFrame()

    corners = events_df[(events_df['type'] == 'Pass') & (events_df['pass_type'] == 'Corner')].copy()
    if corners.empty:
        return pd.DataFrame()
    
    # ==========================================
    # TEMPORARY DEBUG - CHECK FOR PASS_TECHNIQUE
    # ==========================================
    print("\n" + "="*80)
    print("CORNER TECHNIQUE DEBUG")
    print("="*80)
    print(f"Total corners found: {len(corners)}")
    
    # Check if technique data exists
    has_technique = False
    if 'pass_technique' in corners.columns:
        print(f"\n✅ pass_technique column EXISTS!")
        print(f"Sample values (first 5):")
        print(corners['pass_technique'].head())
        has_technique = True
    else:
        print(f"\n❌ pass_technique column NOT FOUND")
    
    if 'pass_technique_name' in corners.columns:
        print(f"\n✅ pass_technique_name column EXISTS!")
        print(f"Sample values (first 5):")
        print(corners['pass_technique_name'].head())
        has_technique = True
    else:
        print(f"\n❌ pass_technique_name column NOT FOUND")
    
    if not has_technique:
        print(f"\n⚠️ No technique columns found - will use coordinate-based calculation")
    
    print("="*80)
    print("END DEBUG")
    print("="*80 + "\n")
    # ==========================================
    # END DEBUG
    # ==========================================

    def safe_extract_coords(location, idx):
        if isinstance(location, (list, tuple)) and len(location) > idx:
            try:
                return float(location[idx])
            except Exception:
                return None
        return None

    corners['start_x'] = corners['location'].apply(lambda x: safe_extract_coords(x, 0))
    corners['start_y'] = corners['location'].apply(lambda x: safe_extract_coords(x, 1))
    corners['end_x'] = corners['pass_end_location'].apply(lambda x: safe_extract_coords(x, 0))
    corners['end_y'] = corners['pass_end_location'].apply(lambda x: safe_extract_coords(x, 1))

    corners = corners.dropna(subset=['start_x', 'start_y', 'end_x', 'end_y'])
    corners = corners[
        (corners['start_x'].between(0, 120)) &
        (corners['start_y'].between(0, 80)) &
        (corners['end_x'].between(0, 120)) &
        (corners['end_y'].between(0, 80))
    ]
    if corners.empty:
        return pd.DataFrame()

    # Determine which goal each team is attacking based on their position in the match
    # StatsBomb coordinates are fixed to the pitch, but we need to know which direction
    # each team is attacking to properly label LEFT/RIGHT corners from viewer perspective
    
    def get_corner_side_viewer_perspective(row):
        """
        Get corner side from viewer's perspective (camera view).
        Takes into account which goal the team is attacking.
        
        In StatsBomb data:
        - X: 0 (own goal) to 120 (opponent goal)
        - Y: 0 (bottom touchline) to 80 (top touchline)
        
        Teams switch ends at halftime, so the same Y coordinate can represent
        different sides depending on which period and which goal they're attacking.
        """
        y = row['start_y']
        period = row.get('period', 1)
        
        # Determine corner side based on Y coordinate and period
        # This assumes StatsBomb orients data with home team attacking left-to-right in period 1
        
        if period == 1:
            # First half: Y < 40 = left side, Y >= 40 = right side
            return 'Left' if y < 40 else 'Right'
        else:
            # Second half: teams switch ends
            # Y < 40 = right side (from camera view), Y >= 40 = left side
            return 'Right' if y < 40 else 'Left'
    
    corners['corner_side'] = corners.apply(get_corner_side_viewer_perspective, axis=1)
    corners['is_short'] = corners['end_x'] < 100

    # FIXED: Match the working script's logic for swing detection
    # Use StatsBomb's pass_technique if available, with proper fallback
    corners['y_diff_calc'] = corners['end_y'] - corners['start_y']
    
    def get_swing_from_technique(row):
        """
        Get swing type using the same logic as the working script.
        Priority: pass_technique field > coordinate calculation
        """
        # Check for pass_technique columns (various possible names)
        for col_name in ['pass_technique_name', 'pass_technique', 'pass.technique.name']:
            if col_name in row.index:
                technique = row.get(col_name)
                if pd.notna(technique):
                    technique_str = str(technique)
                    if "Inswinging" in technique_str or "Inswing" in technique_str:
                        return "Inswing"
                    elif "Outswinging" in technique_str or "Outswing" in technique_str:
                        return "Outswing"
                    elif "Straight" in technique_str:
                        return "Straight"
        
        # Fallback to coordinate-based if no technique data
        # This shouldn't be reached if StatsBomb provides technique
        y_diff = row['y_diff_calc']
        
        # Very conservative threshold - only classify as Inswing if significant curve
        if row['corner_side'] == 'Left':
            # Left corner: only mark as Inswing if LARGE positive movement
            if y_diff > 20.0:
                return "Inswing"
            else:
                return "Outswing"
        else:
            # Right corner: only mark as Inswing if LARGE negative movement
            if y_diff < -20.0:
                return "Inswing"
            else:
                return "Outswing"
    
    corners['swing_type'] = corners.apply(get_swing_from_technique, axis=1)

    corners['zone_y'] = pd.cut(corners['end_y'], bins=ZONE_Y_BINS, labels=ZONE_Y_LABELS, include_lowest=True)
    corners['zone_x'] = pd.cut(corners['end_x'], bins=ZONE_X_BINS, labels=ZONE_X_LABELS, include_lowest=True)
    corners['in_box'] = (
        (corners['end_x'].between(102, 120)) &
        (corners['end_y'].between(18, 62))
    )

    corners['first_contact_won'] = 0
    corners['first_contact_lost'] = 0
    corners['no_touch'] = 0
    corners['total_xg'] = 0.0
    corners['goal_scored'] = 0
    corners['shot_taken'] = 0
    corners['shots_count'] = 0
    corners['first_contact_player'] = ''
    corners['first_contact_method'] = ''  # NEW: Track how first contact was determined
    corners['goal_scorer'] = ''
    corners['shot_body_part'] = ''
    corners['shot_location_x'] = None
    corners['shot_location_y'] = None
    corners['first_contact_x'] = None
    corners['first_contact_y'] = None

    for match_id in corners['match_id'].unique():
        match_corners = corners[corners['match_id'] == match_id].copy()
        # CRITICAL FIX: Don't reset_index - preserve original event indices
        match_events = events_df[events_df['match_id'] == match_id].sort_values('index')
        
        for idx in match_corners.index:
            corner = corners.loc[idx]
            # Find the corner event in match_events
            corner_in_events = match_events[match_events['id'] == corner['id']]
            
            if corner_in_events.empty:
                continue
            
            # Get the position of this corner in the chronological event list
            corner_position = corner_in_events.index[0]
            
            # Get all events after this corner (using iloc position in the sorted dataframe)
            corner_iloc_pos = match_events.index.get_loc(corner_position)
            next_events = match_events.iloc[corner_iloc_pos+1:corner_iloc_pos+51]
            
            if next_events.empty:
                corners.at[idx, 'no_touch'] = 1
                continue
            
            corner_timestamp = corner.get('timestamp')
            if pd.notna(corner_timestamp):
                try:
                    if isinstance(corner_timestamp, str):
                        time_parts = corner_timestamp.split(':')
                        corner_seconds = int(time_parts[0]) * 3600 + int(time_parts[1]) * 60 + float(time_parts[2])
                    else:
                        corner_seconds = float(corner_timestamp)
                    
                    valid_events = []
                    for _, evt in next_events.iterrows():
                        evt_timestamp = evt.get('timestamp')
                        if pd.notna(evt_timestamp):
                            try:
                                if isinstance(evt_timestamp, str):
                                    evt_parts = evt_timestamp.split(':')
                                    evt_seconds = int(evt_parts[0]) * 3600 + int(evt_parts[1]) * 60 + float(evt_parts[2])
                                else:
                                    evt_seconds = float(evt_timestamp)
                                
                                if evt_seconds - corner_seconds <= 10:
                                    valid_events.append(evt)
                            except:
                                valid_events.append(evt)
                        else:
                            valid_events.append(evt)
                    
                    if valid_events:
                        next_events = pd.DataFrame(valid_events)
                    else:
                        corners.at[idx, 'no_touch'] = 1
                        continue
                except:
                    pass

            # ===== IMPROVED: Better first contact detection =====
            # Now properly detects winner by looking at possession after duel
            
            def infer_first_contact(events, corner_team):
                """
                Infer who won first contact by checking:
                1. Explicit duel outcomes
                2. Who gets possession after the duel/event
                3. Which team controls the ball next
                """
                if events.empty:
                    return None, None, None, None
                
                first_event = events.iloc[0]
                event_type = first_event.get('type')
                event_team = first_event.get('team')
                event_player = first_event.get('player', 'Unknown')
                
                # Case 1: Duel with explicit outcome
                if event_type == 'Duel':
                    duel_outcome = first_event.get('duel_outcome')
                    if pd.notna(duel_outcome) and str(duel_outcome) != 'nan':
                        outcome_str = str(duel_outcome).strip()
                        
                        if 'Won' in outcome_str or 'Success' in outcome_str:
                            # This player/team WON the duel
                            won = (event_team == corner_team)
                            return event_player, won, first_event.get('location'), 'duel_won'
                        elif 'Lost' in outcome_str:
                            # This player LOST the duel - opponent won
                            won = (event_team != corner_team)
                            return event_player, won, first_event.get('location'), 'duel_lost'
                    
                    # No explicit outcome - check next event to see who got possession
                    if len(events) > 1:
                        next_event = events.iloc[1]
                        next_team = next_event.get('team')
                        next_player = next_event.get('player', 'Unknown')
                        next_type = next_event.get('type')
                        
                        # Who made the next action?
                        if next_type in ['Pass', 'Ball Receipt*', 'Carry', 'Shot']:
                            # Next team got possession - they won
                            won = (next_team == corner_team)
                            return next_player, won, next_event.get('location'), f'duel_then_{next_type.lower().replace("*", "")}'
                
                # Case 2: Defensive actions (Clearance, Block, Interception)
                # The player who clears/blocks WON the first contact
                # We need to determine: did corner-taking team win or lose?
                if event_type in ['Clearance', 'Block', 'Interception']:
                    # Example: Salford corner, Kilgour (Bristol) clears
                    # event_team = Bristol, corner_team = Salford
                    # Bristol won → Salford lost → won = False
                    
                    if event_team == corner_team:
                        # Same team: corner-taker's teammate cleared (rare, but means they won)
                        won = True
                    else:
                        # Different team: defender cleared → corner team LOST
                        won = False
                    
                    return event_player, won, first_event.get('location'), f'defensive_{event_type.lower()}'
                
                # Case 3: GK action → defender won
                if event_type == 'Goal Keeper':
                    won = (event_team != corner_team)
                    return event_player, won, first_event.get('location'), 'gk_action'
                
                # Case 4: Pass or Ball Receipt → whoever made it controls the ball
                if event_type in ['Pass', 'Ball Receipt*', 'Carry']:
                    # This team has possession
                    won = (event_team == corner_team)
                    return event_player, won, first_event.get('location'), f'possession_{event_type.lower().replace("*", "")}'
                
                # Case 5: Shot → attacker won
                if event_type == 'Shot':
                    won = (event_team == corner_team)
                    return event_player, won, first_event.get('location'), 'shot'
                
                # Case 6: Other events → check next event for possession
                if len(events) > 1:
                    next_event = events.iloc[1]
                    next_team = next_event.get('team')
                    next_player = next_event.get('player', 'Unknown')
                    won = (next_team == corner_team)
                    return next_player, won, next_event.get('location'), f'inferred_from_next_{next_event.get("type", "").lower().replace("*", "")}'
                
                # Fallback
                won = (event_team == corner_team)
                return event_player, won, first_event.get('location'), f'fallback_{event_type.lower()}'
            
            fc_player, fc_won, fc_location, fc_method = infer_first_contact(next_events, corner['team'])
            
            if fc_player is not None:
                corners.at[idx, 'first_contact_player'] = fc_player
                corners.at[idx, 'first_contact_method'] = fc_method
                
                if fc_won:
                    corners.at[idx, 'first_contact_won'] = 1
                    corners.at[idx, 'first_contact_lost'] = 0
                else:
                    corners.at[idx, 'first_contact_won'] = 0
                    corners.at[idx, 'first_contact_lost'] = 1
                
                # Extract location
                if fc_location and isinstance(fc_location, (list, tuple)) and len(fc_location) >= 2:
                    try:
                        fc_x = float(fc_location[0])
                        fc_y = float(fc_location[1])
                        if 80 <= fc_x <= 120 and 0 <= fc_y <= 80:
                            corners.at[idx, 'first_contact_x'] = fc_x
                            corners.at[idx, 'first_contact_y'] = fc_y
                        else:
                            corners.at[idx, 'first_contact_x'] = corner['end_x']
                            corners.at[idx, 'first_contact_y'] = corner['end_y']
                    except:
                        corners.at[idx, 'first_contact_x'] = corner['end_x']
                        corners.at[idx, 'first_contact_y'] = corner['end_y']
                else:
                    corners.at[idx, 'first_contact_x'] = corner['end_x']
                    corners.at[idx, 'first_contact_y'] = corner['end_y']
            else:
                corners.at[idx, 'no_touch'] = 1

            shots = next_events[(next_events['type'] == 'Shot') & (next_events['team'] == corner['team'])]
            
            if not shots.empty:
                corners.at[idx, 'shots_count'] = len(shots)
                corners.at[idx, 'shot_taken'] = 1
                
                primary_shot = shots.iloc[0]
                shot_loc = primary_shot.get('location')
                if shot_loc and isinstance(shot_loc, (list, tuple)) and len(shot_loc) >= 2:
                    corners.at[idx, 'shot_location_x'] = float(shot_loc[0])
                    corners.at[idx, 'shot_location_y'] = float(shot_loc[1])
                
                xg_sum = shots['shot_statsbomb_xg'].fillna(0).sum()
                corners.at[idx, 'total_xg'] = float(xg_sum)
                
                for _, shot in shots.iterrows():
                    outcome = shot.get('shot_outcome')
                    is_goal = False
                    
                    if outcome == 'Goal':
                        is_goal = True
                    elif isinstance(outcome, dict) and outcome.get('name') == 'Goal':
                        is_goal = True
                    
                    if is_goal:
                        corners.at[idx, 'goal_scored'] = 1
                        corners.at[idx, 'goal_scorer'] = shot.get('player', 'Unknown')
                        
                        goal_shot_loc = shot.get('location')
                        if goal_shot_loc and isinstance(goal_shot_loc, (list, tuple)) and len(goal_shot_loc) >= 2:
                            corners.at[idx, 'shot_location_x'] = float(goal_shot_loc[0])
                            corners.at[idx, 'shot_location_y'] = float(goal_shot_loc[1])
                        
                        body_part = shot.get('shot_body_part')
                        if isinstance(body_part, dict):
                            body_part = body_part.get('name', 'Unknown')
                        corners.at[idx, 'shot_body_part'] = str(body_part) if body_part else 'Unknown'
                        break

    if team_name:
        corners = corners[corners['team'] == team_name]

    return corners

def create_comprehensive_delivery_map(corner_df: pd.DataFrame, title: str, team_name: str = "Team"):
    """
    Create comprehensive delivery map showing ALL corner deliveries.
    Shows where the ball landed (end_x, end_y) with color coding for outcome:
    - Green square: Goal scored
    - Red square: Shot taken (no goal)
    - Yellow circle: First contact won
    - Cyan circle: First contact lost  
    - Gray diamond: No touch
    """
    # DEBUG: Print what data we're receiving
    print("\n" + "="*80)
    print("COMPREHENSIVE DELIVERY MAP DEBUG")
    print("="*80)
    print(f"Total corners received: {len(corner_df)}")
    if not corner_df.empty:
        print(f"First Contact Won sum: {corner_df['first_contact_won'].sum()}")
        print(f"First Contact Lost sum: {corner_df['first_contact_lost'].sum()}")
        print("\nFirst few rows:")
        print(corner_df[['minute', 'team', 'first_contact_won', 'first_contact_lost', 
                        'first_contact_player', 'first_contact_method']].head())
    print("="*80 + "\n")
    
    fig, ax = plt.subplots(figsize=(14, 16), facecolor='#1a1a2e')
    pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='#1a1a2e', 
                         line_color='#4a5568', linewidth=2, half=True, pad_top=2, pad_bottom=2)
    pitch.draw(ax=ax)

    if corner_df.empty:
        ax.text(40, 90, 'No corner data available', ha='center', va='center', 
               fontsize=14, color='white')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20, color='white')
        return fig

    valid_corners = corner_df.copy()
    
    goals = 0
    shots = 0
    fc_won = 0
    fc_lost = 0
    no_touch = 0
    
    # Plot each corner with appropriate location based on outcome
    for _, row in valid_corners.iterrows():
        # Determine outcome and styling first
        if row['goal_scored'] == 1:
            # GOALS: Use shot location (where goal was scored from)
            shot_x = row.get('shot_location_x')
            shot_y = row.get('shot_location_y')
            if pd.notna(shot_x) and pd.notna(shot_y):
                plot_x, plot_y = shot_x, shot_y
            elif pd.notna(row.get('first_contact_x')) and pd.notna(row.get('first_contact_y')):
                plot_x, plot_y = row['first_contact_x'], row['first_contact_y']
            else:
                plot_x, plot_y = row['end_x'], row['end_y']
            
            marker, inner_color, outer_color, size = 's', '#00ff00', '#ffffff', 500
            goals += 1
        elif row['shot_taken'] == 1:
            # SHOTS: Use shot location
            shot_x = row.get('shot_location_x')
            shot_y = row.get('shot_location_y')
            if pd.notna(shot_x) and pd.notna(shot_y):
                plot_x, plot_y = shot_x, shot_y
            elif pd.notna(row.get('first_contact_x')) and pd.notna(row.get('first_contact_y')):
                plot_x, plot_y = row['first_contact_x'], row['first_contact_y']
            else:
                plot_x, plot_y = row['end_x'], row['end_y']
            
            marker, inner_color, outer_color, size = 's', '#ff0000', '#ffffff', 450
            shots += 1
        elif row['first_contact_won'] == 1:
            # FIRST CONTACT WON: Use first contact location, fallback to delivery
            if pd.notna(row.get('first_contact_x')) and pd.notna(row.get('first_contact_y')):
                plot_x, plot_y = row['first_contact_x'], row['first_contact_y']
            else:
                plot_x, plot_y = row['end_x'], row['end_y']
            
            marker, inner_color, outer_color, size = 'o', '#ffff00', '#000000', 400
            fc_won += 1
        elif row['first_contact_lost'] == 1:
            # FIRST CONTACT LOST: Use first contact location, fallback to delivery
            if pd.notna(row.get('first_contact_x')) and pd.notna(row.get('first_contact_y')):
                plot_x, plot_y = row['first_contact_x'], row['first_contact_y']
            else:
                plot_x, plot_y = row['end_x'], row['end_y']
            
            marker, inner_color, outer_color, size = 'o', '#00ffff', '#000000', 400
            fc_lost += 1
        else:
            # NO TOUCH: Use delivery location
            plot_x, plot_y = row['end_x'], row['end_y']
            marker, inner_color, outer_color, size = 'D', '#808080', '#ffffff', 300
            no_touch += 1
        
        # Plot if valid coordinates
        if pd.notna(plot_x) and pd.notna(plot_y):
            if 0 <= plot_x <= 120 and 0 <= plot_y <= 80:
                ax.scatter(plot_y, plot_x, marker=marker, s=size, c=inner_color, 
                          edgecolors=outer_color, linewidths=2.5, alpha=0.9, zorder=10)

    # Title
    ax.text(40, 125, title.upper(), ha='center', va='center', 
           fontsize=18, fontweight='bold', color='white', zorder=20)

    # Summary
    summary = f'Total: {len(valid_corners)} | G:{goals} S:{shots} Won:{fc_won} Lost:{fc_lost} None:{no_touch}'
    ax.text(40, 122, summary, ha='center', va='center', fontsize=9, color='yellow', zorder=20)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#00ff00', 
               markersize=14, markeredgecolor='white', markeredgewidth=2, 
               label=f'GOAL ({goals})', linestyle='None'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#ff0000', 
               markersize=13, markeredgecolor='white', markeredgewidth=2, 
               label=f'SHOT ({shots})', linestyle='None'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ffff00', 
               markersize=12, markeredgecolor='black', markeredgewidth=2, 
               label=f'FC WON ({fc_won})', linestyle='None'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#00ffff', 
               markersize=12, markeredgecolor='black', markeredgewidth=2, 
               label=f'FC LOST ({fc_lost})', linestyle='None'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#808080', 
               markersize=10, markeredgecolor='white', markeredgewidth=2, 
               label=f'NO TOUCH ({no_touch})', linestyle='None')
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, 
             framealpha=0.95, facecolor='#2d3748', edgecolor='white', labelcolor='white')
    
    plt.tight_layout()
    return fig


def create_delivery_map(corner_df: pd.DataFrame, title: str, is_defensive: bool = False, team_name: str = "Team"):
    """Create delivery map with proper location hierarchy and filter-aware display."""
    fig, ax = plt.subplots(figsize=(14, 16), facecolor='#1a1a2e')
    pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='#1a1a2e', 
                         line_color='#4a5568', linewidth=2, half=True, pad_top=2, pad_bottom=2)
    pitch.draw(ax=ax)

    if corner_df.empty:
        ax.text(40, 90, 'No corner data available', ha='center', va='center', 
               fontsize=14, color='white')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20, color='white')
        return fig

    valid_corners = corner_df.copy()
    
    goals_plotted = 0
    shots_plotted = 0
    fc_won_plotted = 0
    fc_lost_plotted = 0
    no_touch_plotted = 0
    
    # DEBUG: Track actual coordinates being plotted
    fc_lost_coords = []
    
    # FIXED: Determine what we're filtering for to adjust display priority
    total_corners = len(valid_corners)
    fc_lost_count = (valid_corners['first_contact_lost'] == 1).sum()
    fc_won_count = (valid_corners['first_contact_won'] == 1).sum()
    
    # Only override if we're EXCLUSIVELY filtering (100% of corners match the filter)
    has_only_fc_lost = (fc_lost_count == total_corners) and total_corners > 0
    has_only_fc_won = (fc_won_count == total_corners) and total_corners > 0
    
    print(f"DEBUG: total={total_corners}, fc_lost={fc_lost_count}, fc_won={fc_won_count}, only_lost={has_only_fc_lost}, only_won={has_only_fc_won}")

    for _, row in valid_corners.iterrows():
        plot_x = None
        plot_y = None
        marker = None
        inner_color = None
        outer_color = None
        size = None
        
        # FIXED: If filtering by first contact, show that instead of shots/goals
        if has_only_fc_lost and row['first_contact_lost'] == 1:
            # When filtering "Lost Only", show cyan circles even if there's a shot
            marker, inner_color, outer_color, size = 'o', '#00ffff', '#000000', 400
            plot_x = row.get('first_contact_x')
            plot_y = row.get('first_contact_y')
            if pd.isna(plot_x) or pd.isna(plot_y):
                plot_x = row['end_x']
                plot_y = row['end_y']
            fc_lost_plotted += 1
            fc_lost_coords.append((plot_x, plot_y))
            
        elif has_only_fc_won and row['first_contact_won'] == 1:
            # When filtering "Won Only", show yellow circles even if there's a shot
            marker, inner_color, outer_color, size = 'o', '#ffff00', '#000000', 400
            plot_x = row.get('first_contact_x')
            plot_y = row.get('first_contact_y')
            if pd.isna(plot_x) or pd.isna(plot_y):
                plot_x = row['end_x']
                plot_y = row['end_y']
            fc_won_plotted += 1
            
        # Normal hierarchy when not filtering by first contact
        elif row['goal_scored'] == 1:
            marker, inner_color, outer_color, size = 's', '#00ff00', '#ffffff', 500
            plot_x = row.get('shot_location_x')
            plot_y = row.get('shot_location_y')
            if pd.notna(plot_x) and pd.notna(plot_y):
                goals_plotted += 1
            else:
                plot_x, plot_y = row['end_x'], row['end_y']
                goals_plotted += 1
                
        elif row['shot_taken'] == 1:
            marker, inner_color, outer_color, size = 's', '#ff0000', '#ffffff', 450
            plot_x = row.get('shot_location_x')
            plot_y = row.get('shot_location_y')
            if pd.notna(plot_x) and pd.notna(plot_y):
                shots_plotted += 1
            else:
                plot_x, plot_y = row['end_x'], row['end_y']
                shots_plotted += 1
                
        elif row['first_contact_won'] == 1:
            marker, inner_color, outer_color, size = 'o', '#ffff00', '#000000', 400
            plot_x = row.get('first_contact_x')
            plot_y = row.get('first_contact_y')
            if pd.isna(plot_x) or pd.isna(plot_y):
                plot_x = row['end_x']
                plot_y = row['end_y']
            fc_won_plotted += 1
            
        elif row['first_contact_lost'] == 1:
            marker, inner_color, outer_color, size = 'o', '#00ffff', '#000000', 400
            plot_x = row.get('first_contact_x')
            plot_y = row.get('first_contact_y')
            if pd.isna(plot_x) or pd.isna(plot_y):
                plot_x = row['end_x']
                plot_y = row['end_y']
            fc_lost_plotted += 1
            fc_lost_coords.append((plot_x, plot_y))
            
        else:
            marker, inner_color, outer_color, size = 'D', '#808080', '#ffffff', 300
            plot_x = row['end_x']
            plot_y = row['end_y']
            no_touch_plotted += 1
        
        if pd.notna(plot_x) and pd.notna(plot_y):
            if 0 <= plot_x <= 120 and 0 <= plot_y <= 80:
                ax.scatter(plot_y, plot_x, marker=marker, s=size, c=inner_color, 
                          edgecolors=outer_color, linewidths=2.5, alpha=0.9, zorder=10)

    ax.text(40, 125, title.upper(), ha='center', va='center', 
           fontsize=18, fontweight='bold', color='white', zorder=20)

    # DEBUG: Show coordinates
    debug_text = f'Plotted: G:{goals_plotted} S:{shots_plotted} Won:{fc_won_plotted} Lost:{fc_lost_plotted} NoTouch:{no_touch_plotted}'
    if fc_lost_coords:
        debug_text += f'\nFC Lost coords: {fc_lost_coords[:5]}'  # Show first 5 only
    ax.text(40, 122, debug_text, ha='center', va='center', fontsize=8, color='yellow', zorder=20)

    legend_elements = []
    legend_elements.append(
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#00ff00', 
               markersize=14, markeredgecolor='white', markeredgewidth=2, 
               label=f'GOAL ({int(valid_corners["goal_scored"].sum())})', linestyle='None')
    )
    legend_elements.append(
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#ff0000', 
               markersize=14, markeredgecolor='white', markeredgewidth=2, 
               label=f'SHOT ({int(valid_corners["shot_taken"].sum())})', linestyle='None')
    )
    legend_elements.append(
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#ff8800', 
               markersize=14, markeredgecolor='white', markeredgewidth=2, 
               label='OWN-GOAL (0)', linestyle='None')
    )
    legend_elements.append(
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff00ff', 
               markersize=13, markeredgecolor='white', markeredgewidth=2, 
               label='OTHER TOUCH', linestyle='None')
    )
    
    # FIXED: Context-aware labels for defensive mode
    if is_defensive:
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#ffff00', 
                   markersize=13, markeredgecolor='black', markeredgewidth=2, 
                   label=f'{team_name} WON ({int(valid_corners["first_contact_lost"].sum())})', 
                   linestyle='None')
        )
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#00ffff', 
                   markersize=13, markeredgecolor='black', markeredgewidth=2, 
                   label=f'OPP WON ({int(valid_corners["first_contact_won"].sum())})', 
                   linestyle='None')
        )
    else:
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#ffff00', 
                   markersize=13, markeredgecolor='black', markeredgewidth=2, 
                   label=f'FIRST BALL WON ({int(valid_corners["first_contact_won"].sum())})', 
                   linestyle='None')
        )
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#00ffff', 
                   markersize=13, markeredgecolor='black', markeredgewidth=2, 
                   label=f'FIRST BALL LOST ({int(valid_corners["first_contact_lost"].sum())})', 
                   linestyle='None')
        )
    
    legend_elements.append(
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#808080', 
               markersize=11, markeredgecolor='white', markeredgewidth=2, 
               label=f'NO TOUCH ({int(valid_corners["no_touch"].sum())})', 
               linestyle='None')
    )
    
    legend1 = ax.legend(handles=legend_elements[:4], loc='lower center', 
                       bbox_to_anchor=(0.5, -0.08), ncol=4, fontsize=11, 
                       framealpha=0.95, facecolor='#2d3748', edgecolor='white',
                       labelcolor='white', columnspacing=1.5, handletextpad=0.5)
    
    ax.add_artist(legend1)
    
    ax.legend(handles=legend_elements[4:], loc='lower center', 
             bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=11, 
             framealpha=0.95, facecolor='#2d3748', edgecolor='white',
             labelcolor='white', columnspacing=2, handletextpad=0.5)

    plt.tight_layout()
    return fig

def create_zone_heatmap(corner_df: pd.DataFrame, title: str, metric: str = 'count'):
    """Create zone heatmap with treemap style colors."""
    # DEBUG: Print what data we're receiving
    print("\n" + "="*80)
    print("ZONE HEATMAP DEBUG")
    print("="*80)
    print(f"Metric: {metric}")
    print(f"Total corners received: {len(corner_df)}")
    if not corner_df.empty and 'first_contact_won' in corner_df.columns:
        print(f"First Contact Won sum: {corner_df['first_contact_won'].sum()}")
        print(f"First Contact Lost sum: {corner_df['first_contact_lost'].sum()}")
    print("="*80 + "\n")
    
    fig, ax = plt.subplots(figsize=(10, 14), facecolor='white')
    pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='#e8d5c4', 
                         line_color='#000000', linewidth=2, half=True)
    pitch.draw(ax=ax)

    if corner_df.empty:
        ax.text(40, 90, 'No corner data', ha='center', va='center', fontsize=14, color='black')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20, color='black')
        return fig

    # CRITICAL FIX: For first contact metrics, use first_contact_x/y for zones, not delivery end_x/y
    is_first_contact_metric = metric in ['first_contact_won', 'first_contact_lost', 'metric_to_show']
    
    zone_corners = corner_df.copy()
    
    if is_first_contact_metric:
        # Use first contact locations for zoning
        zone_corners = zone_corners[
            zone_corners['first_contact_x'].notna() & 
            zone_corners['first_contact_y'].notna()
        ].copy()
        
        # Recalculate zones based on first contact location
        zone_corners['zone_y'] = pd.cut(zone_corners['first_contact_y'], bins=ZONE_Y_BINS, labels=ZONE_Y_LABELS, include_lowest=True)
        zone_corners['zone_x'] = pd.cut(zone_corners['first_contact_x'], bins=ZONE_X_BINS, labels=ZONE_X_LABELS, include_lowest=True)
        zone_corners['in_box'] = (
            (zone_corners['first_contact_x'].between(102, 120)) &
            (zone_corners['first_contact_y'].between(18, 62))
        )
    
    # Filter to corners in box with valid zones
    zone_corners = zone_corners[(zone_corners['in_box'] == True) & 
                             zone_corners['zone_y'].notna() & 
                             zone_corners['zone_x'].notna()].copy()
    
    if zone_corners.empty:
        ax.text(40, 90, 'No corners in box', ha='center', va='center', fontsize=14, color='black')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20, color='black')
        return fig

    zone_corners['zone_x_str'] = zone_corners['zone_x'].astype(str)
    zone_corners['zone_y_str'] = zone_corners['zone_y'].astype(str)

    # FIXED: Handle metric properly - check what column actually exists
    if metric == 'xG':
        zone_stats = zone_corners.groupby(['zone_x_str', 'zone_y_str'])['total_xg'].sum().reset_index()
        zone_stats.columns = ['zone_x', 'zone_y', 'value']
        value_format = '.2f'
    elif metric == 'metric_to_show':
        # This is set when in defensive mode
        if 'metric_to_show' in zone_corners.columns:
            zone_stats = zone_corners.groupby(['zone_x_str', 'zone_y_str'])['metric_to_show'].sum().reset_index()
        else:
            # Fallback
            zone_stats = zone_corners.groupby(['zone_x_str', 'zone_y_str']).size().reset_index(name='value')
        zone_stats.columns = ['zone_x', 'zone_y', 'value']
        value_format = '.0f'
    elif metric == 'first_contact_won':
        zone_stats = zone_corners.groupby(['zone_x_str', 'zone_y_str'])['first_contact_won'].sum().reset_index()
        zone_stats.columns = ['zone_x', 'zone_y', 'value']
        value_format = '.0f'
    elif metric == 'first_contact_lost':
        zone_stats = zone_corners.groupby(['zone_x_str', 'zone_y_str'])['first_contact_lost'].sum().reset_index()
        zone_stats.columns = ['zone_x', 'zone_y', 'value']
        value_format = '.0f'
    else:
        # Default: count
        zone_stats = zone_corners.groupby(['zone_x_str', 'zone_y_str']).size().reset_index(name='value')
        zone_stats.columns = ['zone_x', 'zone_y', 'value']
        value_format = '.0f'

    max_value = zone_stats['value'].max() if not zone_stats.empty else 0
    
    zone_bounds = {
        ('Edge of Box (102-108)', 'Near Post (18-30)'): (18, 30, 102, 108),
        ('Edge of Box (102-108)', 'Near-Central (30-44)'): (30, 44, 102, 108),
        ('Edge of Box (102-108)', 'Central (44-50)'): (44, 50, 102, 108),
        ('Edge of Box (102-108)', 'Far Post (50-62)'): (50, 62, 102, 108),
        ('Penalty Spot (108-114)', 'Near Post (18-30)'): (18, 30, 108, 114),
        ('Penalty Spot (108-114)', 'Near-Central (30-44)'): (30, 44, 108, 114),
        ('Penalty Spot (108-114)', 'Central (44-50)'): (44, 50, 108, 114),
        ('Penalty Spot (108-114)', 'Far Post (50-62)'): (50, 62, 108, 114),
        ('6-Yard Box (114-120)', 'Near Post (18-30)'): (18, 30, 114, 120),
        ('6-Yard Box (114-120)', 'Near-Central (30-44)'): (30, 44, 114, 120),
        ('6-Yard Box (114-120)', 'Central (44-50)'): (44, 50, 114, 120),
        ('6-Yard Box (114-120)', 'Far Post (50-62)'): (50, 62, 114, 120),
    }

    def get_color_from_intensity(intensity):
        if intensity == 0:
            return '#f5d5c4'
        elif intensity < 0.25:
            return '#f7b7a3'
        elif intensity < 0.5:
            return '#ea8c7a'
        elif intensity < 0.75:
            return '#c94f3a'
        else:
            return '#8b1a1a'

    for _, row in zone_stats.iterrows():
        zone_key = (row['zone_x'], row['zone_y'])
        if zone_key in zone_bounds:
            y_min, y_max, x_min, x_max = zone_bounds[zone_key]
            
            # FIXED: Initialize intensity before conditional
            intensity = 0
            if max_value > 0:
                intensity = row['value'] / max_value
                color = get_color_from_intensity(intensity)
                alpha = 0.9
            else:
                color = '#f5d5c4'
                alpha = 0.5
            
            rect = patches.Rectangle((y_min, x_min), y_max - y_min, x_max - x_min,
                                     facecolor=color, edgecolor='#000000', linewidth=2.5, 
                                     alpha=alpha, zorder=2)
            ax.add_patch(rect)
            
            text_color = 'white' if intensity > 0.5 else 'black'
            ax.text((y_min + y_max) / 2, (x_min + x_max) / 2, 
                   f'{row["value"]:{value_format}}',
                   ha='center', va='center', fontsize=20, fontweight='bold', 
                   color=text_color, zorder=3)

    ax.set_title(title, fontsize=16, fontweight='bold', pad=20, color='black')
    plt.tight_layout()
    return fig

def create_goals_breakdown(corners: pd.DataFrame, team_name: str, league_name: str, matches_df: pd.DataFrame):
    """Create goals scored breakdown."""
    if corners.empty:
        st.warning("No corner data available")
        return

    header_html = '<div class="section-header">SET PLAYS SCORED</div>'
    st.markdown(header_html, unsafe_allow_html=True)
    
    goals_scored = corners[corners['goal_scored'] == 1].sort_values('minute')
    
    if goals_scored.empty:
        st.info(f"No goals scored from corners for {team_name}")
    else:
        st.write(f"**Total Goals from Corners: {len(goals_scored)}**")
        
        if len(goals_scored) > 0:
            fig, ax = plt.subplots(figsize=(8, 10), facecolor='white')
            pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='#22ab4a', 
                                 line_color='white', linewidth=2, half=True)
            pitch.draw(ax=ax)
            
            for idx, (i, goal) in enumerate(goals_scored.iterrows(), 1):
                # Use shot location if available, fallback to first contact, then delivery
                shot_x = goal.get('shot_location_x')
                shot_y = goal.get('shot_location_y')
                
                if pd.notna(shot_x) and pd.notna(shot_y):
                    plot_x, plot_y = shot_x, shot_y
                elif pd.notna(goal.get('first_contact_x')) and pd.notna(goal.get('first_contact_y')):
                    plot_x, plot_y = goal['first_contact_x'], goal['first_contact_y']
                else:
                    plot_x, plot_y = goal['end_x'], goal['end_y']
                
                ax.scatter(plot_y, plot_x, s=800, c='#FFD700', 
                          edgecolors='#FF0000', linewidths=3, alpha=0.9, zorder=10)
                ax.text(plot_y, plot_x, str(idx), ha='center', va='center',
                       fontsize=16, fontweight='bold', color='black', zorder=11)
            
            ax.set_title('Goal Locations', fontsize=14, fontweight='bold', color='white', pad=10)
            st.pyplot(fig)
            plt.close()
        
        for idx, (i, goal) in enumerate(goals_scored.iterrows(), 1):
            match_id = goal.get('match_id')
            match_detail = "Unknown Match"
            opponent = "Unknown"
            
            if pd.notna(match_id) and not matches_df.empty:
                match_info = matches_df[matches_df['match_id'] == match_id]
                if not match_info.empty:
                    match_row = match_info.iloc[0]
                    home_team = match_row['home_team']
                    away_team = match_row['away_team']
                    home_score = match_row.get('home_score', '?')
                    away_score = match_row.get('away_score', '?')
                    
                    match_detail = f"{home_team} {home_score} - {away_score} {away_team}"
                    opponent = away_team if goal['team'] == home_team else home_team
            
            finish_type = "HEADER"
            shot_body = str(goal.get('shot_body_part', '')).upper()
            if "RIGHT" in shot_body:
                finish_type = "RIGHT FOOT"
            elif "LEFT" in shot_body:
                finish_type = "LEFT FOOT"
            elif "FOOT" in shot_body or "HEAD" not in shot_body:
                finish_type = "FOOT"
            
            corner_type_str = f"CORNER ({goal['corner_side'].upper()})"
            corner_taker = goal.get('player', 'Unknown')
            goal_scorer = goal.get('goal_scorer', 'Unknown')
            pre_goal = f"{goal['swing_type']} corner, taken by {corner_taker}"
            
            minute = goal.get('minute', 'N/A')
            second = goal.get('second', 0)
            if pd.notna(second):
                time_min = f"{minute}:{int(second):02d}"
            else:
                time_min = str(minute)
            
            goal_html = f"""
            <div class="goal-box">
                <strong>GOAL {idx}</strong><br>
                <strong>MATCH:</strong> {match_detail}<br>
                <strong>OPPONENT:</strong> {opponent}<br>
                <strong>FINISH TYPE:</strong> {finish_type}<br>
                <strong>SCORER:</strong> {goal_scorer}<br>
                <strong>TYPE:</strong> {corner_type_str}<br>
                <strong>PRE GOAL:</strong> {pre_goal}<br>
                <strong>TIME:</strong> {time_min}<br>
                <strong>xG:</strong> {goal['total_xg']:.2f}
            </div>
            """
            st.markdown(goal_html, unsafe_allow_html=True)

def main():
    st.markdown('<div class="main-header">⚽ Bristol Rovers Set Piece Analysis</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### League Selection")
        selected_league = st.selectbox("Select League", list(LEAGUES.keys()))
        league_config = LEAGUES[selected_league]
        
        st.markdown("### Team Selection")
        teams = load_teams_for_league(league_config["competition_id"], league_config["season_id"])
        if not teams:
            st.error("Unable to load teams")
            return

        default_team = "Bristol Rovers" if "Bristol Rovers" in teams else teams[0]
        selected_team = st.selectbox("Select Team", teams, index=teams.index(default_team) if default_team in teams else 0)

        matches = load_matches(selected_team, league_config["competition_id"], league_config["season_id"])
        if matches.empty:
            st.warning("No matches available")
            return

        match_options = ["All Matches"] + matches['match_label'].tolist()
        selected_match = st.selectbox("Select Match", match_options)

        st.markdown("---")
        st.markdown("### Filters")

        corner_type = st.radio(
            "Corner Type",
            ["All", f"Offensive ({selected_team})", f"Defensive (Opponents)"]
        )

        corner_sides = st.multiselect("Corner Side", options=['Left', 'Right'], default=['Left', 'Right'])
        delivery_types = st.multiselect("Delivery Type", options=['Inswing', 'Outswing'], default=['Inswing', 'Outswing'])
        first_contact_filter = st.radio("First Contact", ["All", "Won Only", "Lost Only"])
        goals_only = st.checkbox("Goals Only", value=False)

        st.markdown("---")
        st.markdown(f"**Competition:** {selected_league}")

    # Load events
    with st.spinner("Loading match data..."):
        if selected_match == "All Matches":
            all_events = []
            for match_id in matches['match_id']:
                ev = load_events(match_id)
                if not ev.empty:
                    all_events.append(ev)
            events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
            match_info = f"{selected_league} - All Matches"
        else:
            match_idx = match_options.index(selected_match) - 1
            match_id = matches.iloc[match_idx]['match_id']
            match_row = matches.iloc[match_idx]
            events_df = load_events(match_id)
            match_info = f"{match_row['home_team']} {match_row['home_score']} - {match_row['away_score']} {match_row['away_team']} ({match_row['match_date']})"

    if events_df.empty:
        st.error("No events data available")
        return

    all_corners = extract_corner_data(events_df)

    if all_corners.empty:
        st.warning("No corner data available")
        return
    
    # DEBUG: Check team distribution
    if 'team' in all_corners.columns:
        team_counts = all_corners['team'].value_counts()
        print(f"\n=== CORNER TEAM DISTRIBUTION ===")
        print(f"Total corners: {len(all_corners)}")
        print(f"Team distribution:\n{team_counts}")
        print(f"Selected team: {selected_team}")
        print(f"================================\n")

    offensive_corners = all_corners[all_corners['team'] == selected_team].copy()
    defensive_corners = all_corners[all_corners['team'] != selected_team].copy()
    
    # CRITICAL FIX: Double-check filtering worked correctly
    if not offensive_corners.empty:
        wrong_team_off = offensive_corners[offensive_corners['team'] != selected_team]
        if not wrong_team_off.empty:
            st.error(f"⚠️ FILTERING ERROR: {len(wrong_team_off)} corners from wrong team in offensive view!")
            st.write("Wrong team corners:", wrong_team_off[['minute', 'team', 'player']])
    
    if not defensive_corners.empty:
        wrong_team_def = defensive_corners[defensive_corners['team'] == selected_team]
        if not wrong_team_def.empty:
            st.error(f"⚠️ FILTERING ERROR: {len(wrong_team_def)} corners from selected team in defensive view!")
            st.write("Wrong team corners:", wrong_team_def[['minute', 'team', 'player']])
    
    # DEBUG: Verify filtering
    print(f"\n=== AFTER FILTERING ===")
    print(f"Offensive corners (selected team): {len(offensive_corners)}")
    print(f"Defensive corners (opponents): {len(defensive_corners)}")
    if not offensive_corners.empty:
        print(f"Offensive teams: {offensive_corners['team'].unique()}")
    if not defensive_corners.empty:
        print(f"Defensive teams: {defensive_corners['team'].unique()}")
    print(f"=======================\n")

    if corner_type == f"Offensive ({selected_team})":
        corners_df = offensive_corners.copy()
    elif corner_type == f"Defensive (Opponents)":
        corners_df = defensive_corners.copy()
    else:
        # "All" view - need to swap perspective for defensive corners
        corners_df = all_corners.copy()
        
        # For corners where opponent is attacking (team != selected_team),
        # swap won/lost to show from Bristol's defensive perspective
        defensive_mask = corners_df['team'] != selected_team
        
        if defensive_mask.any():
            # Swap only for defensive corners
            temp_won = corners_df.loc[defensive_mask, 'first_contact_won'].copy()
            corners_df.loc[defensive_mask, 'first_contact_won'] = corners_df.loc[defensive_mask, 'first_contact_lost'].copy()
            corners_df.loc[defensive_mask, 'first_contact_lost'] = temp_won

    if corner_sides:
        corners_df = corners_df[corners_df['corner_side'].isin(corner_sides)]
    if delivery_types:
        corners_df = corners_df[corners_df['swing_type'].isin(delivery_types)]
    if first_contact_filter == "Won Only":
        corners_df = corners_df[corners_df['first_contact_won'] == 1]
    elif first_contact_filter == "Lost Only":
        corners_df = corners_df[corners_df['first_contact_lost'] == 1]
    if goals_only:
        corners_df = corners_df[corners_df['goal_scored'] == 1]

    # Overview Stats
    st.markdown('<div class="section-header">📊 Overview Statistics</div>', unsafe_allow_html=True)

    st.markdown(f'### ⚔️ Offensive ({selected_team} Attacking)')
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(offensive_corners)}</div>
            <div class="metric-label">Total Corners</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_goals_off = int(offensive_corners['goal_scored'].sum())
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_goals_off}</div>
            <div class="metric-label">Goals</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        total_xg_off = offensive_corners['total_xg'].sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_xg_off:.2f}</div>
            <div class="metric-label">Total xG</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_xg_off = offensive_corners['total_xg'].mean() if len(offensive_corners) > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_xg_off:.3f}</div>
            <div class="metric-label">Avg xG per Corner</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        # FIXED: Calculate percentage based on corners with first contact, not all corners
        fc_won = int(offensive_corners['first_contact_won'].sum())
        fc_total = int(offensive_corners['first_contact_won'].sum() + offensive_corners['first_contact_lost'].sum())
        fc_rate = (fc_won / fc_total * 100) if fc_total > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{fc_rate:.1f}%</div>
            <div class="metric-label">First Contact Won</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f'### 🛡️ Defensive ({selected_team} Defending)')
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(defensive_corners)}</div>
            <div class="metric-label">Corners Faced</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_goals_def = int(defensive_corners['goal_scored'].sum())
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_goals_def}</div>
            <div class="metric-label">Goals Conceded</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        total_xg_def = defensive_corners['total_xg'].sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_xg_def:.2f}</div>
            <div class="metric-label">Total xG Against</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_xg_def = defensive_corners['total_xg'].mean() if len(defensive_corners) > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_xg_def:.3f}</div>
            <div class="metric-label">Avg xG Against</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        # FIXED: Calculate percentage based on corners with first contact, not all corners
        fc_def_won = int(defensive_corners['first_contact_lost'].sum())
        fc_def_total = int(defensive_corners['first_contact_won'].sum() + defensive_corners['first_contact_lost'].sum())
        fc_def_rate = (fc_def_won / fc_def_total * 100) if fc_def_total > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{fc_def_rate:.1f}%</div>
            <div class="metric-label">{selected_team} Won %</div>
        </div>
        """, unsafe_allow_html=True)

    # Data Validation
    st.markdown('<div class="section-header">🔍 Data Validation</div>', unsafe_allow_html=True)
    
    with st.expander("🎯 Goal Summary (Debug Info)", expanded=False):
        st.write(f"**Total offensive corners analyzed:** {len(offensive_corners)}")
        st.write(f"**Total defensive corners analyzed:** {len(defensive_corners)}")
        st.write(f"**Total goals in offensive corners:** {int(offensive_corners['goal_scored'].sum())}")
        st.write(f"**Total goals in defensive corners (conceded):** {int(defensive_corners['goal_scored'].sum())}")
        
        st.write("\n**Swing Type Distribution (All Offensive Corners):**")
        if not offensive_corners.empty:
            swing_dist = offensive_corners['swing_type'].value_counts()
            st.write(f"  - Inswing: {swing_dist.get('Inswing', 0)}")
            st.write(f"  - Outswing: {swing_dist.get('Outswing', 0)}")
            
            st.write("\n**Sample Y-coordinate changes (first 10 corners):**")
            for idx, row in offensive_corners.head(10).iterrows():
                y_diff = row['end_y'] - row['start_y']
                st.write(f"  - {row['corner_side']} corner: start_y={row['start_y']:.1f}, end_y={row['end_y']:.1f}, y_diff={y_diff:.2f} → {row['swing_type']}")
        
        st.write("\n**First Contact Distribution (Current Filter):**")
        if not corners_df.empty:
            st.write(f"  - First Contact Won: {int(corners_df['first_contact_won'].sum())}")
            st.write(f"  - First Contact Lost: {int(corners_df['first_contact_lost'].sum())}")
            st.write(f"  - No Touch: {int(corners_df['no_touch'].sum())}")
            
            fc_with_location = corners_df[
                (corners_df['first_contact_lost'] == 1) & 
                (corners_df['first_contact_x'].notna()) & 
                (corners_df['first_contact_y'].notna())
            ]
            st.write(f"  - First Contact Lost WITH location data: {len(fc_with_location)}")
            
            fc_without_location = corners_df[
                (corners_df['first_contact_lost'] == 1) & 
                ((corners_df['first_contact_x'].isna()) | (corners_df['first_contact_y'].isna()))
            ]
            st.write(f"  - First Contact Lost WITHOUT location data (will use delivery location): {len(fc_without_location)}")
        
        if not offensive_corners.empty:
            goals_off = offensive_corners[offensive_corners['goal_scored'] == 1]
            if not goals_off.empty:
                st.write(f"\n**Offensive goals breakdown ({len(goals_off)} goals):**")
                for _, g in goals_off.iterrows():
                    corner_taker = g.get('player', 'Unknown')
                    y_diff = g['end_y'] - g['start_y']
                    st.write(f"  - Minute {g.get('minute', 'N/A')}: {g['corner_side']} corner, y_diff={y_diff:.2f}, Swing={g['swing_type']}, Taken by {corner_taker}, xG: {g['total_xg']:.2f}")
            else:
                st.write("**No offensive goals found in data**")
        
        if not defensive_corners.empty:
            goals_def = defensive_corners[defensive_corners['goal_scored'] == 1]
            if not goals_def.empty:
                st.write(f"\n**Defensive goals conceded ({len(goals_def)} goals):**")
                for _, g in goals_def.iterrows():
                    corner_taker = g.get('player', 'Unknown')
                    y_diff = g['end_y'] - g['start_y']
                    st.write(f"  - Minute {g.get('minute', 'N/A')}: {g['corner_side']} corner, y_diff={y_diff:.2f}, Swing={g['swing_type']}, Taken by {corner_taker}, xG: {g['total_xg']:.2f}")
            else:
                st.write("**No defensive goals found in data**")
        
        st.write(f"\n**Total shots from offensive corners:** {int(offensive_corners['shots_count'].sum())}")
        st.write(f"**Total xG from offensive corners:** {offensive_corners['total_xg'].sum():.2f}")
    
    with st.expander("📋 View Detailed Corner Data with Players", expanded=False):
        if not corners_df.empty:
            st.write("**Sample corners to verify swing type calculation:**")
            if len(corners_df) > 0:
                sample_corners = corners_df[['player', 'minute', 'corner_side', 'start_y', 'end_y', 'swing_type', 
                                             'first_contact_won', 'first_contact_lost', 'first_contact_x', 
                                             'first_contact_y']].head(15)
                sample_corners = sample_corners.copy()
                sample_corners['y_diff'] = sample_corners['end_y'] - sample_corners['start_y']
                sample_corners['has_fc_location'] = sample_corners['first_contact_x'].notna() & sample_corners['first_contact_y'].notna()
                st.dataframe(sample_corners, use_container_width=True)
            
            st.write(f"\n**Full corner details (first 30):**")
            display_cols = ['player', 'team', 'minute', 'corner_side', 'swing_type', 
                            'end_x', 'end_y', 'zone_x', 'zone_y', 'in_box', 
                            'first_contact_won', 'first_contact_lost', 'first_contact_player',
                            'total_xg', 'goal_scored', 'goal_scorer', 'shots_count']
            available_cols = [col for col in display_cols if col in corners_df.columns]
            st.dataframe(corners_df[available_cols].head(30), use_container_width=True)
        else:
            st.info("No data available with current filters")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Delivery Map", "🎯 Zone Analysis", "📊 Statistics", "🎯 Goals Breakdown"])

    with tab1:
        st.markdown('<div class="section-header">Corner Delivery Map</div>', unsafe_allow_html=True)
        
        # Add toggle for comprehensive view
        st.info("🎯 **Comprehensive View**: Shows ALL corner deliveries at their landing locations, color-coded by outcome")
        
        if corner_type == f"Defensive (Opponents)":
            st.info("💡 **Defensive Mode**: Showing where opponents deliver corners against your team")
        
        if not corners_df.empty:
            # CRITICAL FIX: In defensive mode, swap won/lost columns for visualizations
            # The raw data is from corner-taker's perspective
            # In defensive mode, we want defending team's perspective
            is_defensive = (corner_type == f"Defensive (Opponents)")
            
            viz_df = corners_df.copy()
            if is_defensive:
                # Swap the columns: 
                # What was "won" by attacker = "lost" by defender
                # What was "lost" by attacker = "won" by defender
                temp_won = viz_df['first_contact_won'].copy()
                viz_df['first_contact_won'] = viz_df['first_contact_lost'].copy()
                viz_df['first_contact_lost'] = temp_won
            
            # Show comprehensive delivery map with swapped data
            fig = create_comprehensive_delivery_map(viz_df, f"{selected_team} - {corner_type} - ALL DELIVERIES", selected_team)
            st.pyplot(fig)
            plt.close()
            
            st.markdown("### Corner Details")
            corner_details = corners_df.copy()
            
            if 'match_id' in corner_details.columns:
                opponent_list = []
                for _, corner in corner_details.iterrows():
                    match_id = corner.get('match_id')
                    if pd.notna(match_id):
                        match_info = matches[matches['match_id'] == match_id]
                        if not match_info.empty:
                            match_row = match_info.iloc[0]
                            if corner['team'] == match_row['home_team']:
                                opponent = match_row['away_team']
                            else:
                                opponent = match_row['home_team']
                            opponent_list.append(opponent)
                        else:
                            opponent_list.append('Unknown')
                    else:
                        opponent_list.append('Unknown')
                corner_details['opponent'] = opponent_list
            
            display_cols = ['player', 'opponent', 'minute', 'corner_side', 'swing_type', 
                          'goal_scored', 'goal_scorer', 'shot_taken', 'first_contact_won', 
                          'first_contact_lost', 'first_contact_player', 'total_xg']
            
            available_cols = [col for col in display_cols if col in corner_details.columns]
            display_df = corner_details[available_cols].copy()
            
            # FIXED: Context-aware column labels AND values based on mode
            if corner_type == f"Defensive (Opponents)":
                # For defensive mode, SWAP the won/lost values
                # because first_contact_won means opponent won (= our team lost)
                # and first_contact_lost means opponent lost (= our team won)
                if 'first_contact_won' in display_df.columns and 'first_contact_lost' in display_df.columns:
                    temp_won = display_df['first_contact_won'].copy()
                    display_df['first_contact_won'] = display_df['first_contact_lost']
                    display_df['first_contact_lost'] = temp_won
                
                col_rename = {
                    'player': 'Corner Taker',
                    'opponent': 'Opponent',
                    'minute': 'Min',
                    'corner_side': 'Side',
                    'swing_type': 'Delivery',
                    'goal_scored': 'Goal',
                    'goal_scorer': 'Scorer',
                    'shot_taken': 'Shot',
                    'first_contact_won': f'{selected_team} Won',  # Now shows actual defending team wins
                    'first_contact_lost': f'Opp Won',  # Now shows actual opponent wins
                    'first_contact_player': 'First Contact',
                    'total_xg': 'xG'
                }
            else:
                col_rename = {
                    'player': 'Corner Taker',
                    'opponent': 'Opponent',
                    'minute': 'Min',
                    'corner_side': 'Side',
                    'swing_type': 'Delivery',
                    'goal_scored': 'Goal',
                    'goal_scorer': 'Scorer',
                    'shot_taken': 'Shot',
                    'first_contact_won': 'Won',
                    'first_contact_lost': 'Lost',
                    'first_contact_player': 'First Contact',
                    'total_xg': 'xG'
                }
            
            display_df = display_df.rename(columns=col_rename)
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("No corner data available for selected filters")

    with tab2:
        st.markdown('<div class="section-header">Zone Analysis</div>', unsafe_allow_html=True)
        
        if corner_type == f"Offensive ({selected_team})":
            metric_options = ["count", "xG", "first_contact_won", "first_contact_lost"]
            metric_labels = {
                "count": "Delivery Count", 
                "xG": "xG Threat", 
                "first_contact_won": "First Contacts Won",
                "first_contact_lost": "First Contacts Lost"
            }
        elif corner_type == f"Defensive (Opponents)":
            # UPDATED: Separate dropdown options for defensive first contact analysis
            metric_options = ["count", "xG", "defensive_first_contact_won", "defensive_first_contact_loss"]
            metric_labels = {
                "count": "Delivery Count", 
                "xG": "xG Against", 
                "defensive_first_contact_won": f"First Contact Won (Defensive)",
                "defensive_first_contact_loss": f"First Contact Loss (Defensive)"
            }
            st.info(f"💡 **Defensive Mode**: 'First Contact Won' shows where {selected_team} won the first contact (cleared). 'First Contact Loss' shows where opponents won first contact against {selected_team} (danger zones).")
        else:
            metric_options = ["count", "xG", "first_contact_won", "first_contact_lost"]
            metric_labels = {
                "count": "Delivery Count", 
                "xG": "Total xG", 
                "first_contact_won": "First Contacts Won",
                "first_contact_lost": "First Contacts Lost"
            }

        selected_metric = st.selectbox("Select Metric", metric_options, format_func=lambda x: metric_labels[x])
        
        if not corners_df.empty:
            # UPDATED: Handle the new defensive first contact options
            if corner_type == f"Defensive (Opponents)":
                display_df = corners_df.copy()
                if selected_metric == "defensive_first_contact_won":
                    # Show where selected team won defensively (cleared the ball)
                    # In defensive corners, 'first_contact_lost' means the opponent (attacking team) lost = our team won
                    display_df['metric_to_show'] = display_df['first_contact_lost']
                    fig = create_zone_heatmap(display_df, f"Zone Analysis - {metric_labels[selected_metric]}", "metric_to_show")
                elif selected_metric == "defensive_first_contact_loss":
                    # Show where opponents won first contact (danger zones for our team)
                    # In defensive corners, 'first_contact_won' means the opponent (attacking team) won = our team lost
                    display_df['metric_to_show'] = display_df['first_contact_won']
                    fig = create_zone_heatmap(display_df, f"Zone Analysis - {metric_labels[selected_metric]}", "metric_to_show")
                else:
                    # Count and xG - no change
                    fig = create_zone_heatmap(corners_df, f"Zone Analysis - {metric_labels[selected_metric]}", selected_metric)
            else:
                # Offensive or All - use normal columns
                fig = create_zone_heatmap(corners_df, f"Zone Analysis - {metric_labels[selected_metric]}", selected_metric)
            
            # FIXED: Add explanation for first contact zone filtering
            is_fc_metric_check = selected_metric in ['first_contact_won', 'first_contact_lost', 
                                                     'defensive_first_contact_won', 'defensive_first_contact_loss']
            
            if is_fc_metric_check:
                # Count total first contacts vs those in box
                fc_data = corners_df.copy()
                if selected_metric == 'first_contact_won' or selected_metric == 'defensive_first_contact_won':
                    if selected_metric == 'defensive_first_contact_won':
                        total_fc = int(corners_df['first_contact_lost'].sum())  # defending team won = attacking team lost
                    else:
                        total_fc = int(corners_df['first_contact_won'].sum())
                else:
                    if selected_metric == 'defensive_first_contact_loss':
                        total_fc = int(corners_df['first_contact_won'].sum())  # defending team lost = attacking team won
                    else:
                        total_fc = int(corners_df['first_contact_lost'].sum())
                
                # Count how many are in defined zones
                fc_with_location = corners_df[
                    (corners_df['first_contact_x'].notna()) & 
                    (corners_df['first_contact_y'].notna())
                ].copy()
                
                if not fc_with_location.empty:
                    fc_with_location['fc_in_box'] = (
                        (fc_with_location['first_contact_x'].between(102, 120)) &
                        (fc_with_location['first_contact_y'].between(18, 62))
                    )
                    
                    # Filter based on metric type
                    if selected_metric == 'first_contact_won':
                        fc_with_location = fc_with_location[fc_with_location['first_contact_won'] == 1]
                    elif selected_metric == 'first_contact_lost':
                        fc_with_location = fc_with_location[fc_with_location['first_contact_lost'] == 1]
                    elif selected_metric == 'defensive_first_contact_won':
                        fc_with_location = fc_with_location[fc_with_location['first_contact_lost'] == 1]
                    elif selected_metric == 'defensive_first_contact_loss':
                        fc_with_location = fc_with_location[fc_with_location['first_contact_won'] == 1]
                    
                    in_box_count = int(fc_with_location['fc_in_box'].sum())
                    outside_box = total_fc - in_box_count
                    
                    if outside_box > 0:
                        st.info(f"📊 **Zone Filtering:** {total_fc} total first contacts - {in_box_count} shown in zones (inside box), {outside_box} occurred outside defined zones")
            
            st.pyplot(fig)
            plt.close()
            
            st.markdown("### Zone Breakdown with Player Details")
            
            # CRITICAL FIX: Use first contact locations for zoning when showing first contact metrics
            is_fc_metric = selected_metric in ['first_contact_won', 'first_contact_lost', 
                                               'defensive_first_contact_won', 'defensive_first_contact_loss']
            
            if is_fc_metric:
                # Recalculate zones based on first contact locations
                zone_data = corners_df[
                    corners_df['first_contact_x'].notna() & 
                    corners_df['first_contact_y'].notna()
                ].copy()
                
                if not zone_data.empty:
                    zone_data['zone_y'] = pd.cut(zone_data['first_contact_y'], bins=ZONE_Y_BINS, 
                                                 labels=ZONE_Y_LABELS, include_lowest=True)
                    zone_data['zone_x'] = pd.cut(zone_data['first_contact_x'], bins=ZONE_X_BINS, 
                                                 labels=ZONE_X_LABELS, include_lowest=True)
                    zone_data['in_box'] = (
                        (zone_data['first_contact_x'].between(102, 120)) &
                        (zone_data['first_contact_y'].between(18, 62))
                    )
                    zone_data = zone_data[zone_data['in_box'] == True].copy()
            else:
                # Use delivery locations for other metrics
                zone_data = corners_df[corners_df['in_box'] == True].copy()
            
            if not zone_data.empty:
                if 'opponent' not in zone_data.columns and 'match_id' in zone_data.columns:
                    opponent_list = []
                    for _, corner in zone_data.iterrows():
                        match_id = corner.get('match_id')
                        if pd.notna(match_id):
                            match_info = matches[matches['match_id'] == match_id]
                            if not match_info.empty:
                                match_row = match_info.iloc[0]
                                if corner['team'] == match_row['home_team']:
                                    opponent = match_row['away_team']
                                else:
                                    opponent = match_row['home_team']
                                opponent_list.append(opponent)
                            else:
                                opponent_list.append('Unknown')
                        else:
                            opponent_list.append('Unknown')
                    zone_data = zone_data.copy()
                    zone_data['opponent'] = opponent_list
                
                zone_summary_list = []
                for (zx, zy), group in zone_data.groupby(['zone_x', 'zone_y'], observed=True):
                    takers = ', '.join(group['player'].unique()[:3])
                    opponents = ', '.join(group['opponent'].unique()[:3]) if 'opponent' in group.columns else 'N/A'
                    scorers = group[group['goal_scored'] == 1]['goal_scorer'].unique()
                    scorer_str = ', '.join([s for s in scorers if pd.notna(s) and s != '']) if len(scorers) > 0 else '-'
                    
                    # FIXED: Swap won/lost for defensive view
                    if corner_type == f"Defensive (Opponents)":
                        # In defensive mode, corners are by opponents
                        # first_contact_won (opponent won) = defending team lost
                        # first_contact_lost (opponent lost) = defending team won
                        won_value = int(group['first_contact_lost'].sum())
                        lost_value = int(group['first_contact_won'].sum())
                    else:
                        won_value = int(group['first_contact_won'].sum())
                        lost_value = int(group['first_contact_lost'].sum())
                    
                    zone_summary_list.append({
                        'Zone X': str(zx),
                        'Zone Y': str(zy),
                        'Count': len(group),
                        'Corner Takers': takers,
                        'Opponents': opponents,
                        'Total xG': f"{group['total_xg'].sum():.2f}",
                        'Goals': int(group['goal_scored'].sum()),
                        'Scorers': scorer_str,
                        'Won': won_value,
                        'Lost': lost_value
                    })
                
                if zone_summary_list:
                    zone_summary_df = pd.DataFrame(zone_summary_list)
                    st.dataframe(zone_summary_df, use_container_width=True, hide_index=True)
            else:
                st.info("No corners in box with current filters")
        else:
            st.info("No corner data available for selected filters")

    with tab3:
        st.markdown('<div class="section-header">Comparative Statistics</div>', unsafe_allow_html=True)
        
        def calc_stats(df, label):
            if df.empty:
                return {}
            return {
                'Team': label,
                'Total Corners': len(df),
                'Goals': int(df['goal_scored'].sum()),
                'Total xG': f"{df['total_xg'].sum():.2f}",
                'Avg xG per Corner': f"{df['total_xg'].mean():.3f}",
                'Shots': int(df['shots_count'].sum()),
                'First Contact Won': int(df['first_contact_won'].sum()),
                'First Contact Lost': int(df['first_contact_lost'].sum()),
                'FC Win %': f"{(df['first_contact_won'].sum() / len(df) * 100):.1f}%" if len(df) > 0 else "0%"
            }
        
        stats_data = []
        if not offensive_corners.empty:
            stats_data.append(calc_stats(offensive_corners, f"{selected_team} (Attacking)"))
        if not defensive_corners.empty:
            stats_data.append(calc_stats(defensive_corners, "Opponents"))
        
        if stats_data:
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

        st.markdown("### Breakdown by Side and Delivery Type")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Corner Side Distribution**")
            if not corners_df.empty:
                side_stats = corners_df.groupby('corner_side', observed=True).agg({
                    'total_xg': 'sum',
                    'goal_scored': 'sum',
                    'first_contact_won': 'sum'
                }).reset_index()
                side_stats.insert(1, 'Count', corners_df.groupby('corner_side', observed=True).size().values)
                side_stats.columns = ['Side', 'Count', 'Total xG', 'Goals', 'First Contact Won']
                st.dataframe(side_stats, use_container_width=True, hide_index=True)
            else:
                st.info("No data with current filters")
        
        with col2:
            st.markdown("**Delivery Type Distribution**")
            if not corners_df.empty:
                delivery_stats = corners_df.groupby('swing_type', observed=True).agg({
                    'total_xg': 'sum',
                    'goal_scored': 'sum',
                    'first_contact_won': 'sum'
                }).reset_index()
                delivery_stats.insert(1, 'Count', corners_df.groupby('swing_type', observed=True).size().values)
                delivery_stats.columns = ['Delivery', 'Count', 'Total xG', 'Goals', 'First Contact Won']
                st.dataframe(delivery_stats, use_container_width=True, hide_index=True)
            else:
                st.info("No data with current filters")
        
        st.markdown("### First Contact Winners")
        if not corners_df.empty and 'first_contact_player' in corners_df.columns:
            fc_data = corners_df[corners_df['first_contact_won'] == 1]
            if not fc_data.empty:
                fc_players = fc_data.groupby('first_contact_player').size().reset_index(name='First Contacts Won')
                fc_players = fc_players.sort_values('First Contacts Won', ascending=False).head(10)
                st.dataframe(fc_players, use_container_width=True, hide_index=True)
            else:
                st.info("No first contact data available")

    with tab4:
        st.markdown('<div class="section-header">Goals Breakdown</div>', unsafe_allow_html=True)
        
        create_goals_breakdown(offensive_corners, selected_team, selected_league, matches)
        
        st.markdown("---")
        
        conceded_header = '<div class="section-header">SET PLAYS CONCEDED</div>'
        st.markdown(conceded_header, unsafe_allow_html=True)
        
        goals_conceded = defensive_corners[defensive_corners['goal_scored'] == 1].sort_values('minute')
        
        if goals_conceded.empty:
            st.info(f"No goals conceded from corners for {selected_team}")
        else:
            st.write(f"**Total Goals Conceded from Corners: {len(goals_conceded)}**")
            
            if len(goals_conceded) > 0:
                fig, ax = plt.subplots(figsize=(8, 10), facecolor='white')
                pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='#22ab4a', 
                                     line_color='white', linewidth=2, half=True)
                pitch.draw(ax=ax)
                
                for idx, (i, goal) in enumerate(goals_conceded.iterrows(), 1):
                    # Use shot location if available, fallback to first contact, then delivery
                    shot_x = goal.get('shot_location_x')
                    shot_y = goal.get('shot_location_y')
                    
                    if pd.notna(shot_x) and pd.notna(shot_y):
                        plot_x, plot_y = shot_x, shot_y
                    elif pd.notna(goal.get('first_contact_x')) and pd.notna(goal.get('first_contact_y')):
                        plot_x, plot_y = goal['first_contact_x'], goal['first_contact_y']
                    else:
                        plot_x, plot_y = goal['end_x'], goal['end_y']
                    
                    ax.scatter(plot_y, plot_x, s=800, c='#FF4444', 
                              edgecolors='#8B0000', linewidths=3, alpha=0.9, zorder=10)
                    ax.text(plot_y, plot_x, str(idx), ha='center', va='center',
                           fontsize=16, fontweight='bold', color='white', zorder=11)
                
                ax.set_title('Goals Conceded Locations', fontsize=14, fontweight='bold', 
                           color='white', pad=10)
                st.pyplot(fig)
                plt.close()
            
            for idx, (i, goal) in enumerate(goals_conceded.iterrows(), 1):
                match_id = goal.get('match_id')
                match_detail = "Unknown Match"
                opponent = goal['team']
                
                if pd.notna(match_id) and not matches.empty:
                    match_info = matches[matches['match_id'] == match_id]
                    if not match_info.empty:
                        match_row = match_info.iloc[0]
                        home_team = match_row['home_team']
                        away_team = match_row['away_team']
                        home_score = match_row.get('home_score', '?')
                        away_score = match_row.get('away_score', '?')
                        
                        match_detail = f"{home_team} {home_score} - {away_score} {away_team}"
                
                finish_type = "HEADER"
                shot_body = str(goal.get('shot_body_part', '')).upper()
                if "RIGHT" in shot_body:
                    finish_type = "RIGHT FOOT"
                elif "LEFT" in shot_body:
                    finish_type = "LEFT FOOT"
                elif "FOOT" in shot_body or "HEAD" not in shot_body:
                    finish_type = "FOOT"
                
                corner_type_str = f"CORNER ({goal['corner_side'].upper()})"
                corner_taker = goal.get('player', 'Unknown')
                goal_scorer = goal.get('goal_scorer', 'Unknown')
                pre_goal = f"{goal['swing_type']} corner by {opponent}, taken by {corner_taker}"
                
                minute = goal.get('minute', 'N/A')
                second = goal.get('second', 0)
                if pd.notna(second):
                    time_min = f"{minute}:{int(second):02d}"
                else:
                    time_min = str(minute)
                
                goal_box_html = f"""
                <div class="goal-box">
                    <strong>GOAL CONCEDED {idx}</strong><br>
                    <strong>MATCH:</strong> {match_detail}<br>
                    <strong>OPPONENT:</strong> {opponent}<br>
                    <strong>FINISH TYPE:</strong> {finish_type}<br>
                    <strong>SCORER:</strong> {goal_scorer}<br>
                    <strong>TYPE:</strong> {corner_type_str}<br>
                    <strong>PRE GOAL:</strong> {pre_goal}<br>
                    <strong>TIME:</strong> {time_min}<br>
                    <strong>xG Against:</strong> {goal['total_xg']:.2f}
                </div>
                """
                st.markdown(goal_box_html, unsafe_allow_html=True)
    
    # ==========================================
    # NEW: FIRST CONTACT DETECTION ANALYSIS
    # ==========================================
    st.markdown("---")
    st.markdown("## 🔍 First Contact Detection Analysis")
    st.info("Shows how first contact winner was determined for each corner (now includes inference when duel data is missing)")
    
    if not all_corners.empty and 'first_contact_method' in all_corners.columns:
        # Overall stats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total = len(all_corners)
            fc_won = int(all_corners['first_contact_won'].sum())
            fc_lost = int(all_corners['first_contact_lost'].sum())
            st.metric("Total Corners", total)
            st.metric("First Contact Won", f"{fc_won} ({fc_won/total*100:.1f}%)")
            st.metric("First Contact Lost", f"{fc_lost} ({fc_lost/total*100:.1f}%)")
        
        with col2:
            method_counts = all_corners['first_contact_method'].value_counts()
            st.write("**Detection Methods:**")
            for method, count in method_counts.items():
                if method:
                    st.write(f"• {method}: {count}")
        
        with col3:
            st.write("**Method Explanations:**")
            st.markdown("""
            - `duel_won/lost`: StatsBomb duel outcome
            - `defensive_*`: Clear/Block/Interception
            - `possession_*`: Inferred from team
            - `gk_action`: GK saved/collected
            - `shot`: Attacker took shot
            """)
        
        # Show sample data
        st.markdown("### Sample Corner First Contacts")
        sample_cols = ['minute', 'team', 'corner_side', 'first_contact_won', 'first_contact_lost', 
                      'first_contact_player', 'first_contact_method']
        available_cols = [col for col in sample_cols if col in all_corners.columns]
        st.dataframe(all_corners[available_cols].head(20), use_container_width=True)
    
    # ==========================================
    # DIAGNOSTIC SECTION - INSIDE MAIN FUNCTION
    # ==========================================
    
    st.markdown("---")
    st.markdown("---")
    st.markdown("## 🔍 DIAGNOSTIC MODE")
    st.info("This section shows detailed analysis of corner data to debug visualization issues")
    
    if all_corners.empty:
        st.warning("⚠️ No corner data available.")
    else:
        diagnostic_corners = all_corners.copy()
        st.success(f"✅ Loaded {len(diagnostic_corners)} corners from current selection")
        
        # ==========================================
        # SECTION 1: Y-DIFF ANALYSIS (Swing Type)
        # ==========================================
        st.markdown("---")
        st.subheader("1️⃣ Y-Coordinate Analysis (Swing Type Detection)")
        
        y_diff_data = diagnostic_corners[['minute', 'team', 'player', 'corner_side', 'start_y', 'end_y']].copy()
        y_diff_data['y_diff'] = y_diff_data['end_y'] - y_diff_data['start_y']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Corners", len(y_diff_data))
            st.metric("Left Corners", len(y_diff_data[y_diff_data['corner_side'] == 'Left']))
        with col2:
            st.metric("Right Corners", len(y_diff_data[y_diff_data['corner_side'] == 'Right']))
            st.metric("Mean Y-diff", f"{y_diff_data['y_diff'].mean():.2f}")
        with col3:
            st.metric("Min Y-diff", f"{y_diff_data['y_diff'].min():.2f}")
            st.metric("Max Y-diff", f"{y_diff_data['y_diff'].max():.2f}")
        
        st.write("**Y-diff Distribution:**")
        st.dataframe(y_diff_data.describe())
        
        # Test different thresholds
        st.markdown("### 🎯 Test Swing Classification Thresholds")
        
        test_threshold = st.slider("Test threshold value", 0.1, 60.0, 15.0, 0.5)
        
        def test_swing(row, thresh):
            y_diff = row['y_diff']
            if row['corner_side'] == 'Left':
                return 'Inswing' if y_diff > thresh else 'Outswing'
            else:
                return 'Inswing' if y_diff < -thresh else 'Outswing'
        
        y_diff_data['test_swing'] = y_diff_data.apply(lambda r: test_swing(r, test_threshold), axis=1)
        
        col1, col2 = st.columns(2)
        with col1:
            inswing_count = len(y_diff_data[y_diff_data['test_swing'] == 'Inswing'])
            st.metric(f"Inswing @ {test_threshold}", inswing_count)
        with col2:
            outswing_count = len(y_diff_data[y_diff_data['test_swing'] == 'Outswing'])
            st.metric(f"Outswing @ {test_threshold}", outswing_count)
        
        st.write("**All corners with test classification:**")
        st.dataframe(y_diff_data[['minute', 'team', 'corner_side', 'y_diff', 'test_swing']].sort_values('minute'))
        
        # Histogram
        st.write("**Y-diff Histogram:**")
        st.bar_chart(y_diff_data['y_diff'].value_counts().sort_index())

        # ==========================================
        # SECTION 2: FIRST CONTACT ANALYSIS
        # ==========================================
        st.markdown("---")
        st.subheader("2️⃣ First Contact Location Data Analysis")
        
        fc_analysis = diagnostic_corners[['minute', 'team', 'corner_side', 'first_contact_won', 'first_contact_lost', 
                                           'no_touch', 'first_contact_player', 'first_contact_x', 'first_contact_y']].copy()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("First Contact Won", int(fc_analysis['first_contact_won'].sum()))
        with col2:
            st.metric("First Contact Lost", int(fc_analysis['first_contact_lost'].sum()))
        with col3:
            st.metric("No Touch", int(fc_analysis['no_touch'].sum()))
        
        # Check location data availability
        fc_lost = fc_analysis[fc_analysis['first_contact_lost'] == 1].copy()
        if not fc_lost.empty:
            fc_lost['has_location'] = fc_lost['first_contact_x'].notna() & fc_lost['first_contact_y'].notna()
            
            with_location = fc_lost['has_location'].sum()
            without_location = len(fc_lost) - with_location
            
            st.write("### 🎯 First Contact Lost - Location Data:")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("✅ WITH location data", int(with_location))
            with col2:
                st.metric("❌ WITHOUT location data", int(without_location))
            
            if without_location > 0:
                st.warning(f"⚠️ {without_location} 'First Contact Lost' events are missing location data. These will use delivery end_x/end_y as fallback.")
            
            st.write("**First Contact Lost Details:**")
            st.dataframe(fc_lost[['minute', 'team', 'first_contact_player', 'first_contact_x', 'first_contact_y', 'has_location']])
        
        # ==========================================
        # SECTION 3: GOALS ANALYSIS
        # ==========================================
        st.markdown("---")
        st.subheader("3️⃣ Goals Analysis")
        
        goals = diagnostic_corners[diagnostic_corners['goal_scored'] == 1].copy()
        if not goals.empty:
            st.write(f"**Found {len(goals)} goals from corners:**")
            
            goal_details = goals[['minute', 'team', 'player', 'corner_side', 'swing_type', 'start_y', 'end_y', 
                                  'goal_scorer', 'shot_body_part', 'total_xg']].copy()
            goal_details['y_diff'] = goal_details['end_y'] - goal_details['start_y']
            
            st.dataframe(goal_details)
            
            # Check for outswing goals specifically
            outswing_goals = goals[goals['swing_type'] == 'Outswing']
            if not outswing_goals.empty:
                st.success(f"✅ Found {len(outswing_goals)} OUTSWING goals!")
                for _, g in outswing_goals.iterrows():
                    st.write(f"- Min {g['minute']}: {g['team']} - {g['corner_side']} corner, y_diff={g['end_y']-g['start_y']:.2f}")
            else:
                st.info("No outswing goals found in current selection")
        else:
            st.info("No goals in current selection")
        
        # ==========================================
        # SECTION 4: SWING TYPE CURRENT DISTRIBUTION
        # ==========================================
        st.markdown("---")
        st.subheader("4️⃣ Current Swing Type Distribution")
        
        if 'swing_type' in diagnostic_corners.columns:
            swing_counts = diagnostic_corners['swing_type'].value_counts()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Current Inswing", int(swing_counts.get('Inswing', 0)))
            with col2:
                st.metric("Current Outswing", int(swing_counts.get('Outswing', 0)))
            
            if swing_counts.get('Outswing', 0) == 0:
                st.error("🚨 PROBLEM: No outswing corners detected! This is the bug we need to fix.")
                st.write("**Checking why all corners are classified as Inswing...**")
                
                # Show actual values
                sample = diagnostic_corners[['minute', 'corner_side', 'start_y', 'end_y', 'swing_type']].head(20).copy()
                sample['y_diff'] = sample['end_y'] - sample['start_y']
                st.dataframe(sample)
        
        # ==========================================
        # SECTION 5: RECOMMENDATIONS
        # ==========================================
        st.markdown("---")
        st.subheader("5️⃣ 💡 Recommendations")
        
        y_diff_stats = diagnostic_corners['end_y'] - diagnostic_corners['start_y']
        recommended_threshold = abs(y_diff_stats.quantile(0.3))
        
        st.write(f"**Recommended threshold for swing detection:** {recommended_threshold:.2f}")
        st.write("**Reasoning:** Based on 30th percentile of absolute y-diff values")
        
        # Count what we'd get with this threshold
        left_corners = diagnostic_corners[diagnostic_corners['corner_side'] == 'Left']
        right_corners = diagnostic_corners[diagnostic_corners['corner_side'] == 'Right']
        
        if not left_corners.empty:
            left_y_diff = left_corners['end_y'] - left_corners['start_y']
            left_inswing = (left_y_diff > recommended_threshold).sum()
            left_outswing = len(left_corners) - left_inswing
            st.write(f"- Left corners: {left_inswing} Inswing, {left_outswing} Outswing")
        
        if not right_corners.empty:
            right_y_diff = right_corners['end_y'] - right_corners['start_y']
            right_inswing = (right_y_diff < -recommended_threshold).sum()
            right_outswing = len(right_corners) - right_inswing
            st.write(f"- Right corners: {right_inswing} Inswing, {right_outswing} Outswing")
        
        st.markdown("---")
        st.success("✅ Diagnostic complete! Use the findings above to adjust thresholds in the main code.")

if __name__ == "__main__":
    main()
