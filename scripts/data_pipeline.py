from requests import get, post
from sys import exit
from pprint import pprint
import os
from dotenv import load_dotenv
import time

load_dotenv()

# Read environment variables
HOST = os.getenv("CHARTMETRIC_API_URL")
REFRESH_TOKEN = os.getenv("CHARTMETRIC_REFRESH_TOKEN")

# Get access token
res = post(f'{HOST}/api/token', json={"refreshtoken": REFRESH_TOKEN})
if res.status_code != 200:
    print(f'ERROR: received a {res.status_code} instead of 200 from /api/token')
    exit(1)

access_token = res.json()['token']

# Define GET function using access token
# def Get(uri):
#     return get(f'{HOST}{uri}', headers={'Authorization': f'Bearer {access_token}'})

def Get(uri, **kwargs):
    # sleep for 1 second to respect rate limit
    time.sleep(1)
    """Wrapper that allows optional query parameters."""
    return get(f'{HOST}{uri}',
               headers={'Authorization': f'Bearer {access_token}'},
               params=kwargs)


# Get artist brief by name
def searchArtist(name):
    return get(f'{HOST}/api/search?q={name}&type=artists', headers={'Authorization': f'Bearer {access_token}'})

# Get artist ID by name
def getArtistID(name):
    data = searchArtist(name).json()
    artists = data.get('obj', {}).get('artists', [])
    if not artists:
        print(f"No artist found for '{name}'")
        return None
    return artists[0]['id']
# def getArtistId(name):
# Example query: get data
# res = Get('/api/artist/3380')
# res = getArtist('Lil Wayne')
# if res.status_code != 200:
#     print(f'ERROR: received a {res.status_code} instead of 200 from /api/artist/:id')
#     exit(1)

# pprint(res.json())
def getArtistFromID(id):
    return get(f'{HOST}/api/artist/{id}', headers={'Authorization': f'Bearer {access_token}'})
#/api/artist/:id/social-audience-stats

def getArtistSocialStats(id, domain="instagram", audience_type="followers", stats_type="stat"):
    """
    Get artist social audience stats for a specific domain.
    
    Available domains: instagram, youtube, tiktok
    Available audience types: followers, reach, engagement, etc.
    Available stats types: country, city, interest, brand, language, stat, demographic
    """
    return Get(f"/api/artist/{id}/social-audience-stats?domain={domain}&audienceType={audience_type}&statsType={stats_type}")

def getSpotifyStats(artist_id):
    """
    Get Spotify stats like followers and monthly listeners (time series).
    Endpoint: /api/artist/{id}/stat/spotify
    """
    uri = f"/api/artist/{artist_id}/stat/spotify"
    res = Get(uri)
    if res.status_code == 200:
        return res.json()
    print(f"{res.status_code} on {uri} → {res.text}")
    return None

def print_time_series(ts, label, value_key="value"):
    if not ts:
        print(f"{label}: No data\n")
        return

    print(f"\n--- {label} (time series) ---")
    for point in ts:
        date = point["timestp"][:10]
        value = point[value_key]
        print(f"{date}: {value:,}" if isinstance(value, int) else f"{date}: {value}")

artist_name = "SZA"
artist_id = getArtistID(artist_name)

if artist_id:
    print(f"\nSpotify stats for {artist_name} (ID: {artist_id})")

    stats = getSpotifyStats(artist_id)
    pprint(stats)

# === SUMMARY PRINT ===
spotify_obj = stats.get("obj", {})

followers_ts = spotify_obj.get("followers", [])
listeners_ts = spotify_obj.get("listeners", [])
ratio_ts     = spotify_obj.get("followers_to_listeners_ratio", [])
popularity   = spotify_obj.get("popularity", [])
