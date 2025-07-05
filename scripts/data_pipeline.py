from requests import get, post
from sys import exit
from pprint import pprint
import os
from dotenv import load_dotenv
import time
from collections import defaultdict
from datetime import datetime

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

def Get(uri, **kwargs):
    # sleep for 1 second to respect rate limit
    time.sleep(1)
    """Wrapper that allows optional query parameters."""
    return get(f'{HOST}{uri}',
               headers={'Authorization': f'Bearer {access_token}'},
               params=kwargs)


# Get artist brief by name
def searchArtist(name):
    res = Get("/api/search", q=name, type="artists")
    if res.status_code != 200:
        print(f"{res.status_code} from /api/search → {res.text}")
        return None
    return res

# Get artist ID by name
def getArtistID(name):
    response = searchArtist(name)
    if not response:
        return None
    try:
        data = response.json()
    except Exception as e:
        print(f"Failed to parse JSON for artist search: {e}")
        return None

    artists = data.get('obj', {}).get('artists', [])
    if not artists:
        print(f"No artist found for '{name}'")
        return None
    return artists[0]['id']

def getArtistFromID(id):
    return get(f'{HOST}/api/artist/{id}', headers={'Authorization': f'Bearer {access_token}'})

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

def findSpotifyStreamsByTrackName(artist_id, track_name, artist_type="main", limit=100):
    """
    Search the artist's track catalog for a title and return its Spotify stream count.
    - artist_type: "main", "featured", or omit for both
    """
    uri = f"/api/artist/{artist_id}/tracks"
    res = Get(uri, limit=limit, artist_type=artist_type)
    if res.status_code != 200:
        print(f"{res.status_code} on {uri} → {res.text}")
        return None

    tracks = res.json().get("obj", [])
    # simple case‐insensitive exact match; adapt to fuzzy if you like
    for t in tracks:
        if t.get("name", "").lower() == track_name.lower():
            return t.get("cm_statistics", {}).get("sp_streams")
    print(f"Track '{track_name}' not found in first {limit} results.")
    return None

def test_findSpotifyStreamsByTrackName():
    artist_name = "SZA"
    track_name = "Good Days"  # change to any known track

    artist_id = getArtistID(artist_name)
    if not artist_id:
        print(f"Could not find artist ID for '{artist_name}'")
        return

    print(f"\n🎵 Fetching Spotify stream count for '{track_name}' by {artist_name}")
    streams = findSpotifyStreamsByTrackName(artist_id, track_name)

    if streams is not None:
        print(f"'{track_name}' has {streams:,} Spotify streams")
    else:
        print(f"Could not find stream data for '{track_name}'")

# Run test
test_findSpotifyStreamsByTrackName()

