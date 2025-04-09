import requests
import csv
import time

# Replace with your own Last.fm API key from https://www.last.fm/api
API_KEY = "15ad6441f239427d972579d3d492be3c"
BASE_URL = "http://ws.audioscrobbler.com/2.0/"

def get_top_tracks(page=1, limit=100):
    """
    Fetches one page of top tracks from Last.fm's chart.gettoptracks API.
    """
    params = {
        "method": "chart.gettoptracks",
        "api_key": API_KEY,
        "format": "json",
        "page": page,
        "limit": limit
    }
    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()
    return response.json()

def collect_top_tracks(target=10000, limit=100):
    """
    Collects top tracks until the target number of songs is reached or all pages are exhausted.
    Returns a list of tuples: (song_name, artist_name).
    """
    tracks = []
    current_page = 1

    while len(tracks) < target:
        print(f"Fetching page {current_page}...")
        try:
            data = get_top_tracks(page=current_page, limit=limit)
        except Exception as e:
            print(f"Error retrieving page {current_page}: {e}")
            break

        # Check for expected structure in the JSON data.
        if "tracks" not in data or "track" not in data["tracks"]:
            print("Unexpected data format received from Last.fm API.")
            break

        page_tracks = data["tracks"]["track"]
        if not page_tracks:
            break

        for track in page_tracks:
            song_name = track.get("name", "").strip()
            artist_name = track.get("artist", {}).get("name", "").strip()
            if song_name and artist_name:
                tracks.append((song_name, artist_name))

        # Determine total pages available
        total_pages = int(data["tracks"]["@attr"].get("totalPages", current_page))
        if current_page >= total_pages:
            print("Reached last available page.")
            break

        current_page += 1
        # Pause briefly to remain under API rate limits.
        time.sleep(0.5)

    print(f"Collected {len(tracks)} tracks in total.")
    return tracks[:target]

def write_tracks_to_csv(tracks, output_csv="top_tracks.csv"):
    """
    Writes the list of tracks to a CSV file with header: song name, artist.
    """
    with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["song name", "artist"])
        for song_name, artist in tracks:
            writer.writerow([song_name, artist])
    print(f"CSV file '{output_csv}' written with {len(tracks)} entries.")

def main():
    target_tracks = 10000  # Set the target number of tracks
    tracks = collect_top_tracks(target=target_tracks, limit=100)

    if len(tracks) < target_tracks:
        print(f"Warning: Only {len(tracks)} tracks were collected, which is less than {target_tracks}.")
    write_tracks_to_csv(tracks)

if __name__ == '__main__':
    main()
