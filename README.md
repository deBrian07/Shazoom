[demo](https://debrian07.github.io/Shazoom/)
# Shazoom
Shazoom is an implementation of Shazam's song recognition algorithm. It uses YouTube to find and download songs. It is based on [source](https://github.com/cgzirim/seek-tune).
<div align="center">
<img src="https://github.com/user-attachments/assets/cb93fadf-cd39-4320-80f4-dcc7f7bbeaa1" alt="Alt Text" style="width:35%; height:auto;">
</div>

## Steps
### Clone the repo
```
git clone https://github.com/deBrian07/Shazoom.git
cd Shazoom
```
### Setup 
```
cd client
npm install
```
## Usage
### Running Backend
```
cd server
source venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 5000
```
### Running Frontend 
```
cd client
npm start
```
### Build/deploy Frontend
```
npm run build
npm run deploy
```
### Build/ship Frontend
```
npm run electron:serve
npm run electron:prod

# build native installers
npm run dist
```

### Download songs
Make sure that all the songs are in `songs.csv` in the order of `song name, artist`
```
cd download
python3 downloader.py
```
Perform post process to generate the `processed.csv` that contains the song name, artist, and the path to each song.
```
python3 processed.csv
```
