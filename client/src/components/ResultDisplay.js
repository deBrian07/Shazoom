import React, { useState, useEffect } from 'react';
import './ResultDisplay.css';

// optional: import a local placeholder if you prefer
// import placeholder from '../assets/placeholder_album.png';

const ResultDisplay = ({ result }) => {
  const [cover, setCover] = useState(
    result.coverUrl || process.env.PUBLIC_URL + '/placeholder_album.png'
  );

  useEffect(() => {
    // only fetch if backend didn’t give us art
    if (!result.coverUrl) {
      const term = encodeURIComponent(`${result.song} ${result.artist || ''}`);
      fetch(`https://itunes.apple.com/search?term=${term}&limit=1&entity=song`)
        .then(res => res.json())
        .then(json => {
          if (json.results?.length) {
            // artworkUrl100 is 100×100px; swap for a larger size
            const artUrl = json.results[0].artworkUrl100.replace('100x100', '300x300');
            setCover(artUrl);
          }
        })
        .catch(() => {
          // swallow errors, keep placeholder
        });
    }
  }, [result.song, result.artist, result.coverUrl]);

  const spotifyUrl =
    result.spotifyUrl ||
    `https://open.spotify.com/search/${encodeURIComponent(
      result.song + (result.artist ? ' ' + result.artist : '')
    )}`;

  const openLink = url => window.open(url, '_blank', 'noopener');

  return (
    <div className="result-container" onClick={() => openLink(spotifyUrl)}>
      <img
        className="album-cover"
        src={cover}
        alt={`${result.song} album cover`}
        onError={e => {
          // if fetch URL 404s, fall back to placeholder
          e.currentTarget.onerror = null;
          e.currentTarget.src = process.env.PUBLIC_URL + '/no_image.png';
        }}
      />
      <div className="song-info">
        <h3 className="song-title">{result.song}</h3>
        {result.artist && <p className="song-artist">{result.artist}</p>}
      </div>
    </div>
  );
};

export default ResultDisplay;