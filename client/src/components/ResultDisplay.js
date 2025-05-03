// ResultDisplay.js
import React, { useState, useEffect } from 'react';
import './ResultDisplay.css';

const ResultDisplay = ({ result }) => {
  const [cover, setCover] = useState(
    process.env.PUBLIC_URL + '/no_image.png'
  );

  useEffect(() => {
    // only attempt JSONP if we don’t already have a coverUrl
    if (!result.coverUrl && result.song) {
      const term = encodeURIComponent(`${result.song} ${result.artist || ''}`);
      const callbackName = `deezer_cb_${Date.now()}`;

      // install the callback
      window[callbackName] = data => {
        if (data && data.data && data.data.length > 0) {
          // grab the big cover if available
          setCover(data.data[0].album.cover_big);
        }
        // cleanup
        document.body.removeChild(script);
        delete window[callbackName];
      };

      // insert JSONP script tag
      const script = document.createElement('script');
      script.src = `https://api.deezer.com/search?q=${term}&limit=1&output=jsonp&callback=${callbackName}`;
      document.body.appendChild(script);
    }

    // if backend ever sends a direct URL, prefer it
    if (result.coverUrl) {
      setCover(result.coverUrl);
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
          // fallback to placeholder if that also 404s
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
