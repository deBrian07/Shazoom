import React, { useState, useEffect } from 'react';
import fetchJsonp from 'fetch-jsonp';
import './ResultDisplay.css';

const ResultDisplay = ({ result }) => {
  const [cover, setCover] = useState(
    process.env.PUBLIC_URL + '/placeholder_album.png'
  );

  useEffect(() => {
    // Trust backend-provided cover if available
    if (result.coverUrl) {
      setCover(result.coverUrl);
      return;
    }

    // Require a song title
    if (!result.song) return;

    // Use Deezer JSONP via fetch-jsonp (no manual script tags)
    const term = encodeURIComponent(
      `${result.song}${result.artist ? ' ' + result.artist : ''}`
    );
    const deezerUrl =
      `https://api.deezer.com/search?q=${term}` +
      `&limit=1&output=jsonp`;

    fetchJsonp(deezerUrl, { jsonpCallback: 'callback' })
      .then(response => response.json())
      .then(data => {
        if (data.data?.length) {
          setCover(data.data[0].album.cover_big);
        }
      })
      .catch(() => {
        // fallback to placeholder on error
      });
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
        alt={`${result.song} cover`}
        onError={e => {
          e.currentTarget.onerror = null;
          e.currentTarget.src = process.env.PUBLIC_URL + '/placeholder_album.png';
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