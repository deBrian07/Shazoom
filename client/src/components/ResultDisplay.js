import React, { useState, useEffect, useRef } from 'react';
import fetchJsonp from 'fetch-jsonp';
import './ResultDisplay.css';

const ResultDisplay = ({ result }) => {
  // State for album cover
  const [cover, setCover] = useState(
    process.env.PUBLIC_URL + '/placeholder_album.png'
  );

  // State for 30-second preview URL
  const [previewUrl, setPreviewUrl] = useState(result.previewUrl || null);
  const [isPlaying, setIsPlaying] = useState(false);
  const audioRef = useRef(null);

  // Fetch album cover via Deezer JSONP (unless provided)
  useEffect(() => {
    if (result.coverUrl) {
      setCover(result.coverUrl);
      return;
    }
    if (!result.song) return;
    const term = encodeURIComponent(
      `${result.song}${result.artist ? ' ' + result.artist : ''}`
    );
    const deezerUrl = `https://api.deezer.com/search?q=${term}&limit=1&output=jsonp`;
    fetchJsonp(deezerUrl, { jsonpCallback: 'callback' })
      .then(response => response.json())
      .then(data => {
        if (data.data?.length) {
          setCover(data.data[0].album.cover_big);
        }
      })
      .catch(() => {
        // fallback on error
      });
  }, [result.song, result.artist, result.coverUrl]);

  // Fetch preview URL via iTunes if backend didn't supply
  useEffect(() => {
    if (previewUrl || !result.song) return;
    const term = encodeURIComponent(
      `${result.song}${result.artist ? ' ' + result.artist : ''}`
    );
    fetch(`https://itunes.apple.com/search?term=${term}&limit=1&entity=song`)
      .then(res => res.json())
      .then(json => {
        if (json.results?.length) {
          setPreviewUrl(json.results[0].previewUrl);
        }
      })
      .catch(() => {
        // ignore errors
      });
  }, [previewUrl, result.song, result.artist]);

  const togglePlay = e => {
    e.stopPropagation();
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play();
    }
    setIsPlaying(!isPlaying);
  };

  if (!result.song && !result.error) {
    return (
      <div className="result-container empty">
        <p>No match found</p>
      </div>
    );
  }

  const coverSrc = cover;
  const spotifyUrl =
    result.spotifyUrl ||
    `https://open.spotify.com/search/${encodeURIComponent(
      result.song + (result.artist ? ' ' + result.artist : '')
    )}`;
  const openLink = url => window.open(url, '_blank', 'noopener');

  return (
    <div className="result-container" onClick={() => openLink(spotifyUrl)} style={{ display: 'flex', alignItems: 'center' }}>
      <img
        className="album-cover"
        src={coverSrc}
        alt={`${result.song} cover`}
        onError={e => {
          e.currentTarget.onerror = null;
          e.currentTarget.src = process.env.PUBLIC_URL + '/placeholder_album.png';
        }}
      />
      <div className="song-info" style={{ flexGrow: 1 }}>
        <h3 className="song-title">{result.song}</h3>
        {result.artist && <p className="song-artist">{result.artist}</p>}
      </div>
      {previewUrl && (
        <button
          onClick={togglePlay}
          style={{
            background: 'transparent',
            border: 'none',
            cursor: 'pointer',
            padding: '0 8px',
            color: 'var(--text-primary)',
            fontSize: '1.25rem'
          }}
          title={isPlaying ? 'Pause preview' : 'Play preview'}
        >
          {isPlaying ? '❚❚' : '▶'}
        </button>
      )}
      {previewUrl && (
        <audio
          ref={audioRef}
          src={previewUrl}
          onEnded={() => setIsPlaying(false)}
        />
      )}
    </div>
  );
};

export default ResultDisplay;