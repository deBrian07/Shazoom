import React from 'react';

const ResultDisplay = ({ result }) => (
  <div className="result">
    {result.song ? (
      <p>Recognized Song: {result.song} by {result.artist}</p>
    ) : result.error ? (
      <p>Error: {result.error}</p>
    ) : (
      <p>No match found</p>
    )}
  </div>
);

export default ResultDisplay;