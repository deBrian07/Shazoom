import React from 'react';

const RecorderButton = ({ handleRecordClick, isRecording, isLoading, buttonWrapperRef }) => (
  <div ref={buttonWrapperRef} className="button-wrapper">
    <button
      onClick={handleRecordClick}
      className={`record-button ${isRecording ? 'recording' : ''} ${isLoading ? 'loading' : ''}`}
      disabled={isRecording}
    />
  </div>
);

export default RecorderButton;