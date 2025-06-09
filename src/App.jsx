import React, { useRef, useEffect, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as cocossd from '@tensorflow-models/coco-ssd';
import './App.css';

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [model, setModel] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadModel = async () => {
      try {
        await tf.setBackend('webgl');
        const loadedModel = await cocossd.load();
        setModel(loadedModel);
        setLoading(false);
        console.log('Model loaded successfully.');
      } catch (error) {
        console.error('Failed to load model:', error);
        setLoading(false);
      }
    };
    loadModel();
  }, []);

  useEffect(() => {
    if (model) {
      startWebcam();
    }
  }, [model]);

  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = stream;
      videoRef.current.addEventListener('loadeddata', detectObjects);
    } catch (error) {
      console.error('Error accessing webcam:', error);
    }
  };

  const detectObjects = async () => {
    if (videoRef.current && canvasRef.current && model) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      setInterval(async () => {
        ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
        const predictions = await model.detect(video);

        predictions.forEach(prediction => {
          ctx.beginPath();
          ctx.rect(...prediction.bbox);
          ctx.lineWidth = 2;
          ctx.strokeStyle = 'red';
          ctx.fillStyle = 'red';
          ctx.stroke();
          ctx.fillText(
            `${prediction.class} (${Math.round(prediction.score * 100)}%)`,
            prediction.bbox[0],
            prediction.bbox[1] > 10 ? prediction.bbox[1] - 5 : 10
          );
        });
      }, 100);
    }
  };

  if (loading) {
    return <div className="App">Loading model...</div>;
  }

  return (
    <div className="App">
      <header className="App-header">
        <h1>Webcam Object Detection</h1>
      </header>
      <div className="video-container">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          width="640"
          height="480"
        />
        <canvas ref={canvasRef} className="detection-canvas" />
      </div>
    </div>
  );
}

export default App;


