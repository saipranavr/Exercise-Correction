window.onload = () => {
    console.log('script.js loaded and window is ready.');

    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const canvasCtx = canvas.getContext('2d');
    const feedbackText = document.getElementById('feedback-text');

    function onResults(results) {
        canvasCtx.save();
        canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
        canvasCtx.drawImage(results.image, 0, 0, canvas.width, canvas.height);

        if (results.poseLandmarks) {
            window.drawConnectors(canvasCtx, results.poseLandmarks, window.POSE_CONNECTIONS, { color: '#00FF00', lineWidth: 4 });
            window.drawLandmarks(canvasCtx, results.poseLandmarks, { color: '#FF0000', lineWidth: 2 });
            analyzeBicepCurl(results.poseLandmarks);
        }
        canvasCtx.restore();
    }

async function analyzeBicepCurl(landmarks) {
    try {
        const response = await fetch('http://127.0.0.1:5001/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ landmarks: landmarks })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        feedbackText.textContent = data.feedback;

    } catch (error) {
        console.error("Could not fetch feedback:", error);
        feedbackText.textContent = "Error getting feedback from server.";
    }
}

    const pose = new window.Pose({
        locateFile: (file) => {
            return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`;
        }
    });

    pose.setOptions({
        modelComplexity: 1,
        smoothLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
    });

    pose.onResults(onResults);

    async function setupCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: true,
            });
            video.srcObject = stream;

            return new Promise((resolve) => {
                video.onloadedmetadata = () => {
                    resolve(video);
                };
            });
        } catch (err) {
            feedbackText.textContent = 'Error accessing webcam. Please grant permission and refresh.';
            console.error('Error accessing webcam:', err);
        }
    }

    async function main() {
        feedbackText.textContent = 'Setting up camera...';
        await setupCamera();
        if (!video.srcObject) return; // Stop if camera setup failed

        const camera = new window.Camera(video, {
            onFrame: async () => {
                await pose.send({ image: video });
            },
            width: 640,
            height: 480
        });
        camera.start();
        feedbackText.textContent = 'Pose detection is active. Start your bicep curls!';
    }

    main();
};
