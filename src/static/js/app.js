// API configuration
const API_BASE_URL = 'http://localhost:8000';
let API_KEY = localStorage.getItem('apiKey');

// Function to prompt for API key
function promptForApiKey() {
    const key = prompt('Please enter your API key:');
    if (key) {
        localStorage.setItem('apiKey', key);
        API_KEY = key;
        return key;
    }
    return null;
}

// Check for API key on page load
if (!API_KEY) {
    promptForApiKey();
}

// Helper function to get headers with API key
function getHeaders() {
    if (!API_KEY) {
        API_KEY = promptForApiKey();
        if (!API_KEY) {
            throw new Error('API key is required');
        }
    }
    return {
        'X-API-Key': API_KEY
    };
}

// List Registered Users
document.getElementById('listUsersBtn').addEventListener('click', async () => {
    const usersListDiv = document.getElementById('usersList');
    
    // Toggle visibility
    if (usersListDiv.style.display === 'block') {
        usersListDiv.style.display = 'none';
        usersListDiv.innerHTML = ''; // Clear content when hidden
        return;
    }
    
    // If hidden, show loading and fetch data
    showLoading(usersListDiv);
    usersListDiv.style.display = 'block'; // Show the loading indicator
    
    try {
        const response = await fetch(`${API_BASE_URL}/debug/registered-users`, {
            method: 'GET',
            headers: getHeaders()
        });
        
        if (response.status === 401 || response.status === 403) {
            localStorage.removeItem('apiKey');
            API_KEY = null;
            alert('Authentication failed. Please enter your API key again.');
            showResult(usersListDiv, 'Authentication failed', true); // Show error in the div
            return;
        }
        
        const data = await response.json();
        if (response.ok) {
            if (data.registered_users && data.registered_users.length > 0) {
                const usersList = data.registered_users.map(user => `<li>${user}</li>`).join('');
                usersListDiv.innerHTML = `
                    <h3>Registered Users (${data.total_users})</h3>
                    <ul class="users-list">
                        ${usersList}
                    </ul>
                `;
            } else {
                showResult(usersListDiv, 'No users registered in the database', true);
            }
        } else {
            const errorMessage = await handleApiError(data, response);
            showResult(usersListDiv, `Error: ${errorMessage}`, true);
        }
    } catch (error) {
        showResult(usersListDiv, `Error: ${error.message}`, true);
    }
});

// Camera configuration
let currentStream = null;
let currentInputId = null;
const cameraModal = document.getElementById('cameraModal');
const cameraFeed = document.getElementById('cameraFeed');
const cameraCanvas = document.getElementById('cameraCanvas');
const captureBtn = document.getElementById('captureBtn');
const closeCameraBtn = document.getElementById('closeCameraBtn');

// Helper functions
function showLoading(element) {
    element.innerHTML = '<div class="loading"></div>';
}

function showResult(element, message, isError = false) {
    // Display the message as is, let the calling code format it if needed
    element.innerHTML = `<div class="${isError ? 'result-error' : 'result-success'}">${message}</div>`;
}

function updatePreview(input, previewId) {
    const preview = document.getElementById(previewId);
    const file = input.files[0];
    
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            preview.src = e.target.result;
            preview.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }
}

async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                facingMode: 'user',
                width: { ideal: 1280 },
                height: { ideal: 720 }
            } 
        });
        cameraFeed.srcObject = stream;
        currentStream = stream;
        cameraModal.classList.remove('hidden');
    } catch (error) {
        console.error('Error accessing camera:', error);
        alert('Error accessing camera. Please make sure you have granted camera permissions.');
    }
}

function stopCamera() {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
        currentStream = null;
    }
    cameraModal.classList.add('hidden');
}

function captureImage() {
    const context = cameraCanvas.getContext('2d');
    cameraCanvas.width = cameraFeed.videoWidth;
    cameraCanvas.height = cameraFeed.videoHeight;
    context.drawImage(cameraFeed, 0, 0);
    
    // Convert canvas to blob
    cameraCanvas.toBlob((blob) => {
        const file = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' });
        const input = document.getElementById(currentInputId);
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        input.files = dataTransfer.files;
        
        // Update preview
        const previewId = currentInputId.replace('Image', 'Preview');
        updatePreview(input, previewId);
        
        // Close camera
        stopCamera();
    }, 'image/jpeg', 0.95);
}

// Camera event listeners
document.getElementById('registerCameraBtn').addEventListener('click', () => {
    currentInputId = 'registerImage';
    startCamera();
});

document.getElementById('verifyCameraBtn').addEventListener('click', () => {
    currentInputId = 'verifyImage';
    startCamera();
});

document.getElementById('identifyCameraBtn').addEventListener('click', () => {
    currentInputId = 'identifyImage';
    startCamera();
});

captureBtn.addEventListener('click', captureImage);
closeCameraBtn.addEventListener('click', stopCamera);

// Event listeners for file inputs
document.getElementById('registerImage').addEventListener('change', (e) => {
    updatePreview(e.target, 'registerPreview');
});

document.getElementById('verifyImage').addEventListener('change', (e) => {
    updatePreview(e.target, 'verifyPreview');
});

document.getElementById('identifyImage').addEventListener('change', (e) => {
    updatePreview(e.target, 'identifyPreview');
});

// Helper function to handle API errors
async function handleApiError(data, response) {
    if (response.status === 401 || response.status === 403) {
        localStorage.removeItem('apiKey');
        API_KEY = null;
        alert('Authentication failed. Please enter your API key again.');
        return 'Authentication failed';
    }
    // Use the already parsed data
    return data.detail || 'An error occurred';
}

// Form submissions
document.getElementById('registerForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const resultDiv = document.getElementById('registerResult');
    showLoading(resultDiv);

    const userId = document.getElementById('registerUserId').value;
    const file = document.getElementById('registerImage').files[0];
    
    if (!file) {
        alert('Please select an image file');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch(`${API_BASE_URL}/register?user_id=${encodeURIComponent(userId)}`, {
            method: 'POST',
            headers: getHeaders(),
            body: formData
        });
        
        const data = await response.json(); // Read the response body once
        
        if (response.ok) {
            showResult(resultDiv, `Successfully registered face for user: ${data.user_id}`);
        } else {
            // Pass the already parsed data to handleApiError
            const errorMessage = await handleApiError(data, response);
            // Display the error message directly without adding 'Error: '
            showResult(resultDiv, errorMessage, true);
        }
    } catch (error) {
        // Handled by showResult now
        showResult(resultDiv, error.message, true);
    }
});

document.getElementById('verifyForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const resultDiv = document.getElementById('verifyResult');
    showLoading(resultDiv);

    const userId = document.getElementById('verifyUserId').value;
    const file = document.getElementById('verifyImage').files[0];
    
    if (!file) {
        alert('Please select an image file');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch(`${API_BASE_URL}/verify?user_id=${encodeURIComponent(userId)}`, {
            method: 'POST',
            headers: getHeaders(),
            body: formData
        });
        
        const data = await response.json(); // Read the response body once
        
        if (response.ok) {
            const message = data.verified 
                ? `Face verified successfully! Similarity: ${(data.similarity * 100).toFixed(2)}%`
                : `Face verification failed. Similarity: ${(data.similarity * 100).toFixed(2)}%`;
            showResult(resultDiv, message, !data.verified);
        } else {
            // Pass the already parsed data to handleApiError
            const errorMessage = await handleApiError(data, response);
            // Handled by showResult now
            showResult(resultDiv, errorMessage, true);
        }
    } catch (error) {
        // Handled by showResult now
        showResult(resultDiv, error.message, true);
    }
});

document.getElementById('identifyForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const resultDiv = document.getElementById('identifyResult');
    showLoading(resultDiv);

    const file = document.getElementById('identifyImage').files[0];
    
    if (!file) {
        alert('Please select an image file');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch(`${API_BASE_URL}/identify`, {
            method: 'POST',
            headers: getHeaders(),
            body: formData
        });
        
        const data = await response.json(); // Read the response body once
        
        if (response.ok) {
            if (data.matches && data.matches.length > 0) {
                showResult(resultDiv, `Face identified as: ${data.matches[0].user_id} (Similarity: ${(data.matches[0].similarity * 100).toFixed(2)}%)`);
            } else {
                showResult(resultDiv, 'No matching face found in the database', true);
            }
        } else {
            // Pass the already parsed data to handleApiError
            const errorMessage = await handleApiError(data, response);
            // Handled by showResult now
            showResult(resultDiv, errorMessage, true);
        }
    } catch (error) {
        // Handled by showResult now
        showResult(resultDiv, error.message, true);
    }
});

// Reset Register Form
document.getElementById('resetRegisterBtn').addEventListener('click', () => {
    document.getElementById('registerUserId').value = '';
    document.getElementById('registerImage').value = ''; // Clear file input
    const preview = document.getElementById('registerPreview');
    preview.src = '';
    preview.style.display = 'none';
    document.getElementById('registerResult').innerHTML = ''; // Clear result message
});

// Reset Verify Form
document.getElementById('resetVerifyBtn').addEventListener('click', () => {
    document.getElementById('verifyUserId').value = '';
    document.getElementById('verifyImage').value = ''; // Clear file input
    const preview = document.getElementById('verifyPreview');
    preview.src = '';
    preview.style.display = 'none';
    document.getElementById('verifyResult').innerHTML = ''; // Clear result message
});

// Reset Identify Form
document.getElementById('resetIdentifyBtn').addEventListener('click', () => {
    document.getElementById('identifyImage').value = ''; // Clear file input
    const preview = document.getElementById('identifyPreview');
    preview.src = '';
    preview.style.display = 'none';
    document.getElementById('identifyResult').innerHTML = ''; // Clear result message
}); 