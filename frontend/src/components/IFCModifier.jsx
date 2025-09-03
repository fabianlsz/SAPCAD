import React, { useState } from 'react';
import axios from 'axios';

const IFCModifier = ({ file, onModificationComplete }) => {
    const [windowId, setWindowId] = useState('');
    const [newWidth, setNewWidth] = useState('');
    const [newHeight, setNewHeight] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleModify = async () => {
        if (!file || !windowId || !newWidth || !newHeight) {
            setError('Please fill in all fields');
            return;
        }

        setLoading(true);
        setError(null);

        const formData = new FormData();
        formData.append('file', file);
        formData.append('modification', JSON.stringify({
            window_id: parseInt(windowId),
            new_width: parseFloat(newWidth),
            new_height: parseFloat(newHeight)
        }));

        try {
            const response = await axios.post('http://localhost:8000/api/modify-window', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
                responseType: 'blob'
            });

            // Create a new File object from the response
            const modifiedFile = new File([response.data], file.name, { type: file.type });

            // Call the callback with the modified file
            onModificationComplete(modifiedFile);
        } catch (err) {
            setError(err.response?.data?.message || 'An error occurred while modifying the file');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="p-4 bg-white rounded-lg shadow">
            <h2 className="text-xl font-bold mb-4">Modify Window</h2>

            <div className="space-y-4">
                <div>
                    <label className="block text-sm font-medium text-gray-700">Window ID</label>
                    <input
                        type="number"
                        value={windowId}
                        onChange={(e) => setWindowId(e.target.value)}
                        className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                        placeholder="Enter window ID"
                    />
                </div>

                <div>
                    <label className="block text-sm font-medium text-gray-700">New Width (meters)</label>
                    <input
                        type="number"
                        value={newWidth}
                        onChange={(e) => setNewWidth(e.target.value)}
                        className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                        placeholder="Enter new width"
                        step="0.01"
                    />
                </div>

                <div>
                    <label className="block text-sm font-medium text-gray-700">New Height (meters)</label>
                    <input
                        type="number"
                        value={newHeight}
                        onChange={(e) => setNewHeight(e.target.value)}
                        className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                        placeholder="Enter new height"
                        step="0.01"
                    />
                </div>

                {error && (
                    <div className="p-4 bg-red-100 text-red-700 rounded">
                        {error}
                    </div>
                )}

                <button
                    onClick={handleModify}
                    disabled={loading || !file || !windowId || !newWidth || !newHeight}
                    className={`w-full px-4 py-2 rounded ${
                        loading || !file || !windowId || !newWidth || !newHeight
                            ? 'bg-gray-300 cursor-not-allowed'
                            : 'bg-blue-500 hover:bg-blue-600 text-white'
                    }`}
                >
                    {loading ? 'Modifying...' : 'Modify Window'}
                </button>
            </div>
        </div>
    );
};

export default IFCModifier;