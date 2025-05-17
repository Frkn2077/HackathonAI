import { useState } from 'react';

export default function FileUpload({ onUpload }) {
    const [selectedFile, setSelectedFile] = useState(null);

    const handleChange = (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const fileType = file.name.split('.').pop().toLowerCase();

        if (fileType === 'csv') {
            setSelectedFile(file);
            onUpload(file);
        } else {
            alert('Sadece .csv dosyası yükleyebilirsiniz.');
        }
    };

    const handleRemove = () => {
        setSelectedFile(null);
        alert("Dosya kaldırıldı.");
    };

    return (
        <div>
            <input
                type="file"
                accept=".csv"
                onChange={handleChange}
                disabled={selectedFile !== null}
            />

            {selectedFile && (
                <div style={{ marginTop: "10px" }}>
                    <p><strong>Seçilen dosya:</strong> {selectedFile.name}</p>
                    <button onClick={handleRemove}>Dosyayı Kaldır</button>
                </div>
            )}
        </div>
    );
}
