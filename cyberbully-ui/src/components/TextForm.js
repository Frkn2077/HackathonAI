import { useState } from 'react';

export default function TextForm({ onAnalyze }) {
    const [text, setText] = useState("");

    const handleSubmit = (e) => {
        e.preventDefault();
        onAnalyze(text);
    };

    return (
        <form onSubmit={handleSubmit}>
            <textarea
                rows={4}
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Metni buraya yaz..."
            />
            <button type="submit">Analiz Et</button>
        </form>
    );
}
