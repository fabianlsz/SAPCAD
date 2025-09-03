import React, { useState } from "react";
import NeonButton from "./general/NeonButton";
import { useGlobal } from "../states/GlobalState";

const ChatMessage = ({ sender, text }) => (
    <div className={`my-2 text-sm ${sender === "user" ? "text-right" : "text-left"}`}>
        <span className="font-semibold">{sender === "user" ? "You" : "AI"}:</span>{" "}
        {text}
    </div>
);

const ChatOllamaConnector = () => {
    const { setModifiedFile, setIsModifiedByAI } = useGlobal();
    const { modifiedFile, isModifiedByAI } = useGlobal(); // Access GlobalContext
    const { originalFile, setOriginalFile} = useGlobal();
    const { pdfFile, setPdfFile } = useGlobal();
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [message, setMessage] = useState("");
    const [error, setError] = useState(null);
    const [history, setHistory] = useState([]);

    const handleChat = async () => {
        if (!message.trim() || loading) return;

        setLoading(true);
        setError(null);
        setResult(null);
        setHistory((prev) => [...prev, { sender: "user", text: message }]);
        setMessage("");

        try {
            if (originalFile && pdfFile && message) {
                const formData = new FormData();
                formData.append("ifc_file", originalFile);
                formData.append("norm_pdf_file", pdfFile);
                formData.append("user_instruction", message);

                const response = await fetch("http://localhost:8000/modify-ifc", {
                    method: "POST",
                    body: formData,
                });

                if (!response.ok) throw new Error("Failed to modify IFC");

                const data = await response.json();
                console.log(JSON.stringify(data))
                const blob = await fetch(`http://localhost:8000${data.download_url}`).then(r => r.blob());

                const modifiedFile = new File([blob], originalFile.name.replace(".ifc", "_modified.ifc"), {
                    type: "application/octet-stream",
                });

                setModifiedFile(modifiedFile);
                setIsModifiedByAI(true);
                setResult({
                    download_url: data.download_url,
                    change_detail: data.change,
                    llm_response: data.llm_response,
                });

                setHistory((prev) => [...prev, {
                    sender: "bot",
                    text: `✅ Model updated.\n📝 Change: ${data.change}`,
                }]);

                setLoading(false);
                return;
            }

            const response = await fetch("http://localhost:8000/llm", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ prompt: message }),
            });

            const data = await response.json();
            const botText = data.llm || data.response || "No response.";

            setHistory((prev) => [...prev, { sender: "bot", text: botText }]);

        } catch (err) {
            console.error(err);
            setError("Something went wrong.");
        }

        setLoading(false);
    };
    return (
        <div className="relative px-6 py-3 rounded-lg h-full flex flex-col bg-zinc-950 text-white">
            {/* Chat History */}
            <div className="flex-1 overflow-y-auto mb-4 hide-scrollbar">
                {history.map((msg, idx) => (
                    <ChatMessage key={idx} sender={msg.sender} text={msg.text} />
                ))}
            </div>

            {/* Input & NeonButton */}
            <div className="flex gap-2">
                <input
                    type="text"
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    placeholder="Type your command..."
                    disabled={loading}
                    onKeyDown={(e) => e.key === "Enter" && handleChat()}
                    className="flex-1 h-12 px-4 rounded-lg bg-zinc-800 border border-zinc-700 focus:outline-none focus:ring focus:ring-cyan-500 text-sm"
                />
                <NeonButton
                    type="button"
                    onClick={handleChat}
                    color="cyan"
                    className={`h-12 flex items-center ${loading || !message.trim() ? "opacity-50 pointer-events-none" : ""}`}
                >
                    {loading ? "..." : "Send"}
                </NeonButton>
            </div>

            {/* Error Message */}
            {error && <div className="text-red-400 mt-2">{error}</div>}
        </div>
    );
};

export default ChatOllamaConnector;