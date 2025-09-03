import React, { useState, useRef } from "react";
import IfcViewer from "./IfcViewer";
import NeonButton from "../general/NeonButton";
import toast from "react-hot-toast";
import { useGlobal } from "../../states/GlobalState.jsx";

const Result = () => {
    const { modifiedFile, isModifiedByAI } = useGlobal(); // Access GlobalContext
    const { originalFile, setOriginalFile } = useGlobal();
    const { pdfFile, setPdfFile } = useGlobal();
    const [isLoading, setIsLoading] = useState(false);

    const ifcInputRef = useRef(null);
    const pdfInputRef = useRef(null);

    const handleIfcClick = () => ifcInputRef.current?.click();
    const handlePdfClick = () => pdfInputRef.current?.click();

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        if (!file) return;

        const maxSizeBytes = 200 * 1024 * 1024;

        if (!file.name.toLowerCase().endsWith(".ifc")) {
            toast.error("Please select a valid IFC file.");
            return;
        }

        if (file.size > maxSizeBytes) {
            toast.error("The selected IFC file exceeds the 200 MB limit.");
            return;
        }

        setOriginalFile(file);
        setIsLoading(false);
    };

    const handleFileUpload = (event) => {
        const file = event.target.files[0];
        if (file?.name.toLowerCase().endsWith(".pdf")) {
            setPdfFile(file);
        } else {
            toast.error("Please select a valid PDF file.");
        }
    };

    return (
        <div className="w-full h-full relative overflow-hidden">
            <div className="p-6 flex justify-between items-start">
                {/* Left: Upload buttons */}
                <div className="flex space-x-8">
                    {/* IFC Upload */}
                    <div className="flex flex-col items-center">
                        <input
                            type="file"
                            accept=".ifc"
                            onChange={handleFileChange}
                            ref={ifcInputRef}
                            className="hidden"
                        />
                        <NeonButton onClick={handleIfcClick}>Upload IFC</NeonButton>
                        {originalFile && <span className="text-sm text-white mt-1">{originalFile.name}</span>}
                    </div>

                    {/* PDF Upload */}
                    <div className="flex flex-col items-center">
                        <input
                            type="file"
                            accept=".pdf"
                            onChange={handleFileUpload}
                            ref={pdfInputRef}
                            className="hidden"
                        />
                        <NeonButton onClick={handlePdfClick} color="red">
                            Upload Norm PDF
                        </NeonButton>
                        {pdfFile && <span className="text-sm text-white mt-1">{pdfFile.name}</span>}
                    </div>
                </div>

                {/* Right: Always-visible download button */}
                <NeonButton
                    color="emerald"
                    onClick={() => {
                        if (modifiedFile) {
                            const url = URL.createObjectURL(modifiedFile);
                            const a = document.createElement("a");
                            a.href = url;
                            a.download = "modified_model.ifc";
                            a.click();
                            URL.revokeObjectURL(url);
                        } else {
                            toast.error("No modifications made to download yet.");
                        }
                    }}
                >
                    Download Modified IFC
                </NeonButton>
            </div>

            {isLoading && (
                <div className="flex items-center justify-center h-full">
                    <div className="text-xl text-gray-600">Loading...</div>
                </div>
            )}

            {/* Viewer Logic */}
            {originalFile && (
                <div className="flex flex-col h-[calc(100vh-180px)]">
                    <div className="flex-1 overflow-hidden">
                        <IfcViewer key={`${originalFile.name}-original`} file={originalFile} />
                    </div>
                </div>
            )}
        </div>
    );
};

export default Result;