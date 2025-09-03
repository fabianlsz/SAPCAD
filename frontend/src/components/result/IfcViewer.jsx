import React, { useRef, useEffect } from 'react';
import { IfcViewerAPI } from 'web-ifc-viewer';
import { useGlobal } from '../../states/GlobalState';

const IfcViewer = ({ file }) => {
    const { modifiedFile, isModifiedByAI } = useGlobal(); // Access GlobalContext
    const originalContainerRef = useRef(null);
    const modifiedContainerRef = useRef(null);
    const originalViewerRef = useRef(null);
    const modifiedViewerRef = useRef(null);
    const originalModelRef = useRef(null);
    const modifiedModelRef = useRef(null);

    // Initialize original viewer
    useEffect(() => {
        const container = originalContainerRef.current;
        if (!originalViewerRef.current && container) {
            const viewer = new IfcViewerAPI({
                container,
                backgroundColor: new Uint8ClampedArray([255, 255, 255, 255]),
            });
            originalViewerRef.current = viewer;
            viewer.IFC.setWasmPath('/wasm/');
        }
    }, []);

    // Initialize modified viewer
    useEffect(() => {
        const container = modifiedContainerRef.current;
        if (!modifiedViewerRef.current && container) {
            const viewer = new IfcViewerAPI({
                container,
                backgroundColor: new Uint8ClampedArray([255, 255, 255, 255]),
            });
            modifiedViewerRef.current = viewer;
            viewer.IFC.setWasmPath('/wasm/');
        }
    }, []);

    // Load original IFC file
    useEffect(() => {
        const loadIfc = async () => {
            if (file && originalViewerRef.current) {
                try {
                    console.log("Loading original IFC file...");
                    const url = URL.createObjectURL(file);

                    // Remove previous original model
                    if (originalModelRef.current) {
                        originalViewerRef.current.context.scene.remove(originalModelRef.current);
                        originalViewerRef.current.IFC.loader.ifcManager.dispose();
                        originalModelRef.current = null;
                    }

                    const model = await originalViewerRef.current.IFC.loadIfcUrl(url);
                    model.position.set(0, 0, 0);
                    originalViewerRef.current.context.scene.add(model);
                    originalViewerRef.current.context.fitToFrame();
                    originalModelRef.current = model;

                    console.log("Original IFC file loaded successfully.");
                } catch (err) {
                    console.error("Failed to load original IFC file:", err);
                }
            }
        };

        loadIfc();
    }, [file]);

    // Load modified IFC file
    useEffect(() => {
        const loadModifiedIfc = async () => {
            if (isModifiedByAI && modifiedFile && modifiedViewerRef.current) {
                try {
                    console.log("Loading modified IFC file...");
                    const url = URL.createObjectURL(modifiedFile);

                    // Remove previous modified model
                    if (modifiedModelRef.current) {
                        modifiedViewerRef.current.context.scene.remove(modifiedModelRef.current);
                        modifiedViewerRef.current.IFC.loader.ifcManager.dispose();
                        modifiedModelRef.current = null;
                    }

                    const model = await modifiedViewerRef.current.IFC.loadIfcUrl(url);
                    model.position.set(0, 0, 0);
                    modifiedViewerRef.current.context.scene.add(model);
                    modifiedViewerRef.current.context.fitToFrame();
                    modifiedModelRef.current = model;

                    console.log("Modified IFC file loaded successfully.");
                } catch (err) {
                    console.error("Failed to load modified IFC file:", err);
                }
            } else if (modifiedViewerRef.current && modifiedModelRef.current) {
                // Clear modified viewer if no modified file exists
                modifiedViewerRef.current.context.scene.remove(modifiedModelRef.current);
                modifiedViewerRef.current.IFC.loader.ifcManager.dispose();
                modifiedModelRef.current = null;
            }
        };

        loadModifiedIfc();
    }, [modifiedFile, isModifiedByAI]);

    return (
        <div className="w-full h-full flex">
            {/* Original Viewer */}
            <div className="w-1/2 h-full flex flex-col m-1">
                <div className="text-white text-sm p-2">Original Model</div>
                <div ref={originalContainerRef} style={{ width: '100%', height: '100%' }} />
            </div>
            {/* Modified Viewer */}
            <div className="w-1/2 h-full flex flex-col m-1">
                <div className="text-white text-sm p-2">
                    {isModifiedByAI && modifiedFile ? 'Modified Model' : 'No Modifications'}
                </div>
                <div ref={modifiedContainerRef} style={{ width: '100%', height: '100%' }} />
            </div>
        </div>
    );
};

export default IfcViewer;